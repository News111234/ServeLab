"""Multi-replica serving cluster simulator (torch-free, event-stepped).

Why a simulator: scheduling/routing/eviction policy research needs thousands
of what-if runs over real traces; a trace-driven simulator with an
analytical step-time model iterates in seconds instead of GPU-days. The
RadixCache / eviction-policy implementations here are the SAME classes the
real engine uses, so policy code moves between sim and engine unchanged
(that is the whole point of the design).

PD-disaggregation mode: replicas split into prefill / decode pools; KV is
"transferred" at a modeled bandwidth between them (cf. DistServe, Mooncake).
"""

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..engine.sampling_params import SamplingParams
from ..engine.sequence import Sequence, SequenceStatus
from ..kv_cache.policies import build_eviction_policy
from ..kv_cache.radix import RadixCache
from .model_cost import GpuSpec, ModelSpec, step_time
from .predictor import OutputLengthPredictor
from .router import Router
from .trace import TraceRequest


@dataclass
class ReplicaConfig:
    max_running_seqs: int = 32
    max_total_tokens: int = 65536       # KV capacity in tokens
    max_num_batched_tokens: int = 1024
    block_size: int = 16
    chunked_prefill: bool = True


@dataclass
class SimConfig:
    num_replicas: int = 2
    gpu: GpuSpec = field(default_factory=lambda: GpuSpec("A100", 156e12, 1.4e12))
    model: ModelSpec = field(default_factory=ModelSpec)
    replica: ReplicaConfig = field(default_factory=ReplicaConfig)
    router: str = "least-loaded"
    predictor: str = "online-mean"
    eviction_policy: str = "lru"
    # PD disaggregation
    pd_mode: bool = False
    pd_transfer_bw: float = 25e9        # bytes/s (e.g. RDMA)
    pd_transfer_latency_s: float = 0.0005


@dataclass
class ReplicaState:
    """Router-visible snapshot of one replica."""
    replica_id: int
    num_running: int = 0
    num_queued: int = 0
    predicted_work: float = 0.0
    cached_prefixes: set = field(default_factory=set)


class Replica:
    def __init__(self, replica_id: int, cfg: SimConfig,
                 predictor: OutputLengthPredictor, role: str = "mixed"):
        self.id = replica_id
        self.cfg = cfg
        self.rc = cfg.replica
        self.role = role                      # "mixed" | "prefill" | "decode"
        self.predictor = predictor
        self.radix = RadixCache(self.rc.block_size,
                                build_eviction_policy(cfg.eviction_policy))
        self.queue: deque = deque()
        self.running: List[Sequence] = []
        self.busy_until = 0.0
        self.stat_prefill_tokens = 0
        self.stat_decode_tokens = 0
        self.stat_preemptions = 0
        self.stat_matched_tokens = 0
        self.stat_prompt_tokens = 0
        self.stat_finished = 0

    # ------------------------------------------------------------- accounting
    def _private_blocks(self, seq: Sequence) -> int:
        matched = getattr(seq, "_matched_tokens", 0)
        return -(-max(0, seq.get_len() - matched) // self.rc.block_size)

    @property
    def used_blocks(self) -> int:
        # matched prefix blocks live inside the radix tree; running seqs own
        # only their private blocks
        return sum(self._private_blocks(s) for s in self.running) + self.radix.num_blocks

    @property
    def capacity_blocks(self) -> int:
        return self.rc.max_total_tokens // self.rc.block_size

    # ------------------------------------------------------------- admission
    def _try_admit(self, now: float) -> None:
        while self.queue and len(self.running) < self.rc.max_running_seqs:
            seq = self.queue[0]
            if self.role == "decode":
                if not self._admit_transferred(seq):
                    break
                continue
            if seq.get_len() > self.rc.max_total_tokens:
                self.queue.popleft()
                seq.status = SequenceStatus.FINISHED_ABORTED
                continue
            hit_tokens = 0
            if seq.num_computed_tokens == 0:
                blocks, nodes, hit_tokens = self.radix.match_prefix(seq.token_ids)
                seq._matched_nodes = nodes
                seq._matched_tokens = hit_tokens
            seq.num_computed_tokens = min(hit_tokens, seq.num_prompt_tokens - 1)
            if not self._ensure_capacity(seq):
                self.radix.unlock(getattr(seq, "_matched_nodes", []))
                seq._matched_nodes, seq._matched_tokens = [], 0
                seq.num_computed_tokens = 0
                break                                     # retry later
            self.stat_prompt_tokens += seq.num_prompt_tokens
            self.stat_matched_tokens += seq._matched_tokens
            self.queue.popleft()
            self.running.append(seq)

    def _admit_transferred(self, seq: Sequence) -> bool:
        """PD mode: KV arrived from a prefill replica; allocate its blocks."""
        if seq.num_prompt_tokens > self.rc.max_total_tokens:
            self.queue.popleft()
            seq.status = SequenceStatus.FINISHED_ABORTED
            return True
        seq._matched_nodes, seq._matched_tokens = [], 0
        # the whole prompt KV arrived with the transfer, so the sequence is
        # prefill-complete on this replica and starts decoding immediately
        seq.num_computed_tokens = seq.num_prompt_tokens
        if not self._ensure_capacity(seq):
            return False                                  # retry when memory frees
        self.queue.popleft()
        self.running.append(seq)
        return True

    def _ensure_capacity(self, seq: Sequence, extra_tokens: int = 0) -> bool:
        target = seq.get_len() + extra_tokens
        matched = getattr(seq, "_matched_tokens", 0)
        need = -(-max(0, target - matched) // self.rc.block_size)
        while self.used_blocks + need > self.capacity_blocks:
            if not self.radix.evict(1):
                return False
        return True

    # ------------------------------------------------------------------ step
    def has_work(self) -> bool:
        return bool(self.queue or self.running)

    def do_step(self, now: float, finished_out: List) -> float:
        """Run one engine step at `now`; returns the step duration."""
        self._try_admit(now)
        budget = self.rc.max_num_batched_tokens
        prefill_tokens = 0
        decode_seqs: List[Sequence] = []
        prefill_chunks: List[Tuple[Sequence, int]] = []
        kv_tokens = 0

        for seq in list(self.running):
            if seq.is_prefill_done():
                if budget >= 1 and self._ensure_capacity(seq, extra_tokens=1):
                    decode_seqs.append(seq)
                    budget -= 1
                else:
                    self._preempt(seq)
        if self.role != "decode":
            for seq in [s for s in self.running if not s.is_prefill_done()]:
                remaining = seq.remaining_prompt_tokens()
                chunk = min(remaining, budget) if self.rc.chunked_prefill else remaining
                chunk = max(1, min(chunk, budget))
                if not self._ensure_capacity(seq, extra_tokens=chunk):
                    self._preempt(seq)
                    continue
                prefill_chunks.append((seq, chunk))
                prefill_tokens += chunk
                budget -= chunk
                if budget <= 0:
                    break

        for seq in decode_seqs:
            kv_tokens += seq.get_len()
        for seq, _ in prefill_chunks:
            kv_tokens += seq.get_len()

        dt = step_time(prefill_tokens, len(decode_seqs), kv_tokens,
                       self.cfg.model, self.cfg.gpu)
        end = now + dt
        self.stat_prefill_tokens += prefill_tokens
        self.stat_decode_tokens += len(decode_seqs)

        # ---- progress sequences
        handed_off: List[Sequence] = []
        for seq, chunk in prefill_chunks:
            seq.num_computed_tokens += chunk
            if seq.is_prefill_done() and seq.metrics.first_token_time == 0.0:
                seq.metrics.first_token_time = end
                if self.role == "prefill":                # PD: hand off
                    handed_off.append(seq)
        for seq in handed_off:
            self.running.remove(seq)
            self._release_blocks(seq)
            self._pending_handoff.append((seq, end))
        for seq in decode_seqs:
            seq.append_token(1)                           # token id irrelevant
            if seq.metrics.first_token_time == 0.0:
                seq.metrics.first_token_time = end
            if seq.num_output_tokens >= seq.target_output_tokens:
                seq.status = SequenceStatus.FINISHED_LENGTH
                self.running.remove(seq)
                self._release_blocks(seq)
                self.stat_finished += 1
                self.predictor.observe(seq)
                finished_out.append((seq, end))
        self.busy_until = end
        return dt

    # ------------------------------------------------------------- PD handoff
    def _pending_handoff_list(self) -> List[Tuple[Sequence, float]]:
        return self._pending_handoff

    def _release_blocks(self, seq: Sequence) -> None:
        self.radix.unlock(getattr(seq, "_matched_nodes", []))
        num_full = seq.get_len() // self.rc.block_size
        self.radix.insert(seq.token_ids[: num_full * self.rc.block_size],
                          _pseudo_block_ids(seq, num_full))
        seq._matched_nodes, seq._matched_tokens = [], 0

    def _preempt(self, seq: Sequence) -> None:
        if seq in self.running:
            self.running.remove(seq)
        self.radix.unlock(getattr(seq, "_matched_nodes", []))
        # only KV that was actually computed exists -- never insert blocks
        # for uncomputed tokens (phantom reuse would corrupt hit accounting)
        num_full = min(seq.get_len(), seq.num_computed_tokens) // self.rc.block_size
        self.radix.insert(seq.token_ids[: num_full * self.rc.block_size],
                          _pseudo_block_ids(seq, num_full))
        seq.on_preempt_recompute()
        seq._matched_nodes, seq._matched_tokens = [], 0
        self.queue.appendleft(seq)
        self.stat_preemptions += 1

    # ----------------------------------------------------------------- state
    def snapshot(self) -> ReplicaState:
        st = ReplicaState(self.id)
        st.num_running = len(self.running)
        st.num_queued = len(self.queue)
        st.predicted_work = float(
            sum(self.predictor.predict_remaining(s) for s in self.running)
            + sum(self.predictor.predict_remaining(s) for s in self.queue))
        st.cached_prefixes = {s.prefix_key for s in self.running if s.prefix_key}
        return st


def _pseudo_block_ids(seq: Sequence, num_full: int) -> List[int]:
    # The simulator has no physical pool; ids only feed dup detection in the
    # radix tree (same content inserted twice keeps the first id).
    base = id(seq) * 100003
    return [base + i for i in range(num_full)]


@dataclass
class RequestResult:
    req_id: str
    arrival_time: float
    first_token_time: float
    finish_time: float
    prompt_tokens: int
    output_tokens: int
    replica_id: int
    preemptions: int = 0


class ClusterSimulator:
    """Event-stepped simulation over a trace."""

    def __init__(self, config: SimConfig, predictor: Optional[OutputLengthPredictor] = None,
                 router: Optional[Router] = None):
        self.cfg = config
        self.predictor = predictor
        if router is None and not config.pd_mode:
            from .router import build_router
            router = build_router(config.router, config.num_replicas, predictor)
        self.router = router
        if config.pd_mode:
            n = config.num_replicas
            self.replicas = [Replica(i, config, predictor, role="prefill")
                             for i in range(n // 2)]
            self.replicas += [Replica(n // 2 + i, config, predictor, role="decode")
                              for i in range(n - n // 2)]
        else:
            self.replicas = [Replica(i, config, predictor)
                             for i in range(config.num_replicas)]
        for r in self.replicas:
            r._pending_handoff = []
        self.results: List[RequestResult] = []

    def run(self, trace: List[TraceRequest], horizon_s: float = 1e9,
            verbose: bool = False) -> List[RequestResult]:
        trace = sorted(trace, key=lambda r: r.arrival_time)
        events: List[Tuple[float, int, str, object]] = []
        for idx, req in enumerate(trace):
            heapq.heappush(events, (req.arrival_time, idx, "arrival", req))
        for r in self.replicas:
            heapq.heappush(events, (0.0, -1 - r.id, "replica", r))

        while events:
            now, _, kind, payload = heapq.heappop(events)
            if now > horizon_s:
                break
            if kind == "arrival":
                seq = self._to_seq(payload)
                if self.cfg.pd_mode:
                    p, d = self._pick_pd_pair()
                    seq._decode_replica = d
                    self.replicas[p].queue.append(seq)
                    heapq.heappush(events, (now, -1 - self.replicas[p].id,
                                            "replica", self.replicas[p]))
                else:
                    states = [r.snapshot() for r in self.replicas]
                    rid = self.router.route(seq, states)
                    self.replicas[rid].queue.append(seq)
                    heapq.heappush(events, (now, -1 - rid, "replica", self.replicas[rid]))
            elif kind == "transfer":
                seq = payload
                self.replicas[seq._decode_replica].queue.append(seq)
                heapq.heappush(events, (now, -1 - self.replicas[seq._decode_replica].id,
                                        "replica", self.replicas[seq._decode_replica]))
            else:  # replica step
                replica = payload
                if replica.busy_until > now + 1e-9:
                    # stale duplicate: a live event at busy_until is already
                    # queued by the step that set it -- dropping avoids the
                    # O(duplicates) reschedule cascade
                    continue
                if not replica.has_work():
                    continue
                finished = []
                replica.do_step(now, finished)
                for seq, end in finished:
                    self.results.append(RequestResult(
                        req_id=seq.request_id,
                        arrival_time=seq.arrival_time,
                        first_token_time=seq.metrics.first_token_time,
                        finish_time=end,
                        prompt_tokens=seq.num_prompt_tokens,
                        output_tokens=seq.num_output_tokens,
                        replica_id=replica.id,
                        preemptions=seq.metrics.num_preemptions,
                    ))
                # PD: queue KV transfers produced by this step immediately
                for seq, ready_at in list(replica._pending_handoff):
                    replica._pending_handoff.remove((seq, ready_at))
                    self._queue_transfer(seq, ready_at, events)
                # the stepped replica gets its next event; kick any IDLE
                # replica with work (busy replicas already have live events)
                heapq.heappush(events, (replica.busy_until, -1 - replica.id,
                                        "replica", replica))
                for r in self.replicas:
                    if r is not replica and r.has_work() \
                            and r.busy_until <= now + 1e-9:
                        heapq.heappush(events, (now, -1 - r.id, "replica", r))
                if verbose and len(self.results) % 500 == 0 and self.results:
                    print(f"  [sim] t={now:8.1f}s finished={len(self.results)}")
        return self.results

    def _queue_transfer(self, seq: Sequence, ready_at: float, events) -> None:
        # KV transfer time modeled from prefill completion; hand the request
        # to its decode replica after the modeled copy duration.
        dt = seq.get_len() * self.cfg.model.kv_bytes_per_token / self.cfg.pd_transfer_bw \
            + self.cfg.pd_transfer_latency_s
        heapq.heappush(events, (ready_at + dt, id(seq) % 100000, "transfer", seq))

    def _to_seq(self, req: TraceRequest) -> Sequence:
        seq = Sequence(
            request_id=req.req_id,
            prompt_token_ids=req.pseudo_tokens(),
            sampling_params=SamplingParams(max_tokens=req.output_tokens,
                                           ignore_eos=True, temperature=0.0),
            priority=req.priority,
            arrival_time=req.arrival_time,
        )
        seq.prefix_key = req.prefix_key
        seq.target_output_tokens = req.output_tokens
        return seq

    def _pick_pd_pair(self) -> Tuple[int, int]:
        prefills = [r for r in self.replicas if r.role == "prefill"]
        decodes = [r for r in self.replicas if r.role == "decode"]
        p = min(prefills, key=lambda r: len(r.queue) + len(r.running)).id
        d = min(decodes, key=lambda r: len(r.queue) + len(r.running)).id
        return p, d
