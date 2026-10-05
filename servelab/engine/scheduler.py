"""Request scheduler: continuous batching + chunked prefill + preemption.

Torch-free: driven by an engine (or by a fake executor in tests). Policy
points, all pluggable and all paper-relevant (docs/research_roadmap.md #4):
    - admission / waiting-queue order        (fcfs | priority | sjf-predicted)
    - chunked prefill budget policy          (vLLM-v0 style vs vLLM-v1 mixed)
    - preemption victim selection + mode     (recompute; swap = roadmap)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Set

from ..config import CacheConfig, SchedulerConfig
from ..kv_cache.manager import BlockManager
from .sequence import Sequence, SequenceStatus


class OutputLengthPredictor(Protocol):
    """predict remaining output tokens for a running/waiting sequence."""

    def predict_remaining(self, seq: Sequence) -> int: ...


class SchedulingPolicy(ABC):
    name = "base"

    @abstractmethod
    def order(self, seqs: List[Sequence]) -> List[Sequence]:
        """Return seqs in scheduling preference order (first = schedule first)."""

    def preempt_order(self, seqs: List[Sequence]) -> List[Sequence]:
        """Return seqs in preemption-victim order (first = preempt first)."""
        return list(reversed(self.order(seqs)))


class FCFSPolicy(SchedulingPolicy):
    name = "fcfs"

    def order(self, seqs):
        return sorted(seqs, key=lambda s: (s.arrival_time, s.request_id))


class PriorityPolicy(FCFSPolicy):
    name = "priority"

    def order(self, seqs):
        return sorted(seqs, key=lambda s: (s.priority, s.arrival_time, s.request_id))


class SJFPredictedPolicy(SchedulingPolicy):
    """Shortest-job-first on PREDICTED remaining output length.

    Baseline for scheduling research: the predictor is imperfect, so this
    trades head-of-line blocking for prediction error -- measure TTFT/TPOT
    distributions vs FCFS on real traces. (cf. S3, response-length prediction)
    """

    name = "sjf-predicted"

    def __init__(self, predictor: OutputLengthPredictor):
        self.predictor = predictor

    def order(self, seqs):
        def key(s: Sequence):
            if s.is_prefill_done():
                remaining = self.predictor.predict_remaining(s) - s.num_output_tokens
            else:
                remaining = self.predictor.predict_remaining(s)
            return (max(1, remaining), s.arrival_time)
        return sorted(seqs, key=key)


def build_scheduler_policy(name: str, predictor: Optional[OutputLengthPredictor] = None):
    if name == "fcfs":
        return FCFSPolicy()
    if name == "priority":
        return PriorityPolicy()
    if name == "sjf-predicted":
        if predictor is None:
            raise ValueError("sjf-predicted policy requires an output-length predictor")
        return SJFPredictedPolicy(predictor)
    raise ValueError(f"unknown scheduler policy: {name}")


@dataclass
class PrefillSched:
    seq: Sequence
    num_new_tokens: int          # chunk size
    num_computed: int            # tokens already in KV before this chunk


@dataclass
class SchedulerOutput:
    prefills: List[PrefillSched] = field(default_factory=list)
    decodes: List[Sequence] = field(default_factory=list)
    preempted: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.prefills and not self.decodes

    @property
    def num_batched_tokens(self) -> int:
        return sum(p.num_new_tokens for p in self.prefills) + len(self.decodes)


class Scheduler:
    def __init__(
        self,
        sched_config: SchedulerConfig,
        cache_config: CacheConfig,
        block_manager: BlockManager,
        clock: Callable[[], float],
        eos_token_id: int = -1,
        predictor: Optional[OutputLengthPredictor] = None,
        pool=None,
    ):
        self.cfg = sched_config
        self.cache_cfg = cache_config
        self.bm = block_manager
        self.clock = clock
        self.eos_token_id = eos_token_id
        self.pool = pool                       # needed for swap-mode copy ops
        self.policy = build_scheduler_policy(sched_config.policy, predictor)
        self.waiting: List[Sequence] = []
        self.running: List[Sequence] = []

    # ---------------------------------------------------------------- intake
    def add_request(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def abort_request(self, request_id: str) -> bool:
        for i, s in enumerate(self.waiting):
            if s.request_id == request_id:
                self.waiting.pop(i)
                s.status = SequenceStatus.FINISHED_ABORTED
                return True
        for i, s in enumerate(self.running):
            if s.request_id == request_id:
                self.running.pop(i)
                self.bm.release(request_id, s.token_ids[:s.num_computed_tokens])
                s.status = SequenceStatus.FINISHED_ABORTED
                return True
        return False

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    # ------------------------------------------------------------- schedule
    def schedule(self) -> SchedulerOutput:
        now = self.clock()
        out = SchedulerOutput()
        budget = self.cfg.max_num_batched_tokens

        # --- 0) resume swapped-out sequences first (vLLM-v0 policy: they
        #        hold progress and their resume is a cheap memory copy)
        i = 0
        while i < len(self.waiting):
            seq = self.waiting[i]
            if not getattr(seq, "_swapped", False):
                i += 1
                continue
            if len(self.running) >= self.cfg.max_num_seqs:
                break
            if self.bm.swap_in(seq.request_id, seq.token_ids):
                seq._swapped = False
                self.waiting.pop(i)
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
            else:
                # fall back to recompute: forget swap state, start over
                seq._swapped = False
                seq.on_preempt_recompute()
                i += 1

        # --- 1) decodes (one token each) for prefill-complete running seqs
        can_decode = not (not self.cfg.chunked_prefill and self.waiting)
        if can_decode:
            for seq in list(self.running):
                if not seq.is_prefill_done():
                    continue
                if len(out.decodes) >= self.cfg.max_num_seqs or budget < 1:
                    break
                if seq.request_id not in self.bm.seq_blocks:
                    continue     # preempted earlier in this same loop pass
                if not self._try_decode_slot(seq, out):
                    continue
                budget -= 1

        # --- 2) in-flight chunked prefills continue first (v1-style mixed batch)
        partial = [s for s in self.running if not s.is_prefill_done()]
        for seq in self.policy.order(partial):
            if budget <= 0:
                break
            chunk = self._plan_prefill_chunk(seq, budget, out)
            if chunk is None:
                continue
            budget -= chunk

        # --- 3) admit from waiting (policy-ordered: fcfs / priority / sjf)
        self.waiting = self.policy.order(self.waiting)
        while (self.waiting
               and len(self.running) < self.cfg.max_num_seqs
               and budget > 0):
            seq = self.waiting[0]
            if getattr(seq, "_swapped", False):
                # a swapped-out sequence may ONLY re-enter via the swap-in
                # pass above (its KV lives on host; normal admission would
                # corrupt its bookkeeping) -- block admission behind it
                break
            need_blocks = (seq.get_len() + self.bm.block_size - 1) // self.bm.block_size
            if need_blocks > self.bm.num_blocks:
                self.waiting.pop(0)
                seq.status = SequenceStatus.FINISHED_ABORTED
                continue
            if self.bm.enable_prefix_caching:
                available = (self.bm.num_free_blocks() + self.bm._evictable()
                             - self.bm._reserve())
                if available < need_blocks:
                    break                                  # head-of-line blocking
            # match prefix right before scheduling (locks matched blocks)
            num_matched = 0
            if self.bm.enable_prefix_caching:
                num_matched = self.bm.maybe_match_prefix(seq.request_id, seq.token_ids)
            else:
                self.bm.seq_blocks[seq.request_id] = []
                self.bm.seq_matched_nodes[seq.request_id] = []
            # a fully cached prompt still needs one row recomputed for logits
            seq.num_computed_tokens = min(num_matched, seq.num_prompt_tokens - 1)
            seq.metrics.first_scheduled_time = seq.metrics.first_scheduled_time or now
            chunk = self._plan_prefill_chunk(seq, budget, out)
            if chunk is None:
                # admission failed this step: undo the prefix match so the
                # next round re-matches cleanly
                self.bm.undo_match(seq.request_id)
                seq.num_computed_tokens = 0
                break
            self.waiting.pop(0)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            budget -= chunk

        return out

    # ------------------------------------------------------------- helpers
    def _plan_prefill_chunk(self, seq: Sequence, budget: int,
                            out: SchedulerOutput) -> Optional[int]:
        """Plan (and pre-allocate) the next chunk for a partial prefill.
        Returns chunk size, or None if the seq was deferred / preempted.

        chunked_prefill=True clamps to the token budget (vLLM-v1 mixed batch);
        chunked_prefill=False follows the v0 semantics of scheduling the whole
        remaining prompt in one go (even if it exceeds the budget)."""
        remaining = seq.remaining_prompt_tokens()
        if self.cfg.chunked_prefill:
            chunk = max(1, min(remaining, budget))
        else:
            chunk = max(1, remaining)
        got = self.bm.allocate_slots(seq.request_id,
                                     seq.num_computed_tokens + chunk)
        if got is None:
            self._handle_alloc_failure(seq, out)
            return None
        out.prefills.append(PrefillSched(seq, chunk, seq.num_computed_tokens))
        return chunk

    def _try_decode_slot(self, seq: Sequence, out: SchedulerOutput) -> bool:
        """Reserve the KV row for this decode step; preempt on failure."""
        block = self.bm.append_slot(seq.request_id, seq.num_computed_tokens)
        if block is not None:
            out.decodes.append(seq)
            return True
        self._handle_alloc_failure(seq, out)
        return False

    def _handle_alloc_failure(self, seq: Sequence, out: SchedulerOutput) -> None:
        """Free memory by preempting the lowest-value running seq; fall back
        to preempting `seq` itself if every running seq is already scheduled
        (guarantees forward progress)."""
        scheduled_ids: Set[str] = {p.seq.request_id for p in out.prefills}
        scheduled_ids.update(s.request_id for s in out.decodes)
        for victim in self.policy.preempt_order(self.running):
            if victim.request_id in scheduled_ids:
                continue
            self._preempt(victim, out)
            return
        # self-preempt
        if seq in out.decodes:
            out.decodes.remove(seq)
        if any(p.seq is seq for p in out.prefills):
            return                       # chunk already planned & allocated
        if seq not in self.running:
            # waiting seq whose prefix match can't extend: undo the match and
            # stop admitting this step (do NOT re-insert into waiting)
            self.bm.undo_match(seq.request_id)
            seq.num_computed_tokens = 0
            return
        self._preempt(seq, out)

    def _preempt(self, seq: Sequence, out: SchedulerOutput) -> None:
        if seq in self.running:
            self.running.remove(seq)
        # swap mode: keep the computed KV on host, resume without recompute
        if self.cfg.preemption_mode == "swap" and self.pool is not None \
                and self.bm.swap_out(seq.request_id, seq.num_computed_tokens):
            seq.status = SequenceStatus.WAITING
            seq._swapped = True
            self.waiting.insert(0, seq)
            out.preempted.append(seq.request_id)
            return
        # recompute mode (default): only computed tokens have KV; the freshly
        # sampled token does not
        self.bm.release(seq.request_id, seq.token_ids[:seq.num_computed_tokens])
        seq.on_preempt_recompute()
        self.waiting.insert(0, seq)
        out.preempted.append(seq.request_id)

    # ------------------------------------------------------ post-execution
    def update_after_exec(
        self,
        sampled: Dict[str, int],
        now: Optional[float] = None,
    ) -> List[Sequence]:
        """Append sampled tokens, apply stop conditions, free finished seqs.
        Returns the sequences that finished this step."""
        now = self.clock() if now is None else now
        finished: List[Sequence] = []
        for seq_id, token in sampled.items():
            seq = self._find_running(seq_id)
            if seq is None:
                continue
            seq.append_token(token)
            if seq.metrics.first_token_time == 0.0:
                seq.metrics.first_token_time = now
            sp = seq.sampling_params
            is_eos = (token == self.eos_token_id or token in sp.stop_token_ids)
            if is_eos and not sp.ignore_eos:
                seq.status = SequenceStatus.FINISHED_EOS
            elif seq.num_output_tokens >= sp.max_tokens:
                seq.status = SequenceStatus.FINISHED_LENGTH
            if seq.is_finished():
                finished.append(seq)
        for seq in finished:
            if seq in self.running:
                self.running.remove(seq)
            self.bm.release(seq.request_id,
                            seq.token_ids[:seq.num_computed_tokens])
            seq.metrics.finished_time = now
        return finished

    def _find_running(self, seq_id: str) -> Optional[Sequence]:
        for s in self.running:
            if s.request_id == seq_id:
                return s
        return None

    # ------------------------------------------------------------- stats
    def stats(self) -> dict:
        evictable = self.bm._evictable()
        total_matched = max(1, self.bm.stat_total_match_tokens)
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "gpu_cache_usage": round(
                (self.bm.num_used_blocks() + evictable) / self.bm.num_blocks, 4),
            "prefix_hit_rate": round(self.bm.stat_prefix_hit_tokens / total_matched, 4),
            "num_preemptions_total": sum(
                s.metrics.num_preemptions for s in self.running + self.waiting),
        }
