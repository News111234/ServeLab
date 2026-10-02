"""The engine: ties scheduler + block manager + KV pool + model together.

Usage (offline, vLLM-like):
    engine = LLMEngine(model_config, cache_config, scheduler_config)
    engine.add_request(prompt_token_ids=[...], sampling_params=...)
    while engine.has_unfinished():
        outputs = engine.step()
"""

import time
from typing import Callable, Dict, List, Optional

import torch

from ..config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from ..kv_cache.manager import BlockManager
from ..kv_cache.pool import KVCachePool
from ..models.decoder import DecoderModel, SeqMeta, flat_rows
from ..models.loader import load_model
from ..parallel.layers import shard_state_dict_tp
from ..parallel.tp_model import TPDecoderModel
from .output import RequestOutput
from .sampler import Sampler
from .scheduler import Scheduler
from .sequence import Sequence
from .sampling_params import SamplingParams


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _resolve_dtype(model_cfg: ModelConfig, device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    return {"float16": torch.float16, "bfloat16": torch.bfloat16,
            "float32": torch.float32}.get(model_cfg.dtype, torch.float16)


class LLMEngine:
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        scheduler_config: SchedulerConfig,
        seed: int = 1234,
        device: str = "auto",
        tokenizer=None,
        predictor=None,
        clock: Optional[Callable[[], float]] = None,
        tp_rank: int = 0,
        tp_world: int = 1,
        collective=None,
    ):
        self.model_config = model_config
        self.cache_config = cache_config
        self.scheduler_config = scheduler_config
        self.device = _resolve_device(device)
        self.dtype = _resolve_dtype(model_config, self.device)
        self.tokenizer = tokenizer
        self.clock = clock or time.monotonic
        self.tp_rank = tp_rank
        self.tp_world = tp_world
        self.collective = collective
        if tp_world > 1:
            assert model_config.num_heads % tp_world == 0, "query heads must split evenly"
            assert model_config.num_kv_heads % tp_world == 0, "kv heads must split evenly"

        # ---- model (dense, or TP shard of it)
        if tp_world > 1:
            _, full_state = load_model(model_config.path)
            self.model = TPDecoderModel(model_config, collective,
                                        device=self.device, dtype=self.dtype)
            shard = shard_state_dict_tp(
                full_state, tp_rank, tp_world,
                num_heads=model_config.num_heads,
                num_kv_heads=model_config.num_kv_heads,
                head_dim=model_config.head_dim,
                vocab_size=model_config.vocab_size)
            self.model.load_shard(shard)
            local_kv_heads = model_config.num_kv_heads // tp_world
        else:
            self.model = DecoderModel(model_config, device=self.device, dtype=self.dtype)
            if model_config.path:
                _, state = load_model(model_config.path)
                unexpected = self.model.load_state_dict_hf(state)
                if unexpected:
                    print(f"[servelab] ignored unexpected params: {unexpected[:5]}...")
            local_kv_heads = model_config.num_kv_heads
        self.model.eval()

        # ---- KV pool + block manager (+ host swap space for swap-mode)
        swap_space = None
        if scheduler_config.preemption_mode == "swap":
            from ..kv_cache.swap import CPUSwapSpace
            swap_space = CPUSwapSpace()
        num_blocks = cache_config.num_blocks or self._default_num_blocks()
        self.num_blocks = num_blocks
        self.pool = KVCachePool(
            num_blocks=num_blocks,
            block_size=cache_config.block_size,
            num_layers=model_config.num_layers,
            num_kv_heads=local_kv_heads,
            head_dim=model_config.head_dim,
            dtype=self.dtype,
            device=self.device,
            kv_cache_dtype=cache_config.kv_cache_dtype,
        )
        self.model.block_size = cache_config.block_size
        self.bm = BlockManager(
            num_blocks=num_blocks,
            block_size=cache_config.block_size,
            enable_prefix_caching=cache_config.enable_prefix_caching,
            eviction_policy=cache_config.eviction_policy,
            watermark=cache_config.watermark,
            swap_space=swap_space,
            pool=self.pool,
        )

        # ---- scheduler + sampler
        self.scheduler = Scheduler(
            scheduler_config, cache_config, self.bm, self.clock,
            eos_token_id=model_config.eos_token_id, predictor=predictor,
            pool=self.pool,
        )
        self.sampler = Sampler(seed)
        self._req_counter = 0

    # -------------------------------------------------------------- intake
    def add_request(
        self,
        prompt: Optional[str] = None,
        sampling_params: Optional[SamplingParams] = None,
        prompt_token_ids: Optional[List[int]] = None,
        request_id: Optional[str] = None,
        priority: int = 0,
    ) -> str:
        if prompt_token_ids is None:
            if self.tokenizer is None or prompt is None:
                raise ValueError("need prompt_token_ids, or a tokenizer + prompt")
            prompt_token_ids = self.tokenizer.encode(prompt)
        self._req_counter += 1
        request_id = request_id or f"req-{self._req_counter}"
        seq = Sequence(
            request_id=request_id,
            prompt_token_ids=list(prompt_token_ids),
            sampling_params=sampling_params or SamplingParams(),
            priority=priority,
            arrival_time=self.clock(),
        )
        self.scheduler.add_request(seq)
        return request_id

    def abort_request(self, request_id: str) -> bool:
        return self.scheduler.abort_request(request_id)

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    # ------------------------------------------------------------- stepping
    def step(self) -> List[RequestOutput]:
        """One scheduler step: build a mixed batch, run forward, sample,
        update state. Returns outputs for sequences finished in this step."""
        sched_out = self.scheduler.schedule()
        if sched_out.is_empty():
            return []

        metas: List[SeqMeta] = []
        tokens: List[int] = []
        positions: List[int] = []
        slots: List[int] = []
        logit_rows: List[int] = []
        logit_seqs: List[Sequence] = []

        for ps in sched_out.prefills:
            seq, start, n = ps.seq, ps.num_computed, ps.num_new_tokens
            off = len(tokens)
            tokens.extend(seq.token_ids[start:start + n])
            positions.extend(range(start, start + n))
            slots.extend(self.bm.slot_mapping(seq.request_id, start, n))
            metas.append(SeqMeta(
                offset=off, q_len=n, ctx_len=start + n,
                block_table=self.bm.get_block_table(seq.request_id)))
            if start + n == seq.num_prompt_tokens:
                logit_rows.append(off + n - 1)
                logit_seqs.append(seq)
            seq.num_computed_tokens = start + n

        for seq in sched_out.decodes:
            off = len(tokens)
            last = seq.get_len() - 1
            tokens.append(seq.token_ids[last])
            positions.append(last)
            slots.append(self.bm.slot_for_token(seq.request_id, last))
            metas.append(SeqMeta(
                offset=off, q_len=1, ctx_len=seq.get_len(),
                block_table=self.bm.get_block_table(seq.request_id)))
            logit_rows.append(off)
            logit_seqs.append(seq)
            # this token's KV is written by the forward below
            seq.num_computed_tokens = seq.get_len()

        # ---- forward
        device = self.device
        ids = torch.tensor(tokens, dtype=torch.long, device=device)
        pos = torch.tensor(positions, dtype=torch.long, device=device)
        slot_t = torch.tensor(slots, dtype=torch.long, device=device)
        with torch.no_grad():
            hidden = self.model.forward_packed(ids, pos, slot_t, metas, self.pool)
            logits = self.model.lm_head_forward(hidden[logit_rows]).float()

        # SPMD sampling: rank 0 samples, every rank uses the same tokens
        # (matches vLLM's driver-only sampler design)
        if self.tp_world > 1:
            if self.tp_rank == 0:
                tokens = self.sampler.sample(
                    logits, [s.sampling_params for s in logit_seqs])
                payload = {s.request_id: t for s, t in zip(logit_seqs, tokens)}
            else:
                payload = None
            payload = self.collective.sync_object(payload, src=0)
            sampled_tokens = [payload[s.request_id] for s in logit_seqs]
        else:
            sampled_tokens = self.sampler.sample(
                logits, [s.sampling_params for s in logit_seqs])

        sampled: Dict[str, int] = {}
        for seq, tok in zip(logit_seqs, sampled_tokens):
            sampled[seq.request_id] = tok
        finished = self.scheduler.update_after_exec(sampled, now=self.clock())

        return [RequestOutput.from_sequence(s, detokenize=self._detokenize)
                for s in finished]

    def _detokenize(self, token_ids: List[int]) -> str:
        if self.tokenizer is None:
            return ""
        return self.tokenizer.decode(token_ids)

    # ------------------------------------------------------------- generate
    def generate(self, prompts: List, sampling_params: Optional[SamplingParams] = None,
                 verbose: bool = False) -> List[RequestOutput]:
        """Offline batch generation: prompts may be str (needs tokenizer) or
        lists of token ids. Returns outputs in input order."""
        params = sampling_params or SamplingParams()
        ids = []
        for p in prompts:
            rid = self.add_request(
                prompt=p if isinstance(p, str) else None,
                prompt_token_ids=None if isinstance(p, str) else list(p),
                sampling_params=params,
            )
            ids.append(rid)
        results: Dict[str, RequestOutput] = {}
        while self.has_unfinished():
            for out in self.step():
                results[out.request_id] = out
            if verbose:
                st = self.scheduler.stats()
                print(f"[step] running={st['running']} waiting={st['waiting']} "
                      f"cache={st['gpu_cache_usage']:.0%}")
        return [results.get(i) for i in ids]

    # --------------------------------------------------------------- misc
    def _default_num_blocks(self) -> int:
        cfg, cc = self.model_config, self.cache_config
        if self.device == "cuda":
            free_bytes = torch.cuda.mem_get_info()[0]
            avail = free_bytes * cc.gpu_memory_utilization
        else:
            # host runs are for correctness/research, not throughput: a
            # modest fixed KV pool keeps startup instant
            return 4096
        store_bytes = 1 if cc.kv_cache_dtype in ("int8", "fp8") else (
            4 if self.dtype == torch.float32 else 2)
        scale_bytes = 4 if cc.kv_cache_dtype in ("int8", "fp8") else 0
        per_row = (2 * cfg.num_kv_heads * cfg.head_dim * store_bytes
                   + 2 * cfg.num_kv_heads * scale_bytes)
        per_block = per_row * cc.block_size * cfg.num_layers
        return max(16, int(avail / per_block))

    def stats(self) -> dict:
        return {
            **self.scheduler.stats(),
            "num_blocks": self.num_blocks,
            "block_size": self.cache_config.block_size,
            "kv_cache_dtype": self.pool.quant.mode,
            "device": self.device,
        }
