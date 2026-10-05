"""Drive W SPMD engines in one process (rank threads) -- simulated TP.

Real multi-process TP uses examples/run_tp_gloo.py with GlooCollective; the
group here exercises the exact same engine code path (sharded weights, local
KV pools, collective-synced sampling) in threads, so it runs in CI.
"""

from typing import List

from ..config import CacheConfig, ModelConfig, SchedulerConfig
from .layers import SimulatedCollective
from .simulated import run_rank_threads


class SimulatedTPEngineGroup:
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        scheduler_config: SchedulerConfig,
        world_size: int,
        seed: int = 1234,
        tokenizer=None,
    ):
        from ..engine.engine import LLMEngine  # lazy: engine imports parallel

        self.world_size = world_size
        self.collective = SimulatedCollective(world_size)
        self.engines = [
            LLMEngine(model_config, cache_config, scheduler_config, seed=seed,
                      device="cpu", tokenizer=tokenizer,
                      tp_rank=r, tp_world=world_size,
                      collective=self.collective)
            for r in range(world_size)
        ]

    # requests must be added in the SAME order on every rank (SPMD contract)
    def add_request(self, *args, **kwargs):
        rids = [e.add_request(*args, **kwargs) for e in self.engines]
        return rids[0]

    def has_unfinished(self) -> bool:
        return self.engines[0].has_unfinished()

    def step(self) -> list:
        box = {}

        def work(rank: int):
            outs = self.engines[rank].step()
            if rank == 0:
                box["outputs"] = outs

        run_rank_threads(self.world_size, self.collective, work)
        return box.get("outputs", [])

    def generate(self, prompts: List, sampling_params, verbose: bool = False) -> list:
        results = {}
        rids = []
        for p in prompts:
            rids.append(self.add_request(
                prompt=p if isinstance(p, str) else None,
                prompt_token_ids=None if isinstance(p, str) else list(p),
                sampling_params=sampling_params))
        while self.has_unfinished():
            for out in self.step():
                if out is not None and out.finished:
                    results[out.request_id] = out
            if verbose:
                st = self.engines[0].stats()
                print(f"[tp-step] running={st['running']} waiting={st['waiting']}")
        return [results.get(rid) for rid in rids]
