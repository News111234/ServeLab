"""Single-process simulated TP execution (rank threads + shared collective).

Lets the full TP code path (sharding, per-rank pools, collectives) run and be
unit-tested on any machine with plain CPU torch. Real multi-process execution
uses GlooCollective via examples/run_tp_gloo.py.
"""

import threading
from typing import Callable, Dict, List, Optional

import torch

from ..config import ModelConfig
from ..models.loader import load_model
from .layers import shard_state_dict_tp
from .tp_model import TPDecoderModel


def run_rank_threads(world_size: int, collective, fn: Callable[[int], None]):
    """Run fn(rank) in W threads; collective calls synchronize them.
    A crash on any rank aborts the collective so the others fail fast
    instead of deadlocking; the first error is re-raised."""
    errors: List[BaseException] = []
    lock = threading.Lock()
    threads = []
    for r in range(world_size):
        def target(rank=r):
            try:
                with collective.rank_scope(rank):
                    fn(rank)
            except BaseException as e:          # noqa: BLE001 - re-raised below
                with lock:
                    errors.append(e)
                if hasattr(collective, "abort"):
                    collective.abort(e)
        t = threading.Thread(target=target)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


class SimulatedTPModelGroup:
    """W sharded TPDecoderModels + per-rank pools, executed in rank threads."""

    def __init__(self, cfg: ModelConfig, world_size: int, collective,
                 device: str = "cpu", dtype: torch.dtype = torch.float32,
                 state_dict: Optional[Dict[str, torch.Tensor]] = None):
        self.cfg = cfg
        self.world_size = world_size
        self.collective = collective
        self.models = []
        for r in range(world_size):
            m = TPDecoderModel(cfg, collective, device=device, dtype=dtype)
            if state_dict is not None:
                shard = shard_state_dict_tp(
                    state_dict, r, world_size,
                    num_heads=cfg.num_heads, num_kv_heads=cfg.num_kv_heads,
                    head_dim=cfg.head_dim, vocab_size=cfg.vocab_size)
                m.load_shard(shard)
            m.eval()
            m.block_size = 16
            self.models.append(m)

    def forward_packed(self, input_ids, positions, slots, pools, metas):
        """pools: one KVCachePool per rank (built with LOCAL kv-head count).
        Returns rank-0's hidden states -- identical across ranks after the
        row-parallel all-reduces."""
        box: Dict[str, torch.Tensor] = {}

        def work(rank: int):
            h = self.models[rank].forward_packed(
                input_ids, positions, slots, metas, pools[rank])
            if rank == 0:
                box["hidden"] = h

        run_rank_threads(self.world_size, self.collective, work)
        return box["hidden"]

    def lm_head_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Full-vocab logits (vocab-parallel + all-gather); identical on all
        ranks, returned from rank 0's copy. EVERY rank must participate --
        the all-gather needs all shards."""
        box: Dict[str, torch.Tensor] = {}

        def work(rank: int):
            lg = self.models[rank].lm_head_forward(hidden)
            if rank == 0:
                box["logits"] = lg

        run_rank_threads(self.world_size, self.collective, work)
        return box["logits"]


def load_full_state(path: str) -> Dict[str, torch.Tensor]:
    _, state = load_model(path)
    return state
