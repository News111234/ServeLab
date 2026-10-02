"""KV cache offloading (LMCache-style hierarchical memory).

Reference implementation: move evicted (but potentially reusable) KV blocks
from GPU to host memory, restore them on a later prefix hit. The admission
policy (WHICH blocks to offload, WHEN to prefetch) is deliberately pluggable
-- it is one of the paper directions (docs/research_roadmap.md #2), cf.
LMCache, AttentionStore, Mooncake.

The offloader works on flat row ranges of the KV pool and is independent of
the engine so it can be unit-tested and reused by the simulator's cost model.
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from .pool import KVCachePool


class OffloadPolicy(ABC):
    """Decide whether a block leaving the GPU cache should be kept on host."""

    @abstractmethod
    def admit(self, key: Tuple[int, ...], num_blocks: int, now: float) -> bool:
        """key: block-hash chain prefix identifying the KV segment."""


class OffloadEverything(OffloadPolicy):
    """Naive baseline: keep every evicted segment (bounded by capacity)."""

    def admit(self, key, num_blocks, now) -> bool:
        return True


class CostAwareOffloadPolicy(OffloadPolicy):
    """Admit only if recompute cost (prompt tokens to redo) exceeds a host
    read amortization threshold. Paper hook: replace the threshold with a
    learned model of future reuse probability (workload-aware admission)."""

    def __init__(self, min_tokens: int = 256):
        self.min_tokens = min_tokens

    def admit(self, key, num_blocks, now) -> bool:
        return num_blocks * 16 >= self.min_tokens   # assumes block_size 16


class CPUKVOffloader:
    """host-memory KV store keyed by block content hash (radix node key)."""

    def __init__(self, capacity_blocks: int, policy: Optional[OffloadPolicy] = None):
        self.capacity_blocks = capacity_blocks
        self.policy = policy or OffloadEverything()
        # key -> (token_ids, per-layer (k, v) float32 host tensors)
        self._store: "OrderedDict[int, Tuple[List[int], List]]" = OrderedDict()
        self.stat_offload_blocks = 0
        self.stat_restore_hits = 0

    def __len__(self) -> int:
        return self._num_blocks

    def __contains__(self, key: int) -> bool:
        return key in self._store

    @property
    def _num_blocks(self) -> int:
        return sum(len(v[1]) for v in self._store.values())

    def put(self, key: int, token_ids: List[int], pool: KVCachePool,
            block_ids: List[int], now: float = 0.0) -> bool:
        if not self.policy.admit((key,), len(block_ids), now):
            return False
        while self._num_blocks + len(block_ids) > self.capacity_blocks \
                and self._store:
            self._store.popitem(last=False)      # FIFO eviction on host
        tensors = []
        rows = pool.rows_per_block(block_ids)
        for layer in range(pool.num_layers):
            k, v = pool.read(layer, rows)
            tensors.append((k.pin_memory() if k.is_cuda else k.clone(),
                            v.pin_memory() if v.is_cuda else v.clone()))
        self._store[key] = (list(token_ids), tensors)
        self.stat_offload_blocks += len(block_ids)
        return True

    def get(self, key: int) -> Optional[Tuple[List[int], List]]:
        item = self._store.get(key)
        if item is None:
            return None
        self._store.move_to_end(key)
        self.stat_restore_hits += 1
        return item

    def remove(self, key: int) -> None:
        self._store.pop(key, None)
