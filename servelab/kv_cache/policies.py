"""Pluggable eviction policies for the radix prefix cache.

This module is the first paper hook: the *policy* decides which cached blocks
to drop when the KV pool is full. The default (LRU, as in SGLang/vLLM v1)
ignores how expensive each cached subtree is to recompute and how likely it is
to be requested again -- both are attackable with a workload-aware policy.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:                       # avoid policies <-> radix import cycle
    from .radix import RadixNode


class EvictionPolicy(ABC):
    """Select eviction victims from unlocked leaves of the radix tree."""

    name = "base"

    def __init__(self):
        # bookkeeping: block_id -> score/order; maintained on touch/insert
        self._order: "OrderedDict[int, Tuple[float, float]]" = OrderedDict()

    def on_touch(self, node: RadixNode, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        self._order.pop(node.block_id, None)
        self._order[node.block_id] = (now, float(node.hits))

    def on_insert(self, node: RadixNode, now: Optional[float] = None) -> None:
        self.on_touch(node, now)

    def on_remove(self, block_id: int) -> None:
        self._order.pop(block_id, None)

    @abstractmethod
    def pick_victims(self, num_needed: int, now: Optional[float] = None) -> List[int]:
        """Return up to num_needed evictable block ids, best victim first."""

    def evictable_count(self) -> int:
        return len(self._order)


class LRUEviction(EvictionPolicy):
    """Least-recently-used leaf first. Default in SGLang / vLLM v1."""

    name = "lru"

    def pick_victims(self, num_needed: int, now: Optional[float] = None) -> List[int]:
        out = []
        for block_id in self._order:          # OrderedDict: oldest first
            out.append(block_id)
            if len(out) >= num_needed:
                break
        return out


class LFUEviction(EvictionPolicy):
    """Least-frequently-used first (ties broken by recency)."""

    name = "lfu"

    def pick_victims(self, num_needed: int, now: Optional[float] = None) -> List[int]:
        items = sorted(
            self._order.items(),
            key=lambda kv: (kv[1][1], kv[1][0]),   # (hits asc, time asc)
        )
        return [block_id for block_id, _ in items[:num_needed]]


class CostAwareEviction(EvictionPolicy):
    """Cost-aware eviction: value = recency * frequency, victims = lowest value.

    Baseline for paper-direction #1: replace the hand-crafted score with a
    learned / workload-model-driven value (e.g. expected hit probability *
    recompute cost * TTL), or an admission policy on top (don't cache
    one-shot mega-prefixes at all).
    """

    name = "costaware"

    def __init__(self, recency_weight: float = 0.7, freq_weight: float = 0.3):
        super().__init__()
        self.recency_weight = recency_weight
        self.freq_weight = freq_weight

    def pick_victims(self, num_needed: int, now: Optional[float] = None) -> List[int]:
        now = time.monotonic() if now is None else now
        scored = []
        for block_id, (t, hits) in self._order.items():
            recency = 1.0 / (1.0 + now - t)          # decay
            freq = 1.0 / (1.0 + hits)
            score = self.recency_weight * recency + self.freq_weight * freq
            scored.append((score, block_id))
        scored.sort()
        return [b for _, b in scored[:num_needed]]


_REGISTRY = {
    "lru": LRUEviction,
    "lfu": LFUEviction,
    "costaware": CostAwareEviction,
}


def build_eviction_policy(name: str) -> EvictionPolicy:
    name = (name or "lru").lower()
    if name not in _REGISTRY:
        raise ValueError(f"unknown eviction policy '{name}', choose from {list(_REGISTRY)}")
    return _REGISTRY[name]()
