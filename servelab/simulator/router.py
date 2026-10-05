"""Request routing policies for multi-replica clusters.

Paper-relevant axis (roadmap #4): cache-aware vs load-aware vs predictive
routing trade-offs, cf. SGLang cache-aware router, Preble, "KV-aware
routing". All routers only see router-visibility state (queue depths, prefix
keys, predictions) -- never per-request internals, keeping the comparison
fair.
"""

import hashlib
from abc import ABC, abstractmethod
from typing import List

from ..engine.sequence import Sequence


class Router(ABC):
    name = "base"

    def __init__(self, num_replicas: int):
        self.num_replicas = num_replicas
        self._rr = 0

    @abstractmethod
    def route(self, seq: Sequence, replicas: List["ReplicaState"]) -> int:  # noqa: F821
        ...

    def _round_robin(self) -> int:
        r = self._rr
        self._rr = (self._rr + 1) % self.num_replicas
        return r


class RoundRobinRouter(Router):
    name = "round-robin"

    def route(self, seq, replicas):
        return self._round_robin()


class LeastLoadedRouter(Router):
    """Fewest running+queued requests (classic load balancing)."""
    name = "least-loaded"

    def route(self, seq, replicas):
        return min(range(len(replicas)),
                   key=lambda r: (replicas[r].num_running + replicas[r].num_queued,
                                  replicas[r].predicted_work))


class ShortestQueueRouter(LeastLoadedRouter):
    name = "shortest-queue"


class PrefixAffinityRouter(Router):
    """Consistent-hash on the request's prefix key -> same prefix always goes
    to the same replica (maximizes prefix cache hits; may imbalance load).
    Baseline for KV-aware routing research (cf. SGLang cache-aware routing)."""
    name = "prefix-affinity"

    def __init__(self, num_replicas: int, fallback: str = "least-loaded"):
        super().__init__(num_replicas)
        self.fallback = fallback

    def route(self, seq, replicas):
        if getattr(seq, "prefix_key", ""):
            h = int(hashlib.md5(seq.prefix_key.encode()).hexdigest(), 16)
            return h % self.num_replicas
        return LeastLoadedRouter(self.num_replicas).route(seq, replicas)


class PredictiveRouter(Router):
    """Pick the replica minimizing predicted completion time:
        score = predicted_work(replica) + own_predicted_tokens
    where predicted_work accumulates predicted remaining decode tokens of
    queued/running requests. Combine with prefix affinity via a bonus for the
    replica already caching the prefix (paper hook: tune the weight)."""
    name = "predictive"

    def __init__(self, num_replicas: int, predictor,
                 prefix_bonus_tokens: int = 512):
        super().__init__(num_replicas)
        self.predictor = predictor
        self.prefix_bonus_tokens = prefix_bonus_tokens

    def route(self, seq, replicas):
        own = max(1, self.predictor.predict_remaining(seq))
        best, best_score = 0, None
        for r, st in enumerate(replicas):
            score = st.predicted_work + own
            if getattr(seq, "prefix_key", "") and seq.prefix_key in st.cached_prefixes:
                score -= self.prefix_bonus_tokens
            if best_score is None or score < best_score:
                best, best_score = r, score
        return best


def build_router(name: str, num_replicas: int, predictor=None) -> Router:
    registry = {
        "round-robin": RoundRobinRouter,
        "least-loaded": LeastLoadedRouter,
        "shortest-queue": ShortestQueueRouter,
        "prefix-affinity": PrefixAffinityRouter,
    }
    if name == "predictive":
        if predictor is None:
            raise ValueError("predictive router requires a predictor")
        return PredictiveRouter(num_replicas, predictor)
    if name not in registry:
        raise ValueError(f"unknown router '{name}', choose from {list(registry)}")
    return registry[name](num_replicas)
