"""KV cache subsystem.

Torch-free modules (radix tree, eviction policies, block manager) are imported
eagerly; torch-backed modules (pool, quantizer) are optional at import time so
the simulator and policy research can run on a machine without torch.
"""

from .policies import (
    EvictionPolicy, LRUEviction, LFUEviction, CostAwareEviction,
    build_eviction_policy,
)
from .radix import RadixCache, RadixNode
from .manager import BlockManager, AllocStatus

__all__ = [
    "EvictionPolicy", "LRUEviction", "LFUEviction", "CostAwareEviction",
    "build_eviction_policy", "RadixCache", "RadixNode",
    "BlockManager", "AllocStatus",
]

# torch-backed pieces, import on demand:
#   from servelab.kv_cache.pool import KVCachePool
#   from servelab.kv_cache.quant import KVQuantizer
