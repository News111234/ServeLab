"""CPU swap space for preempted sequences (vLLM v0 "swap" preemption).

Recompute-mode preemption throws away KV and re-runs prefill (cheap here
because the radix cache recovers the prefix). Swap-mode instead copies the
sequence's private KV blocks to host memory and copies them back on resume
-- paying PCIe traffic instead of recomputation. Which mode wins is a
workload question, cf. roadmap #4.
"""

from typing import Dict, List, Tuple


from .pool import KVCachePool


class CPUSwapSpace:
    """seq_id -> (block_ids, per-layer CPU copies of those rows)."""

    def __init__(self):
        self._buffers: Dict[str, Tuple[List[int], List]] = {}

    def __len__(self) -> int:
        return len(self._buffers)

    def __contains__(self, seq_id: str) -> bool:
        return seq_id in self._buffers

    def swap_out(self, pool: KVCachePool, seq_id: str,
                 block_ids: List[int]) -> None:
        rows = pool.rows_per_block(block_ids)
        layers = []
        for l in range(pool.num_layers):
            k, v = pool.read(l, rows)          # float32 host copies
            layers.append((k, v))
        self._buffers[seq_id] = (list(block_ids), layers)

    def swap_in(self, pool: KVCachePool, seq_id: str,
                new_block_ids: List[int]) -> None:
        old_ids, layers = self._buffers[seq_id]
        assert len(new_block_ids) == len(old_ids), "swap-in must reuse layout"
        rows = pool.rows_per_block(new_block_ids)
        for l, (k, v) in enumerate(layers):
            pool.write(l, rows.tolist(), k, v)
        del self._buffers[seq_id]

    def drop(self, seq_id: str) -> None:
        self._buffers.pop(seq_id, None)
