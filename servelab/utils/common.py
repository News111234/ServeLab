"""Small torch-free helpers shared across modules."""

import hashlib
import struct
from typing import Sequence, Tuple


def hash_block_tokens(parent_hash: int, block_token_ids: Sequence[int]) -> int:
    """Content hash of a full KV block, chained to its parent.

    Same construction as vLLM v1 / SGLang: two identical prefixes always map to
    the same chain of block hashes, which is what makes prefix reuse safe.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(struct.pack("<Q", parent_hash & 0xFFFFFFFFFFFFFFFF))
    h.update(struct.pack(f"<{len(block_token_ids)}q", *block_token_ids))
    return int.from_bytes(h.digest(), "little")


def hash_prompt_blocks(
    token_ids: Sequence[int], block_size: int
) -> Tuple[int, Tuple[int, ...]]:
    """Return (hash of the last full block, full chain of block hashes)."""
    num_full = len(token_ids) // block_size
    chain = []
    parent = 0
    for i in range(num_full):
        blk = token_ids[i * block_size:(i + 1) * block_size]
        parent = hash_block_tokens(parent, blk)
        chain.append(parent)
    return parent, tuple(chain)


def pct(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (torch-free, deterministic)."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q / 100.0 * (len(s) - 1)))))
    return s[k]


class FakeClock:
    """Monotonic clock that can be advanced manually (tests / simulator)."""

    def __init__(self, start: float = 0.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt
