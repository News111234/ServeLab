"""Block manager: ties the physical block pool to per-sequence block tables
and the radix prefix cache. Torch-free and unit-testable (this is the layer
vLLM calls `KVCacheManager` / `BlockSpaceManager`).
"""

import math
from collections import deque
from enum import Enum
from typing import Dict, List, Optional

from ..utils.common import hash_block_tokens
from .radix import RadixCache, RadixNode
from .policies import build_eviction_policy


class AllocStatus(Enum):
    OK = 0        # can allocate now
    LATER = 1     # not enough evictable memory right now -> retry later / preempt
    NEVER = 2     # request larger than total capacity -> reject


class BlockManager:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        enable_prefix_caching: bool = True,
        eviction_policy: str = "lru",
        watermark: float = 0.01,
        swap_space=None,
        offloader=None,
        pool=None,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.watermark = watermark
        self.enable_prefix_caching = enable_prefix_caching
        self.radix: Optional[RadixCache] = (
            RadixCache(block_size, build_eviction_policy(eviction_policy))
            if enable_prefix_caching else None
        )
        self.free_ids: deque = deque(range(num_blocks))
        self.seq_blocks: Dict[str, List[int]] = {}
        self.seq_matched_nodes: Dict[str, List[RadixNode]] = {}
        # hierarchical-memory hooks (need a pool reference for copy ops)
        self.swap_space = swap_space        # CPUSwapSpace or None
        self.offloader = offloader          # CPUKVOffloader or None
        self.pool = pool                    # KVCachePool or None
        self._swap_meta: Dict[str, tuple] = {}   # seq_id -> (matched_blocks, priv_blocks)
        # stats
        self.stat_alloc_fails = 0
        self.stat_evicted_blocks = 0
        self.stat_prefix_hit_tokens = 0
        self.stat_total_match_tokens = 0
        self.stat_prefix_match_requests = 0
        self.stat_offload_restored_blocks = 0
        self.stat_swap_outs = 0
        self.stat_swap_ins = 0
        self.stat_swap_ins_failed = 0

    # ------------------------------------------------------------- capacity
    def _evictable(self) -> int:
        return self.radix.evictable_size() if self.radix else 0

    def num_free_blocks(self) -> int:
        return len(self.free_ids)

    def num_used_blocks(self) -> int:
        return self.num_blocks - len(self.free_ids) - self._evictable()

    def _reserve(self) -> int:
        return int(self.watermark * self.num_blocks)

    def get_num_new_blocks(self, num_total_tokens: int, num_alloc_blocks: int) -> int:
        target = math.ceil(num_total_tokens / self.block_size)
        return max(0, target - num_alloc_blocks)

    # -------------------------------------------------------- prefix match
    def maybe_match_prefix(self, seq_id: str, token_ids: List[int]) -> int:
        """Match + lock cached prefix for a sequence that has no computed
        tokens yet; with an offloader attached, missing prefix blocks are
        re-materialized from host memory first. Returns matched tokens."""
        assert seq_id not in self.seq_blocks
        self.seq_blocks[seq_id] = []
        self.seq_matched_nodes[seq_id] = []
        if self.radix is None:
            return 0
        blocks, nodes, num_tokens = self.radix.match_prefix(token_ids)
        self.stat_total_match_tokens += len(token_ids)
        num_tokens += self._restore_from_offloader(nodes, blocks, token_ids)
        self.seq_blocks[seq_id].extend(blocks)
        self.seq_matched_nodes[seq_id] = nodes
        if num_tokens:
            self.stat_prefix_hit_tokens += num_tokens
        self.stat_prefix_match_requests += 1
        return num_tokens

    def _restore_from_offloader(self, nodes: List[RadixNode],
                                blocks: List[int], token_ids: List[int]) -> int:
        """Re-materialize missing prefix blocks archived on the host. Extends
        the given match chain in place; returns restored token count."""
        if not (self.offloader and self.pool and self.radix):
            return 0
        bs = self.block_size
        num_full = len(token_ids) // bs
        parent = nodes[-1] if nodes else self.radix.root
        parent_hash = nodes[-1].key if nodes else 0
        i = len(nodes)
        restored = 0
        while i < num_full:
            blk = token_ids[i * bs:(i + 1) * bs]
            h = hash_block_tokens(parent_hash, blk)
            child = parent.children.get(h)
            if child is not None:              # tree advanced further: follow
                parent, parent_hash, i = child, h, i + 1
                continue
            entry = self.offloader.get(h)
            if entry is None:
                break
            toks, _tensors = entry
            if not self.free_ids:
                for bid, key, toks2 in self.radix.evict_with_info(1):
                    if key is not None:
                        self.offloader.put(key, toks2, self.pool, [bid])
                    self.free_ids.append(bid)
            if not self.free_ids:
                break
            bid = self.free_ids.popleft()
            rows = [bid * bs + j for j in range(bs)]
            for l, (kk, vv) in enumerate(_tensors):
                self.pool.write(l, rows, kk, vv)
            node = self.radix.attach(parent, h, bid, blk)
            parent, parent_hash = node, h
            nodes.append(node)
            blocks.append(bid)
            restored += bs
            i += 1
            self.stat_offload_restored_blocks += 1
        return restored

    # ----------------------------------------------------------- allocation
    def can_allocate(self, seq_id: str, num_new_tokens: int,
                     num_computed: int) -> AllocStatus:
        if seq_id not in self.seq_blocks:      # match happens at admit time
            return AllocStatus.NEVER
        cur = len(self.seq_blocks[seq_id])
        need = self.get_num_new_blocks(num_computed + num_new_tokens, cur)
        if cur + need > self.num_blocks:
            return AllocStatus.NEVER
        available = self.num_free_blocks() + self._evictable() - self._reserve()
        return AllocStatus.OK if need <= available else AllocStatus.LATER

    def allocate_slots(self, seq_id: str, num_total_tokens: int,
                       ignore_watermark: bool = False) -> Optional[List[int]]:
        """Grow the sequence's block table so it can hold `num_total_tokens`.
        Returns newly allocated block ids, or None on failure (caller should
        defer / preempt)."""
        if seq_id not in self.seq_blocks:
            return None
        cur = len(self.seq_blocks[seq_id])
        need = self.get_num_new_blocks(num_total_tokens, cur)
        if need == 0:
            return []
        if self.radix is not None:
            avail = self.num_free_blocks() + self._evictable()
            if not ignore_watermark:
                avail -= self._reserve()
            if need > avail:
                self.stat_alloc_fails += 1
                return None
            if need > self.num_free_blocks():
                victims = self.radix.evict_with_info(need - self.num_free_blocks())
                self.stat_evicted_blocks += len(victims)
                for bid, key, toks in victims:
                    if key is not None and self.offloader is not None \
                            and self.pool is not None:
                        # archive before the physical block is reused
                        self.offloader.put(key, toks, self.pool, [bid])
                    self.free_ids.append(bid)
                if need > self.num_free_blocks():
                    self.stat_alloc_fails += 1
                    return None
        else:
            if need > self.num_free_blocks():
                self.stat_alloc_fails += 1
                return None
        new_ids = []
        for _ in range(need):
            new_ids.append(self.free_ids.popleft())
        self.seq_blocks[seq_id].extend(new_ids)
        return new_ids

    def append_slot(self, seq_id: str, num_stored_tokens: int) -> Optional[int]:
        """Ensure capacity for one more token (decode step). Returns the block
        id to write into, or None on failure."""
        table = self.seq_blocks[seq_id]
        if num_stored_tokens < len(table) * self.block_size:
            return table[num_stored_tokens // self.block_size]
        got = self.allocate_slots(seq_id, num_stored_tokens + 1)
        if got is None:
            return None
        return self.seq_blocks[seq_id][num_stored_tokens // self.block_size]

    # ------------------------------------------------------------- release
    def undo_match(self, seq_id: str) -> None:
        """Drop a waiting sequence's prefix match without touching the pool:
        every block in its table is a locked tree block that must simply be
        unlocked again (they are still owned by the radix tree)."""
        blocks = self.seq_blocks.pop(seq_id, None)
        if blocks is None:
            return
        nodes = self.seq_matched_nodes.pop(seq_id, [])
        if self.radix is not None:
            self.radix.unlock(nodes)

    def release(self, seq_id: str, token_ids: List[int]) -> List[int]:
        """Release a finished / preempted sequence. With prefix caching on,
        full blocks are re-inserted into the radix tree; blocks whose content
        already exists (duplicates) and any partial tail are returned to the
        free pool. Returns freed block ids."""
        blocks = self.seq_blocks.pop(seq_id, None)
        if blocks is None:
            return []
        nodes = self.seq_matched_nodes.pop(seq_id, [])
        if self.radix is None:
            self.free_ids.extend(blocks)
            return list(blocks)
        self.radix.unlock(nodes)
        # `token_ids` must cover only tokens whose KV was actually written;
        # the caller truncates (a freshly sampled token has no KV yet)
        num_full = len(token_ids) // self.block_size
        dups = self.radix.insert(token_ids[: num_full * self.block_size],
                                 blocks[:num_full])
        freed = set(dups)
        freed.update(blocks[num_full:])          # partial tail block(s)
        self.free_ids.extend(freed)
        return sorted(freed)

    # ----------------------------------------------------------- swap mode
    def swap_out(self, seq_id: str, token_count: int) -> bool:
        """Move a running sequence's PRIVATE KV blocks to host memory; its
        matched prefix blocks simply stay in the tree (they are shared).
        The sequence can resume without recomputation."""
        if self.swap_space is None or self.pool is None:
            return False
        blocks = self.seq_blocks.pop(seq_id, None)
        if blocks is None:
            return False
        nodes = self.seq_matched_nodes.pop(seq_id, [])
        if self.radix is not None:
            self.radix.unlock(nodes)
        private = blocks[len(nodes):]
        if private:
            self.swap_space.swap_out(self.pool, seq_id, private)
        self._swap_meta[seq_id] = (len(nodes), len(private))
        self.free_ids.extend(private)
        self.stat_swap_outs += 1
        return True

    def swap_in(self, seq_id: str, token_ids: List[int]) -> bool:
        """Re-materialize a swapped-out sequence. Re-match the prefix first;
        if the tree can no longer provide the same matched prefix (it may
        have been evicted while the seq was on host), fall back to recompute
        by returning False (caller should reset the sequence)."""
        if seq_id not in self._swap_meta:
            return False
        matched_out, priv_count = self._swap_meta[seq_id]
        # release any leftover state, then re-match
        self.seq_blocks.pop(seq_id, None)
        self.seq_matched_nodes.pop(seq_id, None)
        num_matched = self.maybe_match_prefix(seq_id, token_ids)
        num_matched_blocks = num_matched // self.block_size
        if self.radix is not None and num_matched_blocks != matched_out:
            # prefix coverage changed while swapped -> private data would
            # misalign; fall back to recompute
            self.undo_match(seq_id)
            del self._swap_meta[seq_id]
            self.swap_space.drop(seq_id)
            self.stat_swap_ins_failed += 1
            return False
        new_private = self.allocate_slots(
            seq_id, num_matched + priv_count * self.block_size,
            ignore_watermark=True)
        if new_private is None or len(new_private) < priv_count:
            self.undo_match(seq_id)
            del self._swap_meta[seq_id]
            self.swap_space.drop(seq_id)
            self.stat_swap_ins_failed += 1
            return False
        extra = new_private[priv_count:]
        if extra:
            self.free_ids.extend(extra)      # allocated one block too many
            del self.seq_blocks[seq_id][len(self.seq_blocks[seq_id]) - len(extra):]
        self.swap_space.swap_in(self.pool, seq_id,
                                self.seq_blocks[seq_id][num_matched_blocks:])
        del self._swap_meta[seq_id]
        self.stat_swap_ins += 1
        return True

    def drop_swapped(self, seq_id: str) -> None:
        if self.swap_space is not None:
            self.swap_space.drop(seq_id)
        self._swap_meta.pop(seq_id, None)

    # ------------------------------------------------------------- queries
    def get_block_table(self, seq_id: str) -> List[int]:
        return self.seq_blocks[seq_id]

    def slot_for_token(self, seq_id: str, token_index: int) -> int:
        """Flat physical row index for a logical token position."""
        table = self.seq_blocks[seq_id]
        return table[token_index // self.block_size] * self.block_size \
            + token_index % self.block_size

    def slot_mapping(self, seq_id: str, start: int, num_tokens: int) -> List[int]:
        return [self.slot_for_token(seq_id, start + i) for i in range(num_tokens)]
