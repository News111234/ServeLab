"""Radix-tree prefix cache over paged KV blocks (SGLang RadixCache style).

Design notes (differs from upstream on purpose, kept minimal for research):
- one tree node == one full KV block; the node key is the content hash of
  (parent chain, block token ids) so identical prefixes share physical blocks;
- only FULL blocks are cached (a partial tail block is dropped on release).
  This removes the need for block-level copy-on-write: matched blocks are
  immutable full blocks, a running sequence always appends into its own
  private blocks. vLLM v1 makes the same trade-off.
- eviction is pluggable (see policies.py): default LRU over unlocked leaves.
"""

from typing import Dict, List, Optional, Tuple

from ..utils.common import hash_block_tokens
from .policies import EvictionPolicy, LRUEviction


class RadixNode:
    __slots__ = ("parent", "children", "block_id", "token_ids", "lock", "hits", "key")

    def __init__(self, key: int = 0, block_id: int = -1, token_ids: Optional[List[int]] = None):
        self.parent: Optional[RadixNode] = None
        self.children: Dict[int, "RadixNode"] = {}
        self.key = key                       # content hash of this block
        self.block_id = block_id             # physical KV block index (-1: root)
        self.token_ids: List[int] = token_ids or []
        self.lock = 0                        # >0: held by a running sequence
        self.hits = 0

    @property
    def is_leaf(self) -> bool:
        return not self.children


class RadixCache:
    def __init__(
        self,
        block_size: int,
        policy: Optional[EvictionPolicy] = None,
    ):
        self.block_size = block_size
        self.root = RadixNode()
        self.policy = policy or LRUEviction()
        self._num_blocks = 0

    # ------------------------------------------------------------------ match
    def match_prefix(self, token_ids: List[int]) -> Tuple[List[int], List[RadixNode], int]:
        """Longest full-block prefix match. Locks matched nodes so they cannot
        be evicted while the requesting sequence runs."""
        blocks: List[int] = []
        nodes: List[RadixNode] = []
        node = self.root
        parent_hash = 0
        num_full = len(token_ids) // self.block_size
        for i in range(num_full):
            blk = token_ids[i * self.block_size:(i + 1) * self.block_size]
            h = hash_block_tokens(parent_hash, blk)
            child = node.children.get(h)
            if child is None:
                break
            child.lock += 1
            child.hits += 1
            self.policy.on_touch(child)
            blocks.append(child.block_id)
            nodes.append(child)
            node = child
            parent_hash = h
        return blocks, nodes, len(blocks) * self.block_size

    def unlock(self, nodes: List[RadixNode]) -> None:
        for n in nodes:
            n.lock = max(0, n.lock - 1)

    # ----------------------------------------------------------------- insert
    def insert(self, token_ids: List[int], block_ids: List[int]) -> List[int]:
        """Insert a finished sequence's full blocks. Returns block ids whose
        content already existed in the tree (duplicates) -- caller must free
        their physical KV storage."""
        assert len(token_ids) == len(block_ids) * self.block_size
        node = self.root
        parent_hash = 0
        dups: List[int] = []
        for i, block_id in enumerate(block_ids):
            blk = token_ids[i * self.block_size:(i + 1) * self.block_size]
            h = hash_block_tokens(parent_hash, blk)
            child = node.children.get(h)
            if child is None:
                child = RadixNode(key=h, block_id=block_id, token_ids=list(blk))
                child.parent = node
                node.children[h] = child
                self._num_blocks += 1
                self.policy.on_insert(child)
            elif child.block_id != block_id:
                # same content, different physical block -> ours is redundant
                dups.append(block_id)
            node = child
            parent_hash = h
        return dups

    # ----------------------------------------------------------------- evict
    def evict_with_info(self, num_blocks: int) -> List[Tuple[int, int, List[int]]]:
        """Evict like evict() but return (block_id, content_hash, token_ids)
        so a hierarchical offloader can archive the KV before the physical
        block is reused."""
        freed: List[Tuple[int, int, List[int]]] = []
        attempts = 0
        max_attempts = self.policy.evictable_count() + 1
        while len(freed) < num_blocks and attempts < max_attempts:
            attempts += 1
            candidates = self.policy.pick_victims(
                (num_blocks - len(freed)) * 2 + 4
            )
            progressed = False
            for block_id in candidates:
                node = self._find_node(block_id)
                if node is None or node.lock > 0 or not node.is_leaf:
                    continue
                info = (block_id, node.key, list(node.token_ids))
                self._remove(node)
                freed.append(info)
                progressed = True
                if len(freed) >= num_blocks:
                    break
            if not progressed:
                break
        return freed

    def evict(self, num_blocks: int) -> List[int]:
        """Evict up to num_blocks unlocked leaf blocks (policy-ordered).
        Returns the freed physical block ids."""
        return [bid for bid, _, _ in self.evict_with_info(num_blocks)]

    def attach(self, parent: "RadixNode", key: int, block_id: int,
               token_ids: List[int]) -> "RadixNode":
        """Attach a restored (e.g. offloaded) block under an existing node."""
        node = RadixNode(key=key, block_id=block_id, token_ids=list(token_ids))
        node.parent = parent
        parent.children[key] = node
        self._num_blocks += 1
        self.policy.on_insert(node)
        return node

    def _find_node(self, block_id: int) -> Optional[RadixNode]:
        # policy keeps (time, hits) keyed by block id; walk tree via reverse
        # index would be faster -- fine at research scale, see roadmap.
        stack = [self.root]
        while stack:
            n = stack.pop()
            for c in n.children.values():
                if c.block_id == block_id:
                    return c
                stack.append(c)
        return None

    def _remove(self, node: RadixNode) -> None:
        assert node.parent is not None
        del node.parent.children[node.key]
        self._num_blocks -= 1
        self.policy.on_remove(node.block_id)

    # ----------------------------------------------------------------- stats
    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    def evictable_size(self) -> int:
        return self.policy.evictable_count()

    def reset(self) -> None:
        self.root = RadixNode()
        self._num_blocks = 0
        self.policy = type(self.policy)()
