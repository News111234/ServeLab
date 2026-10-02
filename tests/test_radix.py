import pytest

from servelab.kv_cache.policies import CostAwareEviction, LRUEviction, LFUEviction, build_eviction_policy
from servelab.kv_cache.radix import RadixCache


def bs_list(*blocks):
    out = []
    for b in blocks:
        out.extend(b)
    return out


def test_match_and_insert_roundtrip():
    rc = RadixCache(block_size=4)
    toks = bs_list([1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12])
    blocks = list(range(3))
    dups = rc.insert(toks, blocks)
    assert dups == []
    assert rc.num_blocks == 3

    ids, nodes, n = rc.match_prefix(toks[:8] + [99])     # 2 full blocks shared
    assert n == 8                                        # only full blocks match
    assert ids == [0, 1]
    assert all(node.lock == 1 for node in nodes)

    # re-inserting the same content with new block ids yields duplicates
    dups = rc.insert(toks, [10, 11, 12])
    assert dups == [10, 11, 12]
    assert rc.num_blocks == 3


def test_lru_eviction_order():
    rc = RadixCache(block_size=4, policy=LRUEviction())
    a = bs_list([1, 2, 3, 4])                # siblings (separate chains)
    b = bs_list([9, 9, 9, 9])
    rc.insert(a, [0])
    rc.insert(b, [1])
    _, nodes, _ = rc.match_prefix(b)          # touch b -> a becomes LRU
    freed = rc.evict(1)
    assert freed == [0]
    rc.unlock(nodes)
    freed = rc.evict(1)
    assert freed == [1]


def test_lock_blocks_eviction():
    rc = RadixCache(block_size=4)
    # chained blocks: [1..4] is the parent of [5..8]
    rc.insert(bs_list([1, 2, 3, 4], [5, 6, 7, 8]), [0, 1])
    _, nodes, _ = rc.match_prefix(bs_list([1, 2, 3, 4]))    # locks parent only
    # the locked parent cannot go, but its unlocked leaf child may
    assert rc.evict(5) == [1]
    rc.unlock(nodes)
    assert rc.evict(5) == [0]


def test_policies_registry():
    assert isinstance(build_eviction_policy("lru"), LRUEviction)
    assert isinstance(build_eviction_policy("lfu"), LFUEviction)
    assert isinstance(build_eviction_policy("costaware"), CostAwareEviction)
    with pytest.raises(ValueError):
        build_eviction_policy("nope")


def test_lfu_vs_lru_choice():
    rc = RadixCache(block_size=4, policy=LFUEviction())
    hot = bs_list([1, 2, 3, 4])
    cold = bs_list([9, 9, 9, 9])
    rc.insert(hot, [0])
    rc.insert(cold, [1])
    rc.match_prefix(hot)
    rc.match_prefix(hot)                     # hot hit twice
    freed = rc.evict(1)
    assert freed == [1]                      # cold evicted despite being newer
