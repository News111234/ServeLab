
from servelab.kv_cache.manager import AllocStatus, BlockManager


def make_bm(num_blocks=16, block_size=4, enable=True, policy="lru"):
    return BlockManager(num_blocks=num_blocks, block_size=block_size,
                        enable_prefix_caching=enable, eviction_policy=policy)


def test_allocate_and_release():
    bm = make_bm()
    assert bm.maybe_match_prefix("s1", [1, 2, 3, 4, 5, 6, 7, 8]) == 0
    got = bm.allocate_slots("s1", 8)
    assert got is not None and len(got) == 2
    assert bm.get_block_table("s1") == got
    # one more token -> grows into a new block
    blk = bm.append_slot("s1", 8)
    assert blk is not None and len(bm.get_block_table("s1")) == 3
    assert bm.slot_for_token("s1", 0) == got[0] * 4 + 0
    assert bm.slot_for_token("s1", 5) == got[1] * 4 + 1
    bm.release("s1", list(range(9)))
    assert bm.num_free_blocks() + bm._evictable() == 16


def test_prefix_reuse_across_sequences():
    bm = make_bm()
    prompt = list(range(100, 112))          # 3 full blocks (bs=4)
    bm.maybe_match_prefix("s1", prompt)
    bm.allocate_slots("s1", 12)
    bm.release("s1", prompt)

    hit = bm.maybe_match_prefix("s2", prompt + [7, 7, 7])
    assert hit == 12                         # 3 blocks * 4 tokens
    table = bm.get_block_table("s2")
    assert len(table) == 3
    # capacity grew only by the 3 private suffix tokens (1 block)
    bm.allocate_slots("s2", 15)
    assert len(bm.get_block_table("s2")) == 4


def test_locked_blocks_not_evicted():
    bm = make_bm(num_blocks=8)
    prompt = list(range(100, 108))
    bm.maybe_match_prefix("s1", prompt)
    bm.allocate_slots("s1", 8)
    bm.release("s1", prompt)

    bm.maybe_match_prefix("s2", prompt)      # locks 2 blocks
    # try to evict everything: locked blocks must survive
    bm.radix.evict(10)
    assert bm.radix.num_blocks == 2
    bm.radix.unlock(bm.seq_matched_nodes["s2"])
    bm.radix.evict(10)
    assert bm.radix.num_blocks == 0


def test_eviction_under_pressure_and_rematch():
    bm = make_bm(num_blocks=16)
    # long one-off prefixes fill the cache
    for i in range(3):
        prompt = [i] + list(range(100, 120))     # 5 blocks each, distinct
        bm.maybe_match_prefix(f"p{i}", prompt)
        bm.allocate_slots(f"p{i}", 20)
        bm.release(f"p{i}", prompt)
    assert bm.radix.num_blocks == 15
    assert bm.num_free_blocks() == 1

    # a new seq needs 3 blocks -> evicts LRU cached blocks
    assert bm.maybe_match_prefix("new", list(range(50, 62))) == 0
    got = bm.allocate_slots("new", 12)
    assert got is not None and len(got) == 3
    assert bm.radix.num_blocks < 15


def test_alloc_status_never():
    bm = make_bm(num_blocks=4)
    assert bm.can_allocate("ghost", 100, 0) == AllocStatus.NEVER


def test_no_prefix_cache_mode():
    bm = make_bm(enable=False)
    assert bm.maybe_match_prefix("s1", list(range(8))) == 0
    assert bm.allocate_slots("s1", 8) is not None
    freed = bm.release("s1", list(range(8)))
    assert len(freed) == 2
    assert bm.num_free_blocks() == 16
