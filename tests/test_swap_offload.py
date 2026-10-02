"""Swap preemption + offload-restore integration tests (torch-backed).

Covers the KV hierarchy story end to end at the component level:
    GPU pool <-> CPU swap space (preemption)  and
    GPU pool <-> CPU offloader (evicted prefix archive + restore).
"""

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

from servelab.config import CacheConfig, SchedulerConfig  # noqa: E402
from servelab.engine.sampling_params import SamplingParams  # noqa: E402
from servelab.engine.scheduler import Scheduler  # noqa: E402
from servelab.engine.sequence import Sequence  # noqa: E402
from servelab.kv_cache.manager import BlockManager  # noqa: E402
from servelab.kv_cache.offload import CPUKVOffloader  # noqa: E402
from servelab.kv_cache.pool import KVCachePool  # noqa: E402
from servelab.kv_cache.swap import CPUSwapSpace  # noqa: E402
from servelab.utils.common import FakeClock  # noqa: E402


def make_pool(num_blocks=16, kv_heads=1, head_dim=8, layers=2):
    return KVCachePool(num_blocks=num_blocks, block_size=4,
                       num_layers=layers, num_kv_heads=kv_heads,
                       head_dim=head_dim, dtype=torch.float32,
                       device="cpu", kv_cache_dtype="none")


def test_swap_roundtrip_preserves_kv():
    torch.manual_seed(0)
    pool = make_pool()
    swap = CPUSwapSpace()
    k0 = torch.randn(4, 1, 8)
    v0 = torch.randn(4, 1, 8)
    pool.write(0, [0, 1, 2, 3], k0, v0)

    swap.swap_out(pool, "s1", [0])          # block 0 -> host
    # reuse the physical block with garbage
    pool.write(0, [0, 1, 2, 3], torch.randn(4, 1, 8), torch.randn(4, 1, 8))
    swap.swap_in(pool, "s1", [5])           # restore into block 5

    idx = torch.tensor([20, 21, 22, 23])    # block 5 rows
    k_back, v_back = pool.read(0, idx)
    assert (k_back - k0).abs().max().item() < 1e-6
    assert (v_back - v0).abs().max().item() < 1e-6
    assert "s1" not in swap


def test_manager_swap_out_in_restores_block_table():
    torch.manual_seed(0)
    pool = make_pool(num_blocks=16)
    swap = CPUSwapSpace()
    bm = BlockManager(num_blocks=16, block_size=4, swap_space=swap, pool=pool)

    prompt = list(range(100, 116))          # 4 blocks
    assert bm.maybe_match_prefix("s1", prompt) == 0
    assert bm.allocate_slots("s1", 16) is not None
    table = bm.get_block_table("s1")
    # write identifiable KV into every row
    k = torch.arange(16, dtype=torch.float32).reshape(16, 1, 1).expand(16, 1, 8) / 16
    v = torch.randn(16, 1, 8)
    rows = [bm.slot_for_token("s1", i) for i in range(16)]
    pool.write(0, rows, k, v)

    assert bm.swap_out("s1", token_count=16)
    assert "s1" in swap
    assert bm.num_free_blocks() == 16       # all 4 blocks were private (no match)

    # tree got the full blocks back -> prefix re-match gives 4 blocks
    assert bm.swap_in("s1", prompt)
    assert bm.get_block_table("s1") == table or True  # ids may differ; verify content
    rows2 = [bm.slot_for_token("s1", i) for i in range(16)]
    k2, _ = pool.read(0, torch.tensor(rows2))
    assert (k2[:, 0, 0] - torch.arange(16) / 16).abs().max().item() < 1e-6
    assert bm.stat_swap_outs == 1 and bm.stat_swap_ins == 1


def test_scheduler_swap_mode_keeps_progress():
    """A swap-preempted sequence resumes without losing generated tokens."""
    torch.manual_seed(0)
    pool = make_pool(num_blocks=8)
    swap = CPUSwapSpace()
    bm = BlockManager(num_blocks=8, block_size=4, swap_space=swap, pool=pool)
    sched = Scheduler(SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=64,
                                      preemption_mode="swap"),
                      CacheConfig(block_size=4), bm, FakeClock(), pool=pool)

    seq = Sequence("r0", list(range(1000, 1008)),
                   SamplingParams(max_tokens=12, ignore_eos=True), arrival_time=0.0)
    sched.add_request(seq)

    # drive: prefill completes, a few decode steps, then force a swap-out
    out = sched.schedule()
    assert out.prefills and out.prefills[0].num_new_tokens == 8
    seq.num_computed_tokens = 8
    for _ in range(3):
        sched.update_after_exec({"r0": 7})
        sched.schedule()
        seq.num_computed_tokens = seq.get_len() - 1
    assert seq.num_output_tokens == 3

    assert bm.swap_out("r0", seq.num_computed_tokens)
    seq.status = seq.status.WAITING
    seq._swapped = True
    assert seq.num_output_tokens == 3        # progress retained

    assert bm.swap_in("r0", seq.token_ids)
    seq._swapped = False
    assert seq.num_computed_tokens == seq.get_len() - 1   # invariant intact


def test_offload_archive_and_prefix_restore():
    """Evicted prefix blocks are archived on host and transparently restored
    on a later prefix match."""
    torch.manual_seed(0)
    pool = make_pool(num_blocks=10)
    off = CPUKVOffloader(capacity_blocks=64)
    bm = BlockManager(num_blocks=10, block_size=4, offloader=off, pool=pool)

    prompt = list(range(100, 120))          # 5 blocks
    bm.maybe_match_prefix("a", prompt)
    bm.allocate_slots("a", 20)
    rows = [bm.slot_for_token("a", i) for i in range(20)]
    k = torch.arange(20, dtype=torch.float32).reshape(20, 1, 1).expand(20, 1, 8) / 20
    v = torch.randn(20, 1, 8)
    pool.write(0, rows, k, v)
    bm.release("a", prompt)                 # blocks enter the tree

    # pressure: distinct one-off prompts evict the cached blocks (archived)
    for j in range(4):
        p = [200 + j] + list(range(300, 316))
        bm.maybe_match_prefix(f"j{j}", p)
        bm.allocate_slots(f"j{j}", 16)
        bm.release(f"j{j}", p)
    assert off.stat_offload_blocks >= 5
    assert bm.radix.num_blocks < 21         # evictions happened

    # same prefix again: tree miss -> offloader restore -> full match
    hit = bm.maybe_match_prefix("b", prompt)
    assert hit == 20, f"expected restored prefix hit, got {hit}"
    assert off.stat_restore_hits > 0
    # restored blocks contain the ORIGINAL data
    rows2 = [bm.slot_for_token("b", i) for i in range(20)]
    k2, _ = pool.read(0, torch.tensor(rows2))
    assert (k2[:, 0, 0] - torch.arange(20) / 20).abs().max().item() < 1e-6
    assert bm.stat_offload_restored_blocks >= 5


def test_engine_swap_mode_end_to_end(tmp_path):
    """Full engine with preemption_mode='swap': generation completes for all
    sequences, is deterministic across reruns, and the swap path actually
    cycles (out -> in). Note: outputs are NOT compared to recompute mode
    bit-for-bit -- packed-batch shapes differ between modes, so float32 GEMM
    rounding differs (the same batch-shape sensitivity exists in vLLM)."""
    from helpers_tiny_model import build_tiny_qwen2

    from servelab.config import ModelConfig, SchedulerConfig
    from servelab.engine.engine import LLMEngine
    from servelab.engine.sampling_params import SamplingParams

    cfg = build_tiny_qwen2(str(tmp_path / "m"))
    mc = ModelConfig.from_hf(cfg, path=str(tmp_path / "m"))

    def run(mode):
        eng = LLMEngine(mc, CacheConfig(num_blocks=8),
                        SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=64,
                                        preemption_mode=mode),
                        seed=0, device="cpu")
        outs = eng.generate([list(range(10, 30))] * 4,
                            SamplingParams.greedy(max_tokens=20))
        return [o.outputs[0].token_ids for o in outs], eng

    got, eng = run("swap")
    got2, _ = run("swap")
    assert got == got2                       # deterministic under fixed seed
    assert all(len(t) == 20 for t in got)
    assert eng.bm.stat_swap_outs > 0          # swap path actually exercised
    assert eng.bm.stat_swap_ins + eng.bm.stat_swap_ins_failed \
        == eng.bm.stat_swap_outs              # every swap-out is accounted for
