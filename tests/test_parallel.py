"""Tensor-parallel correctness: sharding rules, model math, engine E2E.

Core property: TP=2 (and TP=4) outputs are IDENTICAL to the dense model --
bit-for-bit in fp32 on the tiny model, within fp16 tolerance otherwise.
"""

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from helpers_tiny_model import build_tiny_qwen2  # noqa: E402

from servelab.config import CacheConfig, ModelConfig, SchedulerConfig  # noqa: E402
from servelab.engine.engine import LLMEngine  # noqa: E402
from servelab.engine.sampling_params import SamplingParams  # noqa: E402
from servelab.kv_cache.manager import BlockManager  # noqa: E402
from servelab.kv_cache.pool import KVCachePool  # noqa: E402
from servelab.models.decoder import DecoderModel, SeqMeta  # noqa: E402
from servelab.parallel.layers import shard_state_dict_tp, slice_shard  # noqa: E402
from servelab.parallel.simulated import SimulatedTPModelGroup  # noqa: E402
from servelab.parallel.layers import SimulatedCollective  # noqa: E402
from servelab.parallel.engine_group import SimulatedTPEngineGroup  # noqa: E402

PROMPT = list(range(10, 34))            # 24 tokens = 1.5 blocks (bs=16)


def test_shard_rules_roundtrip():
    """Every shard type must concat back to the original tensor."""
    world = 2
    state = {
        "model.embed_tokens.weight": torch.arange(96 * 32.).reshape(96, 32),
        "lm_head.weight": torch.arange(96 * 32.).reshape(96, 32) + 1,
        "model.layers.0.self_attn.q_proj.weight": torch.arange(32 * 32.).reshape(32, 32),
        "model.layers.0.self_attn.k_proj.weight": torch.arange(16 * 32.).reshape(16, 32),
        "model.layers.0.self_attn.v_proj.weight": torch.arange(16 * 32.).reshape(16, 32),
        "model.layers.0.self_attn.o_proj.weight": torch.arange(32 * 32.).reshape(32, 32),
        "model.layers.0.mlp.gate_proj.weight": torch.arange(64 * 32.).reshape(64, 32),
        "model.layers.0.mlp.down_proj.weight": torch.arange(32 * 64.).reshape(32, 64),
        "model.norm.weight": torch.ones(32),
    }
    for name, t in state.items():
        short = name[len("model."):] if name.startswith("model.") else name
        parts = [shard_state_dict_tp(state, r, world, 4, 2, 8, 96)[short]
                 for r in range(world)]
        if any(s in short for s in ("q_proj.weight", "embed_tokens", "lm_head",
                                    "k_proj", "v_proj", "gate_proj", "up_proj")):
            expect = torch.cat(parts, dim=0)
        elif "o_proj" in short or "down_proj" in short:
            expect = torch.cat(parts, dim=1)
        else:                                   # norms replicate
            expect = parts[0]
            assert all(torch.equal(p, expect) for p in parts), name
        assert torch.equal(expect, t), name


def test_slice_shard_basic():
    t = torch.arange(12.).reshape(4, 3)
    a, b = slice_shard(t, 0, 0, 2), slice_shard(t, 0, 1, 2)
    assert torch.equal(torch.cat([a, b], 0), t)


def test_tp_model_matches_dense(tmp_path):
    cfg = build_tiny_qwen2(str(tmp_path / "m"))
    mc = ModelConfig.from_hf(cfg, path=str(tmp_path / "m"))
    _, state = {}, None
    from servelab.models.loader import load_model
    _, state = load_model(str(tmp_path / "m"))

    dense = DecoderModel(mc, device="cpu", dtype=torch.float32)
    dense.load_state_dict_hf(state)
    dense.block_size = 16
    dense_pool = KVCachePool(num_blocks=8, block_size=16, num_layers=mc.num_layers,
                             num_kv_heads=mc.num_kv_heads, head_dim=mc.head_dim,
                             dtype=torch.float32, device="cpu", kv_cache_dtype="none")

    col = SimulatedCollective(2)
    group = SimulatedTPModelGroup(mc, 2, col, state_dict=state)
    pools = [KVCachePool(num_blocks=8, block_size=16, num_layers=mc.num_layers,
                         num_kv_heads=mc.num_kv_heads // 2, head_dim=mc.head_dim,
                         dtype=torch.float32, device="cpu", kv_cache_dtype="none")
             for _ in range(2)]

    bm = BlockManager(num_blocks=8, block_size=16, enable_prefix_caching=False)
    bm.seq_blocks["s"] = []
    bm.seq_matched_nodes["s"] = []
    bm.allocate_slots("s", len(PROMPT))
    table = bm.get_block_table("s")
    slots = [bm.slot_for_token("s", i) for i in range(len(PROMPT))]
    metas = [SeqMeta(offset=0, q_len=len(PROMPT), ctx_len=len(PROMPT),
                     block_table=table)]
    ids = torch.tensor(PROMPT, dtype=torch.long)
    pos = torch.arange(len(PROMPT))
    slot_t = torch.tensor(slots, dtype=torch.long)

    with torch.no_grad():
        h_dense = dense.forward_packed(ids, pos, slot_t, metas, dense_pool)
        h_tp = group.forward_packed(ids, pos, slot_t, pools, metas)
        lg_dense = dense.lm_head_forward(h_dense[-1:]).float()
        lg_tp = group.lm_head_forward(h_tp[-1:]).float()
    assert (h_dense - h_tp).abs().max().item() < 1e-4
    assert (lg_dense - lg_tp).abs().max().item() < 1e-3


@pytest.mark.parametrize("world", [2, 4])
def test_tp_engine_generation_matches_single_gpu(tmp_path, world):
    # world=4 needs kv_heads divisible by 4
    cfg = build_tiny_qwen2(str(tmp_path / f"m{world}"), heads=8, kv_heads=4)
    mc = ModelConfig.from_hf(cfg, path=str(tmp_path / f"m{world}"))
    cc = CacheConfig(num_blocks=32)
    sc = SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=64)

    single = LLMEngine(mc, cc, sc, seed=0, device="cpu")
    ref = single.generate([PROMPT, [40, 41, 42, 43]],
                          SamplingParams.greedy(max_tokens=10))
    ref_tokens = [o.outputs[0].token_ids for o in ref]

    group = SimulatedTPEngineGroup(mc, cc, sc, world_size=world, seed=0)
    got = group.generate([PROMPT, [40, 41, 42, 43]],
                         SamplingParams.greedy(max_tokens=10))
    got_tokens = [o.outputs[0].token_ids for o in got]
    assert got_tokens == ref_tokens


def test_tp_engine_group_generate_reusable(tmp_path):
    """generate() must work across repeated calls (request-id mapping)."""
    cfg = build_tiny_qwen2(str(tmp_path / "m"))
    mc = ModelConfig.from_hf(cfg, path=str(tmp_path / "m"))
    group = SimulatedTPEngineGroup(mc, CacheConfig(num_blocks=32),
                                   SchedulerConfig(max_num_seqs=4,
                                                   max_num_batched_tokens=64),
                                   world_size=2, seed=0)
    first = group.generate([PROMPT], SamplingParams.greedy(max_tokens=6))
    second = group.generate([PROMPT], SamplingParams.greedy(max_tokens=6))
    assert first[0] is not None and second[0] is not None
    assert first[0].outputs[0].token_ids == second[0].outputs[0].token_ids
