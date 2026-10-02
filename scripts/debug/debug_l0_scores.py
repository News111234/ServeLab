"""Compare L0 attention scores: paged path vs runner path, same weights."""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from helpers_tiny_model import build_tiny_qwen2
from servelab.config import ModelConfig
from servelab.kv_cache.manager import BlockManager
from servelab.kv_cache.pool import KVCachePool
from servelab.models.decoder import DecoderModel, SeqMeta, apply_rope, flat_rows
from servelab.models.loader import load_model

d = tempfile.mkdtemp()
cfg = build_tiny_qwen2(os.path.join(d, "m"))
mc = ModelConfig.from_hf(cfg, path=os.path.join(d, "m"))
_, state = load_model(os.path.join(d, "m"))
PROMPT = list(range(10, 30))
T = len(PROMPT)

dense = DecoderModel(mc, device="cpu", dtype=torch.float32)
dense.load_state_dict_hf(state)
dense.block_size = 16
pool = KVCachePool(8, 16, mc.num_layers, mc.num_kv_heads, mc.head_dim,
                   dtype=torch.float32, device="cpu", kv_cache_dtype="none")
bm = BlockManager(8, 16, enable_prefix_caching=False)
bm.seq_blocks["s"] = []
bm.seq_matched_nodes["s"] = []
bm.allocate_slots("s", T)
metas = [SeqMeta(0, T, T, bm.get_block_table("s"))]
slots = [bm.slot_for_token("s", i) for i in range(T)]
ids = torch.tensor(PROMPT)
pos = torch.arange(T)
slot_t = torch.tensor(slots)
groups = mc.num_heads // mc.num_kv_heads

layer = dense.layers[0]
with torch.no_grad():
    hidden = dense.embed_tokens(ids)
    x = layer.input_layernorm(hidden)
    q, k, v = layer.self_attn.qkv(x)
    q, k = apply_rope(q, k, pos, dense.inv_freq)

    # --- paged path
    pool.write(0, slots, k, v)
    rows = flat_rows(bm.get_block_table("s"), T, 16)
    k_c, v_c = pool.read(0, rows)
    k_c = k_c.repeat_interleave(groups, dim=1)
    v_c = v_c.repeat_interleave(groups, dim=1)
    s_paged = torch.einsum("qhd,rhd->hqr", q.float(), k_c.float()) \
        * layer.self_attn.scale

    # --- runner path
    k_all = k.clone()
    v_all = v.clone()
    k_all = k_all.repeat_interleave(groups, dim=1)
    v_all = v_all.repeat_interleave(groups, dim=1)
    s_run = torch.einsum("qhd,rhd->hqr", q.float(), k_all.float()) \
        * layer.self_attn.scale

    print("k diff (pool vs mem):", (k_c - k_all).abs().max().item())
    print("v diff (pool vs mem):", (v_c - v_all).abs().max().item())
    print("scores diff:", (s_paged - s_run).abs().max().item())

    p_paged = torch.softmax(s_paged, -1)
    p_run = torch.softmax(s_run, -1)
    o_paged = torch.einsum("hqr,rhd->qhd", p_paged, v_c.float())
    o_run = torch.einsum("hqr,rhd->qhd", p_run, v_all.float())
    print("attn out diff:", (o_paged - o_run).abs().max().item())

    a_paged = layer.self_attn.out(o_paged.reshape(T, -1))
    a_run = layer.self_attn.out(o_run.reshape(T, -1))
    print("o_proj diff:", (a_paged - a_run).abs().max().item())
