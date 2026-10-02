"""Layer-by-layer bisect: SimpleKVRunner vs DecoderModel.forward_packed."""
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
from servelab.models.decoder import DecoderModel, SeqMeta, apply_rope
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

# ---- reference: inline of DecoderModel.forward_packed, keeping per-layer hidden
with torch.no_grad():
    hidden_ref = dense.embed_tokens(ids)
    layer_hiddens_ref = []
    for i, layer in enumerate(dense.layers):
        residual = hidden_ref
        x = layer.input_layernorm(hidden_ref)
        q, k, v = layer.self_attn.qkv(x)
        q, k = apply_rope(q, k, pos, dense.inv_freq)
        pool.write(i, slots, k, v)
        attn = dense._paged_attention(layer, i, q, metas, pool)
        hidden_ref = residual + layer.self_attn.out(attn)
        residual = hidden_ref
        hidden_ref = residual + layer.mlp(layer.post_attention_layernorm(hidden_ref))
        layer_hiddens_ref.append(hidden_ref.clone())

# ---- runner: replicate SimpleKVRunner._forward with per-layer diffs
runner_hidden = dense.embed_tokens(ids)
k_cache = []
v_cache = []
groups = mc.num_heads // mc.num_kv_heads
with torch.no_grad():
    for i, layer in enumerate(dense.layers):
        residual = runner_hidden
        x = layer.input_layernorm(runner_hidden)
        q, k, v = layer.self_attn.qkv(x)
        q, k = apply_rope(q, k, pos, dense.inv_freq)
        k_cache.append(k)
        v_cache.append(v)
        k_all = torch.cat(k_cache, dim=0)
        v_all = torch.cat(v_cache, dim=0)
        if groups > 1:
            k_all = k_all.repeat_interleave(groups, dim=1)
            v_all = v_all.repeat_interleave(groups, dim=1)
        # compare against what the paged path reads from the pool
        from servelab.models.decoder import flat_rows
        rows = flat_rows(bm.get_block_table("s"), T, 16)
        pk, pv = pool.read(i, rows)
        if groups > 1:
            pk = pk.repeat_interleave(groups, dim=1)
            pv = pv.repeat_interleave(groups, dim=1)
        print(f"L{i} pool-vs-mem k diff:", (pk - k_all).abs().max().item(),
              "v diff:", (pv - v_all).abs().max().item())
        scores = torch.einsum("qhd,rhd->hqr", q.float(), k_all.float()) \
            * layer.self_attn.scale
        q_pos = pos[:, None]
        k_pos = torch.arange(k_all.shape[0])[None, :]
        scores = scores.masked_fill((q_pos < k_pos).unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        o = torch.einsum("hqr,rhd->qhd", probs, v_all.float())
        runner_hidden = residual + layer.self_attn.out(
            o.reshape(T, -1).to(residual.dtype))
        residual = runner_hidden
        runner_hidden = residual + layer.mlp(layer.post_attention_layernorm(runner_hidden))
        diff = (runner_hidden - layer_hiddens_ref[i]).abs().max().item()
        print(f"L{i} hidden diff: {diff:.3e}")
