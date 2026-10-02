"""Tensor-parallel decoder model (Qwen2/Llama), Megatron sharding rules.

Same math as `models.decoder.DecoderModel` but:
    - attention heads / KV heads are rank-local (attention needs no comm);
    - q/k/v/gate/up are column-parallel (split output dim), o/down are
      row-parallel (split input dim, all-reduce output);
    - embedding and LM head are vocab-parallel (mask + reduce, all-gather);
    - the KV pool passed to forward holds only this rank's KV heads.

Weight loading: `shard_state_dict_tp()` slices a full HF state dict per rank
with the exact split rules; the module parameter names match, so a plain
load_state_dict works.
"""

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..models.decoder import RMSNorm, SeqMeta, apply_rope, flat_rows
from .layers import (
    ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding,
    VocabParallelLMHead,
)

ARCH_QKV_BIAS = {"qwen2": True, "llama": False}


class TPAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, collective):
        super().__init__()
        bias = ARCH_QKV_BIAS[cfg.architecture]
        world = collective.world_size
        assert cfg.num_heads % world == 0 and cfg.num_kv_heads % world == 0
        self.num_heads = cfg.num_heads // world          # local query heads
        self.num_kv_heads = cfg.num_kv_heads // world    # local kv heads
        self.head_dim = cfg.head_dim
        self.scale = 1.0 / math.sqrt(cfg.head_dim)
        h, kv, d = self.num_heads, self.num_kv_heads, self.head_dim
        self.q_proj = ColumnParallelLinear(cfg.hidden_size, h * d, collective, bias)
        self.k_proj = ColumnParallelLinear(cfg.hidden_size, kv * d, collective, bias)
        self.v_proj = ColumnParallelLinear(cfg.hidden_size, kv * d, collective, bias)
        self.o_proj = RowParallelLinear(h * d, cfg.hidden_size, collective, bias=False)

    def qkv(self, x: torch.Tensor):
        T = x.shape[0]
        q = self.q_proj(x).view(T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(T, self.num_kv_heads, self.head_dim)
        return q, k, v


class TPMLP(nn.Module):
    def __init__(self, cfg: ModelConfig, collective):
        super().__init__()
        i_local = cfg.intermediate_size // collective.world_size
        self.gate_proj = ColumnParallelLinear(cfg.hidden_size, i_local,
                                              collective, bias=False)
        self.up_proj = ColumnParallelLinear(cfg.hidden_size, i_local,
                                            collective, bias=False)
        self.down_proj = RowParallelLinear(i_local, cfg.hidden_size, collective,
                                           bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TPDecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, collective):
        super().__init__()
        self.self_attn = TPAttention(cfg, collective)
        self.mlp = TPMLP(cfg, collective)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)


class TPDecoderModel(nn.Module):
    def __init__(self, cfg: ModelConfig, collective, device: str = "cpu",
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.cfg = cfg
        self.collective = collective
        world = collective.world_size
        self.embed_tokens = VocabParallelEmbedding(cfg.vocab_size // world,
                                                   cfg.hidden_size, collective)
        self.layers = nn.ModuleList(
            TPDecoderLayer(cfg, collective) for _ in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = VocabParallelLMHead(cfg.vocab_size // world,
                                           cfg.hidden_size, collective)
        inv_freq = 1.0 / (cfg.rope_theta ** (
            torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.to(device=device, dtype=dtype)
        self.block_size = 16

    def load_shard(self, shard: Dict[str, torch.Tensor]) -> None:
        mine = set(self.state_dict().keys())
        filtered = {}
        for name, t in shard.items():
            if name.endswith("rotary_emb.inv_freq"):
                continue
            if name in mine:
                filtered[name] = t
        missing = {n for n in mine if n not in filtered} - {"inv_freq"}
        if missing:
            raise RuntimeError(f"missing TP shards: {sorted(missing)[:8]}")
        self.load_state_dict(filtered, strict=False)
        if self.cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def lm_head_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.cfg.tie_word_embeddings:
            return self.collective.all_gather_cat(
                F.linear(hidden, self.embed_tokens.weight), dim=-1)
        return self.lm_head(hidden)

    def forward_packed(self, input_ids, positions, slots, metas: List[SeqMeta],
                       pool) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            residual = hidden
            x = layer.input_layernorm(hidden)
            q, k, v = layer.self_attn.qkv(x)
            q, k = apply_rope(q, k, positions, self.inv_freq)
            pool.write(i, slots.tolist(), k, v)
            attn = self._paged_attention(layer, i, q, metas, pool)
            hidden = residual + layer.self_attn.o_proj(attn)
            residual = hidden
            hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
        return self.norm(hidden)

    def _paged_attention(self, layer, layer_idx, q, metas, pool):
        T, H, D = q.shape
        out = torch.empty(T, H * D, dtype=q.dtype, device=q.device)
        groups = H // pool.num_kv_heads
        bs = self.block_size
        for m in metas:
            rows = flat_rows(m.block_table, m.ctx_len, bs).to(q.device)
            k_c, v_c = pool.read(layer_idx, rows)
            if groups > 1:
                k_c = k_c.repeat_interleave(groups, dim=1)
                v_c = v_c.repeat_interleave(groups, dim=1)
            q_s = q[m.offset:m.offset + m.q_len].float()
            scores = torch.einsum("qhd,rhd->hqr", q_s, k_c) * layer.self_attn.scale
            if m.q_len > 1:
                q_pos = torch.arange(m.ctx_len - m.q_len, m.ctx_len)
                k_pos = torch.arange(m.ctx_len)
                mask = (q_pos[:, None] < k_pos[None, :]).unsqueeze(0)
                scores = scores.masked_fill(mask.to(scores.device), float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            o = torch.einsum("hqr,rhd->qhd", probs, v_c)
            out[m.offset:m.offset + m.q_len] = o.reshape(m.q_len, H * D).to(q.dtype)
        return out
