"""Shared test helpers: build a tiny random Qwen2 checkpoint on disk."""

import json
import os


def build_tiny_qwen2(
    path,
    vocab=96,
    hidden=32,
    layers=2,
    heads=4,
    kv_heads=2,
    head_dim=8,
    intermediate=64,
    seed=0,
):
    import torch

    torch.manual_seed(seed)
    cfg = {
        "model_type": "qwen2", "hidden_size": hidden,
        "num_hidden_layers": layers, "num_attention_heads": heads,
        "num_key_value_heads": kv_heads, "head_dim": head_dim,
        "intermediate_size": intermediate, "vocab_size": vocab,
        "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
        "max_position_embeddings": 512, "tie_word_embeddings": False,
        "torch_dtype": "float32", "eos_token_id": vocab - 1,
    }

    def rnd(*shape, std=0.05):
        return torch.randn(*shape) * std

    sd = {"model.embed_tokens.weight": rnd(vocab, hidden)}
    for i in range(layers):
        p = f"model.layers.{i}."
        sd[p + "self_attn.q_proj.weight"] = rnd(heads * head_dim, hidden)
        sd[p + "self_attn.k_proj.weight"] = rnd(kv_heads * head_dim, hidden)
        sd[p + "self_attn.v_proj.weight"] = rnd(kv_heads * head_dim, hidden)
        sd[p + "self_attn.q_proj.bias"] = torch.zeros(heads * head_dim)
        sd[p + "self_attn.k_proj.bias"] = torch.zeros(kv_heads * head_dim)
        sd[p + "self_attn.v_proj.bias"] = torch.zeros(kv_heads * head_dim)
        sd[p + "self_attn.o_proj.weight"] = rnd(hidden, heads * head_dim)
        sd[p + "mlp.gate_proj.weight"] = rnd(intermediate, hidden)
        sd[p + "mlp.up_proj.weight"] = rnd(intermediate, hidden)
        sd[p + "mlp.down_proj.weight"] = rnd(hidden, intermediate)
        sd[p + "input_layernorm.weight"] = torch.ones(hidden)
        sd[p + "post_attention_layernorm.weight"] = torch.ones(hidden)
    sd["model.norm.weight"] = torch.ones(hidden)
    sd["lm_head.weight"] = rnd(vocab, hidden, std=0.2)

    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    torch.save(sd, os.path.join(path, "pytorch_model.bin"))
    return cfg
