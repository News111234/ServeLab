"""Analytical GPU / model cost model for the simulator.

Step-time model (roofline-style):
    t_step = max( FLOPs / FLOPs_eff ,  Bytes / BW_eff ) + launch_overhead
    FLOPs  = 2 * num_params * (prefill_tokens + num_decode_seqs)
    Bytes  = 2 * num_params (weights) + kv_bytes_per_token * total_kv_tokens

Effective FLOPs/BW constants are *achievable* values measured from real
systems (per-GPU presets below); calibrate them against your own
`servelab.bench.benchmark_serving` runs -- closing the sim-vs-real gap is
itself part of the research (roadmap #5).
"""

from dataclasses import dataclass


@dataclass
class GpuSpec:
    name: str
    flops_eff: float          # achievable fp16/bf16 FLOP/s
    bw_eff: float             # achievable HBM bytes/s
    launch_overhead_ms: float = 2.0


GPU_PRESETS = {
    "A100":   GpuSpec("A100", 156e12, 1.4e12, 2.0),
    "H100":   GpuSpec("H100", 660e12, 2.9e12, 2.0),
    "H800":   GpuSpec("H800", 660e12, 2.0e12, 2.0),
    "A800":   GpuSpec("A800", 156e12, 1.4e12, 2.0),
    "4090":   GpuSpec("4090", 165e12, 0.9e12, 2.0),
    "L20":    GpuSpec("L20", 119e12, 0.75e12, 2.0),
    "T4":     GpuSpec("T4", 65e12, 0.3e12, 3.0),
}


@dataclass
class ModelSpec:
    name: str = "qwen2.5-7b"
    num_params: float = 7.6e9
    kv_bytes_per_token: float = 0.0   # 2 (K+V) * layers * kv_heads * head_dim * 2B

    @staticmethod
    def from_config(num_layers: int, num_kv_heads: int, head_dim: int,
                    num_params: float, dtype_bytes: int = 2,
                    name: str = "model") -> "ModelSpec":
        return ModelSpec(
            name=name,
            num_params=num_params,
            kv_bytes_per_token=2 * num_layers * num_kv_heads * head_dim * dtype_bytes,
        )


# common open models (approximate, bf16 KV)
MODEL_PRESETS = {
    "llama2-7b":   ModelSpec("llama2-7b", 6.7e9, 2 * 32 * 32 * 128 * 2),
    "llama3-8b":   ModelSpec("llama3-8b", 8.0e9, 2 * 32 * 8 * 128 * 2),
    "qwen2.5-7b":  ModelSpec("qwen2.5-7b", 7.6e9, 2 * 28 * 4 * 128 * 2),
    "qwen2.5-32b": ModelSpec("qwen2.5-32b", 32.5e9, 2 * 64 * 8 * 128 * 2),
    "qwen2.5-72b": ModelSpec("qwen2.5-72b", 72.7e9, 2 * 80 * 8 * 128 * 2),
    "deepseek-v3-lite": ModelSpec("deepseek-v3-lite", 16.0e9, 2 * 27 * 1 * 128 * 2),
}


def step_time(
    num_prefill_tokens: int,
    num_decode_seqs: int,
    total_kv_tokens: int,
    model: ModelSpec,
    gpu: GpuSpec,
) -> float:
    """Seconds for one engine step (chunked prefill + decode mixed batch)."""
    flops = 2.0 * model.num_params * (num_prefill_tokens + num_decode_seqs)
    byte = 2.0 * model.num_params + model.kv_bytes_per_token * max(0, total_kv_tokens)
    t = max(flops / gpu.flops_eff, byte / gpu.bw_eff)
    return t + gpu.launch_overhead_ms / 1000.0


def prefill_time(num_tokens: int, model: ModelSpec, gpu: GpuSpec) -> float:
    return step_time(num_tokens, 0, 0, model, gpu)


def decode_step_time(batch_size: int, total_kv_tokens: int,
                     model: ModelSpec, gpu: GpuSpec) -> float:
    return step_time(0, batch_size, total_kv_tokens, model, gpu)
