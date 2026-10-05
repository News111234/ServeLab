"""Dataclass configs shared by engine / simulator / bench.

Keep everything torch-free: the simulator imports these too.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Static description of a decoder model (Qwen2 / LLaMA family)."""

    path: str = ""
    architecture: str = "qwen2"          # "qwen2" | "llama"
    hidden_size: int = 896
    num_layers: int = 24
    num_heads: int = 14
    num_kv_heads: int = 2
    head_dim: int = 64
    intermediate_size: int = 4864
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 32768
    tie_word_embeddings: bool = False
    dtype: str = "float16"               # "float16" | "bfloat16" | "float32"
    eos_token_id: int = 151645

    @property
    def num_query_groups(self) -> int:
        return self.num_heads // self.num_kv_heads

    def num_params(self) -> int:
        """Approximate parameter count (embedding-tied aware)."""
        h, v = self.hidden_size, self.vocab_size
        per_layer = (
            3 * h * (self.num_heads + 2 * self.num_kv_heads) * self.head_dim  # qkv
            + h * self.num_heads * self.head_dim                              # o_proj
            + 2 * h * self.intermediate_size + h * self.intermediate_size     # mlp
        )
        emb = v * h
        n = self.num_layers * per_layer + 2 * h + emb
        if not self.tie_word_embeddings:
            n += emb
        return n

    @staticmethod
    def from_hf(cfg: dict, path: str = "") -> "ModelConfig":
        arch = cfg.get("model_type", "qwen2")
        if arch not in ("qwen2", "llama", "qwen2_vl_text", "qwen3"):
            raise ValueError(f"unsupported architecture: {arch}")
        if arch == "qwen3":
            arch = "qwen2"  # close enough at the decoder level (bias flags handled below)
        return ModelConfig(
            path=path,
            architecture=arch,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_hidden_layers"],
            num_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
            head_dim=cfg.get("head_dim",
                             cfg["hidden_size"] // cfg["num_attention_heads"]),
            intermediate_size=cfg["intermediate_size"],
            vocab_size=cfg["vocab_size"],
            rms_norm_eps=cfg["rms_norm_eps"],
            rope_theta=cfg.get("rope_theta", 1000000.0),
            max_position_embeddings=cfg.get("max_position_embeddings", 4096),
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
            dtype=cfg.get("torch_dtype", "float16"),
            eos_token_id=cfg.get("eos_token_id", 0),
        )


@dataclass
class CacheConfig:
    block_size: int = 16
    num_blocks: Optional[int] = None       # None => derived from memory budget
    gpu_memory_utilization: float = 0.85   # fraction of *free* device mem for KV
    enable_prefix_caching: bool = True
    kv_cache_dtype: str = "auto"           # "auto" | "float16" | "fp8" | "int8"
    eviction_policy: str = "lru"           # lru | lfu | costaware  (paper hook)
    watermark: float = 0.01                # keep this fraction of blocks free
    enable_kv_offload: bool = False        # archive evicted prefix blocks on host
    kv_offload_capacity_blocks: int = 8192


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 2048
    chunked_prefill: bool = True
    policy: str = "fcfs"                   # fcfs | priority | sjf-predicted
    preemption_mode: str = "recompute"     # recompute | swap(not implemented)
    decode_first: bool = False             # reserve budget for decodes first


@dataclass
class EngineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    seed: int = 1234
    enforce_eager: bool = True             # cuda-graph capture is a roadmap item
    device: str = "auto"                   # "auto" | "cuda" | "cpu"
    tp_rank: int = 0
    tp_world: int = 1                      # >1: tensor-parallel SPMD engine
