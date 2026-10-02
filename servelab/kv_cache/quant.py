"""KV cache quantization (reference implementations, torch-backed).

All modes store one scale per (token row, kv head) vector -- simple, never
needs requantization on append, and good enough as a correctness baseline.
Research hooks (see docs/research_roadmap.md #3):
    - per-channel K scales (KIVI-style) via a transposed store
    - FP8 with online per-tensor scale calibration (vLLM-style)
    - outlier-aware rotation (QuaRot/SpinVer style) before quantization
"""

from typing import Optional, Tuple

import torch

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = 448.0


class KVQuantizer:
    mode = "none"

    def alloc_scale(self, num_layers: int, num_rows: int, num_heads: int,
                    device: str) -> Optional[torch.Tensor]:
        return None

    def store_dtype(self) -> torch.dtype:
        raise NotImplementedError

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError

    def dequantize(self, q: torch.Tensor, scale: Optional[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class FloatKVQuantizer(KVQuantizer):
    """Passthrough (float16/bfloat16 cache)."""

    mode = "none"

    def __init__(self, dtype: torch.dtype = torch.float16):
        self.dtype = dtype

    def store_dtype(self) -> torch.dtype:
        return self.dtype

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return x.to(self.dtype), None

    def dequantize(self, q: torch.Tensor, scale: None) -> torch.Tensor:
        return q.to(torch.float32)


class RowWiseInt8KVQuantizer(KVQuantizer):
    """Symmetric int8, scale per (row, head) vector."""

    mode = "int8"

    def store_dtype(self) -> torch.dtype:
        return torch.int8

    def alloc_scale(self, num_layers, num_rows, num_heads, device):
        return torch.zeros(num_layers, num_rows, num_heads, dtype=torch.float32, device=device)

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        amax = x.abs().amax(dim=-1, keepdim=False).float().clamp_min(1e-8)  # [R, H]
        scale = amax / 127.0
        q = torch.round(x.float() / scale.unsqueeze(-1)).clamp_(-127, 127).to(torch.int8)
        return q, scale

    def dequantize(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return q.float() * scale.unsqueeze(-1)


class RowWiseFP8KVQuantizer(KVQuantizer):
    """E4M3 fp8 with per-(row, head) scale."""

    mode = "fp8"

    def store_dtype(self) -> torch.dtype:
        return _FP8_DTYPE

    def alloc_scale(self, num_layers, num_rows, num_heads, device):
        return torch.zeros(num_layers, num_rows, num_heads, dtype=torch.float32, device=device)

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        amax = x.abs().amax(dim=-1, keepdim=False).float().clamp_min(1e-8)  # [R, H]
        scale = amax / _FP8_MAX
        q = (x.float() / scale.unsqueeze(-1)).clamp_(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)
        return q, scale

    def dequantize(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return q.float() * scale.unsqueeze(-1)


def build_kv_quantizer(mode: str, compute_dtype: torch.dtype = torch.float16) -> KVQuantizer:
    mode = (mode or "none").lower()
    if mode in ("none", "auto", "float16", "fp16"):
        return FloatKVQuantizer(compute_dtype)
    if mode in ("bfloat16", "bf16"):
        return FloatKVQuantizer(torch.bfloat16)
    if mode in ("float32", "fp32", "f32"):
        return FloatKVQuantizer(torch.float32)
    if mode == "int8":
        return RowWiseInt8KVQuantizer()
    if mode in ("fp8", "e4m3"):
        return RowWiseFP8KVQuantizer()
    raise ValueError(f"unknown kv_cache_dtype: {mode}")
