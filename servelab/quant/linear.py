"""W8A8 dynamic-quantized linear (reference implementation).

Per-token activation quantization x per-channel weight quantization, the
standard W8A8 scheme (cf. vLLM's compressed-tensors INT8 path, SmoothQuant).
On CUDA this should call torch._int_mm / a CUTLASS int8 GEMM; here we
dequantize to fp16 for the matmul so the numerics can be validated anywhere.
The kernel itself is a paper/bench extension (see docs/research_roadmap.md).
"""

from typing import Optional

import torch
import torch.nn as nn


def quant_per_token_absmax(x: torch.Tensor) -> tuple:
    """x: [..., K] -> (q int8, scale [..., 1])."""
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = amax / 127.0
    q = torch.round(x / scale).clamp_(-127, 127).to(torch.int8)
    return q, scale


def quant_per_channel_weight(w: torch.Tensor) -> tuple:
    """w: [out, in] -> (q int8 [out, in], scale [out, 1])."""
    amax = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = amax / 127.0
    q = torch.round(w / scale).clamp_(-127, 127).to(torch.int8)
    return q, scale


class W8A8Linear(nn.Module):
    """Drop-in replacement for nn.Linear with dynamic per-token INT8 acts."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_q", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.ones(out_features, 1))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features))
        else:
            self.bias = None

    @classmethod
    def from_float(cls, linear: nn.Linear) -> "W8A8Linear":
        mod = cls(linear.in_features, linear.out_features, linear.bias is not None)
        q, s = quant_per_channel_weight(linear.weight.data.float())
        mod.weight_q.copy_(q)
        mod.weight_scale.copy_(s)
        if linear.bias is not None:
            mod.bias.copy_(linear.bias.data)
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, act_scale = quant_per_token_absmax(x.float())
        w = self.weight_q.to(torch.float32) * self.weight_scale      # dequant [out, in]
        a = q.to(torch.float32) * act_scale                          # [.., in]
        out = torch.matmul(a, w.t())
        if self.bias is not None:
            out = out + self.bias
        return out.to(x.dtype)
