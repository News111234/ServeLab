"""Attention module: torch reference paths live in servelab.models.decoder;
Triton kernels (perf path) in servelab.attention.triton."""

from .triton.paged_attention import (
    HAS_TRITON, paged_decode_attention_triton, rmsnorm_triton,
)

__all__ = ["HAS_TRITON", "paged_decode_attention_triton", "rmsnorm_triton"]
