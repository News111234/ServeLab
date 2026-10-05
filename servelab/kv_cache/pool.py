"""Physical KV cache pool (torch-backed, reference implementation).

Layout is vLLM-style but flattened for simple indexing:
    k_store[layer] : [num_rows = num_blocks * block_size, num_kv_heads, head_dim]
Block b occupies rows [b*block_size, (b+1)*block_size); a token's physical row
is block_id * block_size + offset (this integer is the "slot").

Storage dtype is controlled by the quantizer: float16/bfloat16 passthrough,
int8, or fp8(e4m3) with per-(row, head) scales. See quant.py -- the quant
granularity is itself a research axis (paper hook: per-tensor / per-block /
per-token / per-channel trade-offs, cf. KIVI, KVQuant).
"""

from typing import Sequence, Tuple

import torch

from .quant import KVQuantizer, build_kv_quantizer


class KVCachePool:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cpu",
        kv_cache_dtype: str = "auto",
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.compute_dtype = dtype

        if kv_cache_dtype == "auto":
            # follow the compute dtype exactly: a float32 (CPU) model must
            # not silently lose precision through an fp16 KV store
            kv_cache_dtype = {torch.float32: "float32",
                              torch.bfloat16: "bfloat16"}.get(dtype, "float16")
        self.quant: KVQuantizer = build_kv_quantizer(kv_cache_dtype,
                                                     compute_dtype=dtype)

        num_rows = num_blocks * block_size
        self._is_fp8 = self.quant.mode == "fp8"
        # fp8 index_copy_/index_select are not implemented on CPU; store the
        # fp8 payload through a uint8 view (same element size)
        store_dtype = torch.uint8 if self._is_fp8 else self.quant.store_dtype()
        self.k_store = torch.zeros(num_layers, num_rows, num_kv_heads, head_dim,
                                   dtype=store_dtype, device=device)
        self.v_store = torch.zeros(num_layers, num_rows, num_kv_heads, head_dim,
                                   dtype=store_dtype, device=device)
        self.k_scale = self.quant.alloc_scale(num_layers, num_rows, num_kv_heads, device)
        self.v_scale = self.quant.alloc_scale(num_layers, num_rows, num_kv_heads, device)

    def _to_store(self, q: torch.Tensor) -> torch.Tensor:
        return q.view(torch.uint8) if self._is_fp8 else q

    def _from_store(self, rows: torch.Tensor) -> torch.Tensor:
        return rows.view(torch.float8_e4m3fn) if self._is_fp8 else rows

    # ------------------------------------------------------------------ write
    def write(self, layer: int, slots: Sequence[int],
              k: torch.Tensor, v: torch.Tensor) -> None:
        """k, v: [T, num_kv_heads, head_dim] in compute dtype."""
        idx = torch.as_tensor(slots, dtype=torch.long, device=self.device)
        qk, sk = self.quant.quantize(k)
        qv, sv = self.quant.quantize(v)
        self.k_store[layer].index_copy_(0, idx, self._to_store(qk))
        self.v_store[layer].index_copy_(0, idx, self._to_store(qv))
        if sk is not None:
            self.k_scale[layer].index_copy_(0, idx, sk)
            self.v_scale[layer].index_copy_(0, idx, sv)

    # ------------------------------------------------------------------- read
    def read(self, layer: int, row_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather rows (flat physical indices) and dequantize.
        Returns (k, v) in float32, shape [R, num_kv_heads, head_dim]."""
        idx = row_indices.to(self.device)
        k = self._from_store(self.k_store[layer].index_select(0, idx))
        v = self._from_store(self.v_store[layer].index_select(0, idx))
        if self.k_scale is not None:
            k = self.quant.dequantize(k, self.k_scale[layer].index_select(0, idx))
            v = self.quant.dequantize(v, self.v_scale[layer].index_select(0, idx))
        else:
            k = self.quant.dequantize(k, None)
            v = self.quant.dequantize(v, None)
        return k, v

    # ------------------------------------------------------------------ misc
    def rows_per_block(self, block_ids: Sequence[int]) -> torch.Tensor:
        """All row indices of the given blocks, flattened in order."""
        r = torch.arange(self.block_size, device="cpu")
        return torch.cat([(torch.tensor(b) * self.block_size + r) for b in block_ids])

    def nbytes(self) -> int:
        n = self.k_store.element_size() * self.k_store.numel() * 2  # k + v
        if self.k_scale is not None:
            n += self.k_scale.element_size() * self.k_scale.numel() * 2
        return n
