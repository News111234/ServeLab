"""Triton kernels (perf path; requires Linux + GPU + triton).

The engine runs the torch reference path by default (correctness-first);
these kernels are the optimization targets for the "GPU kernel" workstream
and are validated against the torch path by tests/test_triton_kernels.py
(run on a GPU box). Layout matches servelab's flat paged KV pool:
    k/v store: [num_rows = num_blocks*block_size, num_kv_heads, head_dim]

Roadmap: flash-decoding block-split two-phase reduction, fp8/int8 in-kernel
dequant, fused rope+attention, TMA / persistent scheduling on Hopper.
"""

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:          # pragma: no cover - CPU machines
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _paged_decode_attention_kernel(
        q_ptr, k_ptr, v_ptr, o_ptr,
        block_tables_ptr, ctx_lens_ptr,
        scale,
        stride_q_seq, stride_q_head,
        stride_k_row, stride_k_head,
        stride_v_row, stride_v_head,
        stride_o_seq, stride_o_head,
        stride_bt_seq,
        NUM_GROUPS: tl.constexpr,       # query heads per kv head (GQA)
        BLOCK_SIZE: tl.constexpr,       # tokens per KV block
        HEAD_DIM: tl.constexpr,
    ):
        seq = tl.program_id(0)
        head = tl.program_id(1)
        kv_head = head // NUM_GROUPS

        ctx = tl.load(ctx_lens_ptr + seq)
        offs_d = tl.arange(0, HEAD_DIM)
        q = tl.load(q_ptr + seq * stride_q_seq + head * stride_q_head
                    + offs_d).to(tl.float32)                       # [D]

        m_i = float("-inf")
        l_i = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        for b in range(0, tl.cdiv(ctx, BLOCK_SIZE)):
            block_id = tl.load(block_tables_ptr + seq * stride_bt_seq + b)
            offs_n = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask_n = offs_n < ctx
            rows = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            k = tl.load(k_ptr + rows[:, None] * stride_k_row
                        + kv_head * stride_k_head + offs_d[None, :],
                        mask=mask_n[:, None], other=0.0).to(tl.float32)
            s = tl.sum(q[None, :] * k, axis=1) * scale             # [B]
            s = tl.where(mask_n, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=0))
            p = tl.exp(s - m_new)                                  # [B]
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            v = tl.load(v_ptr + rows[:, None] * stride_v_row
                        + kv_head * stride_v_head + offs_d[None, :],
                        mask=mask_n[:, None], other=0.0).to(tl.float32)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            m_i = m_new

        acc = acc / l_i
        tl.store(o_ptr + seq * stride_o_seq + head * stride_o_head + offs_d,
                 acc.to(o_ptr.dtype.element_ty))

    @triton.jit
    def _rmsnorm_kernel(
        x_ptr, w_ptr, y_ptr,
        stride_x_row, stride_y_row,
        N: tl.constexpr, eps,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + row * stride_x_row + cols, mask=mask,
                    other=0.0).to(tl.float32)
        rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(y_ptr + row * stride_y_row + cols, y, mask=mask)


def paged_decode_attention_triton(
    q: torch.Tensor,            # [num_seqs, num_heads, head_dim]
    k_store: torch.Tensor,      # [num_rows, num_kv_heads, head_dim]
    v_store: torch.Tensor,
    block_tables: torch.Tensor, # [num_seqs, max_blocks] int32
    ctx_lens: torch.Tensor,     # [num_seqs]
    scale: float,
    num_kv_heads: int,
    block_size: int,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("triton not available; use the torch reference path")
    num_seqs, num_heads, head_dim = q.shape
    out = torch.empty_like(q)
    grid = (num_seqs, num_heads)
    _paged_decode_attention_kernel[grid](
        q, k_store, v_store, out,
        block_tables, ctx_lens, scale,
        q.stride(0), q.stride(1),
        k_store.stride(0), k_store.stride(1),
        v_store.stride(0), v_store.stride(1),
        out.stride(0), out.stride(1),
        block_tables.stride(0),
        NUM_GROUPS=num_heads // num_kv_heads,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
    )
    return out


def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("triton not available; use the torch reference path")
    n = x.shape[-1]
    y = torch.empty_like(x)
    block = max(16, triton.next_power_of_2(n))
    _rmsnorm_kernel[(x.numel() // n,)](
        x, weight, y, x.stride(-2), y.stride(-2), N=n, eps=eps, BLOCK=block,
        num_warps=8 if block >= 4096 else 4,
    )
    return y
