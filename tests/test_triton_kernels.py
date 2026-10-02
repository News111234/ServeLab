"""Triton kernel correctness tests -- run on a Linux + CUDA + triton box.

On machines without triton/CUDA (e.g. the Windows dev laptop) both tests
skip, so the suite stays green everywhere.
"""

import pytest
import torch

from servelab.attention.triton.paged_attention import HAS_TRITON
from servelab.models.decoder import flat_rows

needs_gpu_triton = pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()),
    reason="needs Linux + CUDA + triton",
)


@needs_gpu_triton
def test_triton_paged_decode_matches_torch():
    torch.manual_seed(0)
    batch, ctx, H, Hkv, D, bs = 8, 256, 32, 8, 64, 16
    groups = H // Hkv
    device = "cuda"
    q = torch.randn(batch, H, D, dtype=torch.float16, device=device)
    num_blocks = batch * (ctx // bs) + 4
    k = torch.randn(num_blocks * bs, Hkv, D, dtype=torch.float16, device=device)
    v = torch.randn_like(k)
    tables = [[(i * (ctx // bs) + j) % num_blocks for j in range(ctx // bs)]
              for i in range(batch)]
    bt = torch.tensor(tables, dtype=torch.int32, device=device)
    cl = torch.tensor([ctx] * batch, dtype=torch.int32, device=device)
    scale = D ** -0.5

    from servelab.attention.triton.paged_attention import paged_decode_attention_triton
    out = paged_decode_attention_triton(q, k, v, bt, cl, scale, Hkv, bs)

    ref = torch.empty_like(out)
    for i in range(batch):
        rows = flat_rows(tables[i], ctx, bs).to(device)
        k_c = k.index_select(0, rows).repeat_interleave(groups, 1).float()
        v_c = v.index_select(0, rows).repeat_interleave(groups, 1).float()
        s = torch.einsum("hd,rhd->hr", q[i].float(), k_c) * scale
        p = torch.softmax(s, dim=-1)
        ref[i] = torch.einsum("hr,rhd->hd", p, v_c).to(ref.dtype)
    err = (out.float() - ref.float()).abs().max().item()
    assert err < 5e-2, f"triton vs torch paged attention max err {err}"


@needs_gpu_triton
def test_triton_rmsnorm_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(512, 4096, dtype=torch.float16, device="cuda")
    w = torch.randn(4096, dtype=torch.float16, device="cuda")
    from servelab.attention.triton.paged_attention import rmsnorm_triton
    y = rmsnorm_triton(x, w, 1e-6)
    xf = x.float()
    ref = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
           * w.float()).to(x.dtype)
    err = (y.float() - ref.float()).abs().max().item()
    assert err < 5e-2, f"triton vs torch rmsnorm max err {err}"
