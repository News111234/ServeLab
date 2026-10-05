"""Kernel microbenchmarks: torch reference vs Triton (GPU only).

    python -m servelab.bench.kernel_bench            # on a CUDA+triton box
"""

import torch

from ..attention.triton.paged_attention import HAS_TRITON
from ..models.decoder import flat_rows


def bench(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters            # ms


def bench_paged_decode(batch=32, ctx=1024, num_heads=32, num_kv_heads=8,
                       head_dim=128, block_size=16, dtype=torch.float16):
    device = "cuda"
    num_blocks = batch * (ctx // block_size + 2) + 1
    q = torch.randn(batch, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(num_blocks * block_size, num_kv_heads, head_dim,
                    dtype=dtype, device=device)
    v = torch.randn_like(k)
    tables, lens, _metas = [], [], []
    for i in range(batch):
        n_blocks = ctx // block_size
        tables.append(list(range(i * n_blocks, (i + 1) * n_blocks)))
        lens.append(ctx)
    bt = torch.tensor(tables, dtype=torch.int32, device=device)
    cl = torch.tensor(lens, dtype=torch.int32, device=device)
    scale = head_dim ** -0.5
    groups = num_heads // num_kv_heads

    def torch_ref():
        outs = []
        for i in range(batch):
            rows = flat_rows(tables[i], ctx, block_size).to(device)
            k_c = k.index_select(0, rows).repeat_interleave(groups, 1)
            v_c = v.index_select(0, rows).repeat_interleave(groups, 1)
            s = torch.einsum("hd,rhd->hr", q[i].float(), k_c.float()) * scale
            o = torch.einsum("hr,rhd->hd",
                             torch.softmax(s, -1), v_c.float())
            outs.append(o.reshape(-1))
        return torch.stack(outs)

    print(f"paged-decode bs={batch} ctx={ctx} heads={num_heads}/{num_kv_heads} d={head_dim}")
    print(f"  torch ref loop : {bench(torch_ref):.3f} ms")

    if HAS_TRITON:
        from ..attention.triton.paged_attention import paged_decode_attention_triton
        t_ms = bench(lambda: paged_decode_attention_triton(
            q, k, v, bt, cl, scale, num_kv_heads, block_size))
        print(f"  triton kernel  : {t_ms:.3f} ms")
        out_t = paged_decode_attention_triton(q, k, v, bt, cl, scale,
                                              num_kv_heads, block_size)
        out_r = torch_ref().reshape(batch, num_heads, head_dim).to(out_t.dtype)
        err = (out_t.float() - out_r.float()).abs().max().item()
        print(f"  max abs err    : {err:.2e}")
    else:
        print("  triton unavailable -> skipped")


def bench_rmsnorm(rows=4096, dim=4096, dtype=torch.float16):
    device = "cuda"
    x = torch.randn(rows, dim, dtype=dtype, device=device)
    w = torch.randn(dim, dtype=dtype, device=device)

    def torch_ref():
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
                * w.float()).to(dtype)

    print(f"rmsnorm rows={rows} dim={dim}")
    print(f"  torch  : {bench(torch_ref):.3f} ms")
    if HAS_TRITON:
        from ..attention.triton.paged_attention import rmsnorm_triton
        t_ms = bench(lambda: rmsnorm_triton(x, w, 1e-6))
        print(f"  triton : {t_ms:.3f} ms")
        err = (rmsnorm_triton(x, w, 1e-6).float() - torch_ref().float()).abs().max().item()
        print(f"  max abs err: {err:.2e}")
    else:
        print("  triton unavailable -> skipped")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available on this machine -- kernel bench needs a GPU.")
        print("Run this on a Linux GPU box (see docs/benchmark_guide.md).")
    else:
        bench_rmsnorm()
        bench_paged_decode()
