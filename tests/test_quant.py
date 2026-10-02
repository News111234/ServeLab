import math

import pytest

import torch

from servelab.kv_cache.pool import KVCachePool
from servelab.kv_cache.quant import build_kv_quantizer
from servelab.quant.linear import W8A8Linear


@pytest.mark.parametrize("mode", ["int8", "fp8"])
def test_kv_quant_roundtrip(mode):
    x = torch.randn(64, 4, 32) * 3
    q = build_kv_quantizer(mode)
    qx, scale = q.quantize(x)
    xr = q.dequantize(qx, scale)
    rel = (xr - x).norm() / x.norm()
    assert rel.item() < 0.05, f"{mode} rel err {rel.item()}"


@pytest.mark.parametrize("mode", ["none", "int8", "fp8"])
def test_pool_write_read(mode):
    torch.manual_seed(0)
    pool = KVCachePool(num_blocks=4, block_size=8, num_layers=2, num_kv_heads=2,
                       head_dim=16, dtype=torch.float32, device="cpu",
                       kv_cache_dtype=mode)
    k = torch.randn(6, 2, 16)
    v = torch.randn(6, 2, 16)
    slots = list(range(6))
    pool.write(0, slots, k, v)
    idx = torch.tensor(slots)
    kr, vr = pool.read(0, idx)
    err = (kr - k).abs().max().item()
    assert err < 0.2, f"{mode} max abs err {err}"
    assert vr.shape == v.shape


def test_pool_block_row_mapping():
    pool = KVCachePool(num_blocks=4, block_size=8, num_layers=1, num_kv_heads=1,
                       head_dim=4, dtype=torch.float32, device="cpu", kv_cache_dtype="none")
    rows = pool.rows_per_block([2, 0]).tolist()
    assert rows == [16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7]


def test_w8a8_linear_close_to_fp32():
    torch.manual_seed(1)
    lin = torch.nn.Linear(128, 64)
    w8 = W8A8Linear.from_float(lin)
    x = torch.randn(32, 128)
    y_ref = lin(x)
    y_w8 = w8(x)
    rel = (y_w8 - y_ref).norm() / y_ref.norm()
    assert rel.item() < 0.05, f"W8A8 rel err {rel.item()}"
