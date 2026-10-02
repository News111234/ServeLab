"""Calibration + sweep tests."""

import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from servelab.simulator.calibration import (  # noqa: E402
    CostCalibrator, Measurement,
)
from servelab.simulator.model_cost import ModelSpec, step_time  # noqa: E402
from servelab.simulator.sweep import run_row  # noqa: E402


class _Args:
    num_requests = 40
    prompt_mean = 128.0
    output_mean = 64.0
    prefix_groups = 4
    prefix_len = 32
    gpu = "A100"
    model = "qwen2.5-7b"
    predictor = "online-mean"
    eviction_policy = "lru"
    pd = False
    seed = 42


def test_calibrator_recovers_ground_truth():
    """Generate noise-free samples from KNOWN constants; the fit must recover
    them within a few percent."""
    model = ModelSpec("m", num_params=8e9, kv_bytes_per_token=65536)
    truth = (150e12, 1.4e12)     # flops_eff, bw_eff
    cal = CostCalibrator(model)
    # prefill-heavy (compute regime) and decode-heavy (memory regime) samples
    for pt, ds, kv in [
        (4096, 0, 0), (8192, 0, 0), (16384, 0, 0),
        (0, 32, 32768), (0, 64, 65536), (0, 16, 8192),
        (2048, 16, 16384), (4096, 32, 32768),
    ]:
        m = Measurement(pt, ds, kv)
        t = step_time(pt, ds, kv, model,
                      __import__("servelab.simulator.model_cost",
                                 fromlist=["GpuSpec"]).GpuSpec(
                          "truth", truth[0], truth[1], 2.0))
        cal.add(m, t)
    fit = cal.fit()
    assert fit is not None
    assert fit.flops_eff == pytest.approx(truth[0], rel=0.05)
    assert fit.bw_eff == pytest.approx(truth[1], rel=0.05)
    assert fit.mape < 0.05


def test_calibrator_needs_min_samples():
    model = ModelSpec("m", 8e9, 65536)
    cal = CostCalibrator(model)
    cal.add(Measurement(1024, 0, 0), 0.01)
    cal.add(Measurement(2048, 0, 0), 0.02)
    assert cal.fit() is None


def test_sweep_row_runs_and_returns_metrics():
    row = run_row(_Args(), router="prefix-affinity", rate=2.0, replicas=2)
    assert row["finished"] == _Args.num_requests
    assert row["ttft_p50_ms"] >= 0 and 0 <= row["goodput"] <= 1
    assert row["cache_hit"] >= 0
