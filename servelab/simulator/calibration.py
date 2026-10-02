"""Calibration: fit the simulator's cost model against real measurements.

The simulator's step-time model is t = max(FLOPs/flops_eff, bytes/bw_eff) +
launch_overhead. This module fits the three hidden constants from measured
(engine-run) samples -- closing the sim-vs-real gap is roadmap direction #5.

Fitting approach: the max() makes this a piecewise problem; we fit a linear
model t ~ a*flops + b*bytes + c (an upper-bound surrogate of the max) and
report both the linear fit and the per-regime max-fit. For research use,
collect measurements spanning prefill-heavy AND decode-heavy batches so both
regimes are covered.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .model_cost import GpuSpec, ModelSpec, step_time


@dataclass
class Measurement:
    num_prefill_tokens: int
    num_decode_seqs: int
    total_kv_tokens: int

    def flops(self, model: ModelSpec) -> float:
        return 2.0 * model.num_params * (self.num_prefill_tokens
                                         + self.num_decode_seqs)

    def bytes(self, model: ModelSpec) -> float:
        return 2.0 * model.num_params + \
            model.kv_bytes_per_token * max(0, self.total_kv_tokens)


@dataclass
class FitResult:
    flops_eff: float                 # fitted achievable FLOP/s
    bw_eff: float                    # fitted achievable bytes/s
    launch_overhead_ms: float
    mape: float                      # mean abs percent error on the samples

    def gpu_spec(self, name: str = "fitted") -> GpuSpec:
        return GpuSpec(name, self.flops_eff, self.bw_eff,
                       self.launch_overhead_ms)


class CostCalibrator:
    def __init__(self, model: ModelSpec):
        self.model = model
        self.samples: List[Tuple[Measurement, float]] = []

    def add(self, m: Measurement, measured_seconds: float) -> None:
        assert measured_seconds > 0
        self.samples.append((m, measured_seconds))

    def fit(self, iterations: int = 50) -> Optional[FitResult]:
        """Alternating least squares on the max() model with THREE unknowns
        (flops_eff, bw_eff, launch overhead):
            1. assign each sample to its dominant regime;
            2. per regime, least-squares work-coefficient against (t - c);
            3. refit c = median(t - dominant_work);
            4. repeat until stable."""
        n = len(self.samples)
        if n < 3:
            return None
        flops_eff, bw_eff, overhead = 100e12, 1e12, 0.0
        for _ in range(iterations):
            groups = {0: [], 1: []}
            for m, t in self.samples:
                tc = m.flops(self.model) / flops_eff
                tm = m.bytes(self.model) / bw_eff
                groups[0 if tc >= tm else 1].append((m, t))
            for regime, coef_attr in ((0, "flops"), (1, "bytes")):
                num = den = 0.0
                for m, t in groups[regime]:
                    x = m.flops(self.model) if regime == 0 else m.bytes(self.model)
                    y = t - overhead
                    if y <= 0:
                        continue
                    num += x * y
                    den += x * x
                if den > 0 and num > 0:
                    inv = num / den
                    if regime == 0:
                        flops_eff = min(max(1.0 / inv, 1e9), 1e15)
                    else:
                        bw_eff = min(max(1.0 / inv, 1e9), 1e14)
            # refit overhead: median residual against the dominant work term
            residuals = []
            for m, t in self.samples:
                tc = m.flops(self.model) / flops_eff
                tm = m.bytes(self.model) / bw_eff
                work = tc if tc >= tm else tm
                residuals.append(t - work)
            overhead = max(0.0, sorted(residuals)[len(residuals) // 2])
        errs = []
        for m, t in self.samples:
            pred = step_time(m.num_prefill_tokens, m.num_decode_seqs,
                             m.total_kv_tokens, self.model,
                             GpuSpec("f", flops_eff, bw_eff,
                                     overhead * 1000))
            errs.append(abs(pred - t) / t)
        return FitResult(flops_eff, bw_eff, overhead * 1000,
                         sum(errs) / len(errs))

    @staticmethod
    def load_csv(path: str, model: ModelSpec) -> "CostCalibrator":
        """Measurements CSV: prefill_tokens,decode_seqs,kv_tokens,seconds"""
        import csv
        cal = CostCalibrator(model)
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cal.add(Measurement(int(row["prefill_tokens"]),
                                    int(row["decode_seqs"]),
                                    int(row["kv_tokens"])),
                        float(row["seconds"]))
        return cal
