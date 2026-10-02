"""Experiment sweep harness: run benchmark grids, emit reproducible CSVs.

    python -m servelab.simulator.sweep --out results/routing_sweep.csv

The sweep is the paper-workhorse: every row is one simulation run with a
fixed seed, and the CSV is directly plottable (see scripts/plot_results.py).
"""

import argparse
import csv
import itertools
import os
import time
from typing import List

from .cluster import ClusterSimulator, SimConfig
from .metrics import compute_metrics
from .model_cost import GPU_PRESETS, MODEL_PRESETS
from .predictor import build_predictor
from .router import build_router
from .trace import make_synthetic_trace

FIELDS = ["router", "replicas", "rate", "prefix_groups", "prefix_len", "seed",
          "num_requests", "finished", "ttft_p50_ms", "ttft_p99_ms",
          "tpot_p50_ms", "tpot_p99_ms", "throughput_tok_s", "goodput",
          "jain", "cache_hit", "preemptions", "sim_seconds"]


def run_row(args, router: str, rate: float, replicas: int) -> dict:
    trace = make_synthetic_trace(
        num_requests=args.num_requests, request_rate=rate,
        prompt_mean=args.prompt_mean, output_mean=args.output_mean,
        num_prefix_groups=args.prefix_groups, prefix_len=args.prefix_len,
        seed=args.seed)
    cfg = SimConfig(
        num_replicas=replicas,
        gpu=GPU_PRESETS[args.gpu],
        model=MODEL_PRESETS[args.model],
        router=router,
        predictor=args.predictor,
        eviction_policy=args.eviction_policy,
        pd_mode=args.pd,
    )
    predictor = build_predictor(args.predictor)
    router_obj = None if args.pd else build_router(router, replicas, predictor)
    sim = ClusterSimulator(cfg, predictor=predictor, router=router_obj)
    t0 = time.perf_counter()
    results = sim.run(trace)
    wall = time.perf_counter() - t0
    m = compute_metrics(results, num_replicas=replicas)
    hit = (sum(r.stat_matched_tokens for r in sim.replicas)
           / max(1, sum(r.stat_prompt_tokens for r in sim.replicas)))
    return {
        "router": router, "replicas": replicas, "rate": rate,
        "prefix_groups": args.prefix_groups, "prefix_len": args.prefix_len,
        "seed": args.seed, "num_requests": args.num_requests,
        "finished": m.num_finished,
        "ttft_p50_ms": round(m.ttft_p50_ms, 1), "ttft_p99_ms": round(m.ttft_p99_ms, 1),
        "tpot_p50_ms": round(m.tpot_p50_ms, 1), "tpot_p99_ms": round(m.tpot_p99_ms, 1),
        "throughput_tok_s": round(m.throughput_tok_s, 1),
        "goodput": round(m.goodput, 4), "jain": round(m.jain_load_index, 4),
        "cache_hit": round(hit, 4), "preemptions": m.total_preemptions,
        "sim_seconds": round(wall, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/sweep.csv")
    ap.add_argument("--num-requests", type=int, default=300)
    ap.add_argument("--prompt-mean", type=float, default=256)
    ap.add_argument("--output-mean", type=float, default=128)
    ap.add_argument("--prefix-groups", type=int, default=16)
    ap.add_argument("--prefix-len", type=int, default=128)
    ap.add_argument("--gpu", default="A100")
    ap.add_argument("--model", default="qwen2.5-7b")
    ap.add_argument("--predictor", default="online-mean")
    ap.add_argument("--eviction-policy", default="lru")
    ap.add_argument("--pd", action="store_true")
    ap.add_argument("--routers", default="round-robin,least-loaded,"
                                        "prefix-affinity,predictive")
    ap.add_argument("--rates", default="2,4,8")
    ap.add_argument("--replicas-list", default="2,4")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    grid = itertools.product(args.routers.split(","),
                             [float(r) for r in args.rates.split(",")],
                             [int(r) for r in args.replicas_list.split(",")])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for router, rate, replicas in grid:
            row = run_row(args, router, rate, replicas)
            writer.writerow(row)
            f.flush()
            print(f"{router:>16} rate={rate:<4} replicas={replicas} "
                  f"ttft_p50={row['ttft_p50_ms']:>6} goodput={row['goodput']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
