"""Trace-driven cluster simulator CLI -- the research workhorse.

    # synthetic trace, compare routers:
    python examples/run_simulator.py --compare-routers

    # prefix-heavy workload, affinity vs least-loaded:
    python examples/run_simulator.py --replicas 4 --prefix-groups 8 \
        --prefix-len 256 --router prefix-affinity

    # PD disaggregation:
    python examples/run_simulator.py --pd --replicas 4

    # real traces:
    python examples/run_simulator.py --trace sharegpt.json --router least-loaded
    python examples/run_simulator.py --trace Azure_LLM_Inference_Trace.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from servelab.simulator.cluster import ClusterSimulator, SimConfig
from servelab.simulator.metrics import compute_metrics
from servelab.simulator.model_cost import GPU_PRESETS, MODEL_PRESETS
from servelab.simulator.predictor import build_predictor
from servelab.simulator.router import build_router
from servelab.simulator.trace import make_synthetic_trace


def load_trace(args):
    if args.trace == "synthetic":
        return make_synthetic_trace(
            num_requests=args.num_requests, request_rate=args.request_rate,
            prompt_mean=args.prompt_mean, output_mean=args.output_mean,
            num_prefix_groups=args.prefix_groups, prefix_len=args.prefix_len,
            seed=args.seed)
    from servelab.simulator.trace import load_trace as loader
    return loader(args.trace, max_requests=args.num_requests)


def run_once(args, router_name=None):
    trace = load_trace(args)
    cfg = SimConfig(
        num_replicas=args.replicas,
        gpu=GPU_PRESETS.get(args.gpu, GPU_PRESETS["A100"]),
        model=MODEL_PRESETS.get(args.model, MODEL_PRESETS["qwen2.5-7b"]),
        router=router_name or args.router,
        predictor=args.predictor if args.predictor != "oracle" else "online-mean",
        eviction_policy=args.eviction_policy,
        pd_mode=args.pd,
        pd_transfer_bw=args.pd_transfer_bw,
    )
    predictor = build_predictor(args.predictor)
    if args.predictor == "oracle":
        for r in trace:
            predictor.register(r.req_id, r.output_tokens)
    router = None if args.pd else build_router(cfg.router, args.replicas, predictor)
    sim = ClusterSimulator(cfg, predictor=predictor, router=router)
    results = sim.run(trace, verbose=args.verbose)
    metrics = compute_metrics(results, ttft_slo_ms=args.ttft_slo,
                              tpot_slo_ms=args.tpot_slo,
                              num_replicas=args.replicas)
    hit = (sum(r.stat_matched_tokens for r in sim.replicas)
           / max(1, sum(r.stat_prompt_tokens for r in sim.replicas)))
    metrics.mean_cache_hit_rate = hit
    preemptions = sum(r.stat_preemptions for r in sim.replicas)
    return metrics, preemptions, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="synthetic")
    ap.add_argument("--num-requests", type=int, default=300)
    ap.add_argument("--request-rate", type=float, default=4.0)
    ap.add_argument("--prompt-mean", type=float, default=256)
    ap.add_argument("--output-mean", type=float, default=128)
    ap.add_argument("--prefix-groups", type=int, default=16)
    ap.add_argument("--prefix-len", type=int, default=128)
    ap.add_argument("--replicas", type=int, default=2)
    ap.add_argument("--gpu", default="A100", choices=list(GPU_PRESETS))
    ap.add_argument("--model", default="qwen2.5-7b", choices=list(MODEL_PRESETS))
    ap.add_argument("--router", default="least-loaded")
    ap.add_argument("--predictor", default="online-mean",
                    choices=["oracle", "constant", "online-mean", "prompt-reg"])
    ap.add_argument("--eviction-policy", default="lru")
    ap.add_argument("--pd", action="store_true", help="PD disaggregation mode")
    ap.add_argument("--pd-transfer-bw", type=float, default=25e9)
    ap.add_argument("--ttft-slo", type=float, default=2000)
    ap.add_argument("--tpot-slo", type=float, default=50)
    ap.add_argument("--compare-routers", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.compare_routers:
        rows = []
        for router in ("round-robin", "least-loaded", "prefix-affinity", "predictive"):
            m, pre, _ = run_once(args, router_name=router)
            rows.append([router, f"{m.ttft_p50_ms:.0f}", f"{m.ttft_p99_ms:.0f}",
                         f"{m.tpot_p99_ms:.1f}", f"{m.throughput_tok_s:.0f}",
                         f"{m.goodput:.1%}", f"{m.mean_cache_hit_rate:.1%}",
                         f"{m.jain_load_index:.3f}", pre])
        from tabulate import tabulate
        print(tabulate(rows, headers=[
            "router", "TTFT p50", "TTFT p99", "TPOT p99", "tok/s",
            "goodput", "cache hit", "jain", "preemptions"], tablefmt="github"))
    else:
        m, pre, trace = run_once(args)
        m.print_table(title=f"sim: {args.router} / {args.replicas} replicas / "
                            f"{len(trace)} requests")
        print(f"total preemptions: {pre}")


if __name__ == "__main__":
    main()
