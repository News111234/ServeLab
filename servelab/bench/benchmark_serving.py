"""End-to-end serving benchmark on the real engine (vLLM benchmark_serving style).

Two load patterns:
    --load-pattern fixed   : closed loop, keep --concurrency requests in flight
    --load-pattern poisson : open loop, arrivals at --request-rate req/s

Reports TTFT / TPOT percentiles, throughput and goodput under SLOs.

Example:
    python -m servelab.bench.benchmark_serving --model /path/to/Qwen2.5-0.5B \
        --num-prompts 64 --concurrency 8
"""

import argparse
import time
from typing import List

from tabulate import tabulate

from ..config import CacheConfig, ModelConfig, SchedulerConfig
from ..engine.engine import LLMEngine
from ..engine.sampling_params import SamplingParams
from ..utils.common import pct
from .datasets import sample_synthetic_prompts


def load_tokenizer(path: str):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception as e:  # pragma: no cover
        print(f"[bench] tokenizer unavailable ({e}); using raw token-id prompts")
        return None


def build_engine(args) -> LLMEngine:
    import json
    import os
    with open(os.path.join(args.model, "config.json")) as f:
        hf_cfg = json.load(f)
    mc = ModelConfig.from_hf(hf_cfg, path=args.model)
    return LLMEngine(
        mc,
        CacheConfig(block_size=args.block_size,
                    enable_prefix_caching=not args.no_prefix_cache,
                    kv_cache_dtype=args.kv_cache_dtype,
                    eviction_policy=args.eviction_policy,
                    gpu_memory_utilization=args.gpu_memory_utilization),
        SchedulerConfig(max_num_seqs=args.max_num_seqs,
                        max_num_batched_tokens=args.max_batched_tokens,
                        chunked_prefill=not args.no_chunked_prefill,
                        policy=args.scheduler_policy),
        device=args.device,
        tokenizer=load_tokenizer(args.model) if not args.raw_token_ids else None,
    )


def run_fixed(engine, workloads, concurrency: int):
    """Closed-loop: keep `concurrency` requests in flight."""
    pending = list(workloads)
    in_flight = []
    results = []
    while pending or in_flight:
        while pending and len(in_flight) < concurrency:
            w = pending.pop(0)
            rid = engine.add_request(
                prompt=w.get("prompt"),
                prompt_token_ids=w.get("prompt_token_ids"),
                sampling_params=SamplingParams(max_tokens=w["expected_output_len"],
                                               ignore_eos=True, temperature=0.0))
            in_flight.append(rid)
        outs = engine.step()
        for out in outs:
            if out.request_id in in_flight:
                in_flight.remove(out.request_id)
                results.append(out)
    return results


def run_poisson(engine, workloads, rate: float):
    """Open-loop with real-time Poisson arrivals (wall clock)."""
    import random
    rng = random.Random(7)
    results, next_arrival = [], time.monotonic()
    t0 = time.monotonic()
    for i, w in enumerate(workloads):
        # pace arrivals against wall clock while stepping the engine
        while time.monotonic() < next_arrival:
            outs = engine.step()
            results.extend(outs)
        rid = engine.add_request(
            prompt=w.get("prompt"),
            prompt_token_ids=w.get("prompt_token_ids"),
            sampling_params=SamplingParams(max_tokens=w["expected_output_len"],
                                           ignore_eos=True, temperature=0.0))
        next_arrival += rng.expovariate(rate)
    while engine.has_unfinished():
        results.extend(engine.step())
    return results


def report(results: List, elapsed: float, ttft_slo_ms: float, tpot_slo_ms: float):
    ttfts, tpots, out_toks = [], [], 0
    for r in results:
        m = r.metrics
        ttfts.append(m["ttft"] * 1000)
        n_out = len(r.outputs[0].token_ids)
        out_toks += n_out
        if n_out > 1:
            tpots.append((m["latency"] - m["ttft"]) / (n_out - 1) * 1000)
    good = sum(1 for t, p in zip(ttfts, tpots)
               if t <= ttft_slo_ms and p <= tpot_slo_ms) / max(1, len(tpots))
    rows = [
        ["requests", len(results)],
        ["elapsed_s", round(elapsed, 2)],
        ["output_throughput_tok/s", round(out_toks / elapsed, 1)],
        ["TTFT ms p50/p90/p99", f"{pct(ttfts,50):.0f} / {pct(ttfts,90):.0f} / {pct(ttfts,99):.0f}"],
        ["TPOT ms p50/p99", f"{pct(tpots,50):.1f} / {pct(tpots,99):.1f}"],
        ["goodput(TTFT & TPOT SLO)", f"{good:.1%}"],
    ]
    print(tabulate(rows, headers=["metric", "value"], tablefmt="github"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-prompts", type=int, default=32)
    ap.add_argument("--input-mean", type=int, default=200)
    ap.add_argument("--output-tokens", type=int, default=64)
    ap.add_argument("--load-pattern", choices=["fixed", "poisson"], default="fixed")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--request-rate", type=float, default=2.0)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--max-batched-tokens", type=int, default=2048)
    ap.add_argument("--no-prefix-cache", action="store_true")
    ap.add_argument("--no-chunked-prefill", action="store_true")
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--eviction-policy", default="lru")
    ap.add_argument("--scheduler-policy", default="fcfs")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--raw-token-ids", action="store_true",
                    help="skip tokenizer; use random token ids as prompts")
    ap.add_argument("--ttft-slo-ms", type=float, default=2000)
    args = ap.parse_args()

    engine = build_engine(args)
    workloads = sample_synthetic_prompts(args.num_prompts, args.input_mean,
                                         output_mean=args.output_tokens)
    t0 = time.monotonic()
    if args.load_pattern == "fixed":
        results = run_fixed(engine, workloads, args.concurrency)
    else:
        results = run_poisson(engine, workloads, args.request_rate)
    elapsed = time.monotonic() - t0
    report(results, elapsed, args.ttft_slo_ms, tpot_slo_ms=50)
    print("engine stats:", engine.stats())


if __name__ == "__main__":
    main()
