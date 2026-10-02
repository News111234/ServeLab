"""Offline generation demo on the ServeLab engine.

    # 1) tiny random model (no download, runs anywhere):
    python examples/run_engine.py

    # 2) a real checkpoint (e.g. Qwen2.5-0.5B-Instruct), needs GPU build of torch:
    python examples/run_engine.py --model D:/models/Qwen2.5-0.5B-Instruct --prompt "你好"
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from servelab.config import CacheConfig, ModelConfig, SchedulerConfig
from servelab.engine.engine import LLMEngine
from servelab.engine.sampling_params import SamplingParams


def build_tiny_model(path="outputs/tiny-qwen2"):
    from tests.helpers_tiny_model import build_tiny_qwen2
    cfg = build_tiny_qwen2(path)
    return ModelConfig.from_hf(cfg, path=path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="HF checkpoint dir; empty = tiny random model")
    ap.add_argument("--prompt", default="The meaning of life is")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--no-prefix-cache", action="store_true")
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--num-blocks", type=int, default=0, help="0 = auto")
    args = ap.parse_args()

    if args.model:
        from servelab.models.loader import load_model
        mc, _ = load_model(args.model)
        tokenizer = None
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        except Exception as e:
            print(f"[warn] no tokenizer: {e}")
    else:
        mc = build_tiny_model()
        tokenizer = None

    engine = LLMEngine(
        mc,
        CacheConfig(block_size=args.block_size,
                    enable_prefix_caching=not args.no_prefix_cache,
                    kv_cache_dtype=args.kv_cache_dtype,
                    num_blocks=args.num_blocks or None),
        SchedulerConfig(max_num_seqs=16, max_num_batched_tokens=1024),
        tokenizer=tokenizer,
    )

    if tokenizer is not None:
        prompts = [args.prompt]
    else:
        prompts = [list(range(10, 40))]      # token ids for the tiny model

    outputs = engine.generate(prompts, SamplingParams(max_tokens=args.max_tokens),
                              verbose=True)
    for out in outputs:
        comp = out.outputs[0]
        text = comp.text if comp.text else str(comp.token_ids)
        print(f"[{out.request_id}] finish={comp.finish_reason} -> {text}")
    print("stats:", engine.stats())


if __name__ == "__main__":
    main()
