"""Real multi-process tensor-parallel inference via torch.distributed (gloo).

    # 2-rank TP on one machine (CPU gloo; on a GPU box swap backend to nccl):
    python examples/run_tp_gloo.py --model D:/models/Qwen2.5-0.5B-Instruct --tp 2

Every rank runs the SAME deterministic scheduler (SPMD, Megatron style);
rank 0 samples and broadcasts tokens. Outputs are identical to tp=1.
"""

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from servelab.config import CacheConfig, ModelConfig, SchedulerConfig
from servelab.engine.sampling_params import SamplingParams


def worker(rank, world, init_file, model_path, prompt_token_ids, max_tokens):
    from servelab.engine.engine import LLMEngine
    from servelab.parallel.layers import GlooCollective

    collective = GlooCollective(rank, world, f"file://{init_file}")
    _, state_path = None, model_path
    import json
    with open(os.path.join(model_path, "config.json")) as f:
        hf_cfg = json.load(f)
    mc = ModelConfig.from_hf(hf_cfg, path=model_path)
    engine = LLMEngine(mc, CacheConfig(num_blocks=256),
                       SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=1024),
                       seed=0, device="cpu",
                       tp_rank=rank, tp_world=world, collective=collective)
    # SPMD: every rank adds the same requests in the same order
    outputs = engine.generate([prompt_token_ids],
                              SamplingParams.greedy(max_tokens=max_tokens))
    if rank == 0:
        comp = outputs[0].outputs[0]
        print(f"[TP={world}] tokens: {comp.token_ids}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="HF checkpoint dir; empty = tiny random model")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=16)
    args = ap.parse_args()

    import torch.multiprocessing as mp

    if args.model:
        model_path = args.model
        prompt = list(range(10, 40))
    else:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
        from helpers_tiny_model import build_tiny_qwen2
        model_path = os.path.join(tempfile.mkdtemp(), "tiny")
        build_tiny_qwen2(model_path)
        prompt = list(range(10, 40))

    init_file = os.path.join(tempfile.mkdtemp(), "dist_init")
    mp.spawn(worker, args=(args.tp, init_file, model_path, prompt, args.max_tokens),
             nprocs=args.tp, join=True)


if __name__ == "__main__":
    main()
