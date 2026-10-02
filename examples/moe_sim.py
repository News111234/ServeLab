"""MoE expert-parallel load-balance simulation demo.

    python examples/moe_sim.py --experts 64 --gpus 8 --topk 6
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from servelab.simulator.moe import rebalanced_sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--zipf", type=float, default=1.4)
    args = ap.parse_args()

    base, better = rebalanced_sim(num_tokens=args.tokens, num_experts=args.experts,
                                  top_k=args.topk, num_gpus=args.gpus,
                                  zipf_alpha=args.zipf)
    print(f"MoE EP sim: {args.tokens} tokens, {args.experts} experts, "
          f"top-{args.topk}, EP over {args.gpus} GPUs, zipf={args.zipf}")
    print(f"  round-robin placement : imbalance={base.imbalance_factor:.2f} "
          f"dropped={base.dropped_token_frac:.1%} all2all_pairs={base.all2all_tokens}")
    print(f"  greedy re-placement   : imbalance={better.imbalance_factor:.2f} "
          f"dropped={better.dropped_token_frac:.1%}")
    print("  per-gpu load (rebalanced):", better.per_gpu_load)
    print("\npaper hook: replace greedy_rebalance with a learned placement / "
          "dynamic capacity policy and measure imbalance vs all2all volume.")


if __name__ == "__main__":
    main()
