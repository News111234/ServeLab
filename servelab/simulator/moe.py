"""MoE expert-parallel load simulation (torch-free).

Models the load-balance / all-to-all side of MoE serving (cf. DeepEP,
DeepSeek-V3's auxiliary-loss-free balancing, expert placement work):
    - tokens routed top-k by a synthetic gate (Zipf expert popularity)
    - experts sharded across EP GPUs -> per-GPU token load
    - all2all dispatch volume and capacity-factor token drops
    - a greedy expert-placement rebalancing baseline to beat

Paper hooks (roadmap #6): dynamic capacity factors, learned placement,
prediction-guided rebalancing, overlap of dispatch with grouped GEMM.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class MoEResult:
    imbalance_factor: float          # max_gpu_load / mean_gpu_load
    dropped_token_frac: float
    all2all_tokens: int              # dispatched (token, expert) pairs volume driver
    per_gpu_load: List[int]
    expert_loads: List[int]


def route_tokens(num_tokens: int, num_experts: int, top_k: int,
                 zipf_alpha: float, rng: random.Random) -> List[int]:
    """Expert popularity follows a Zipf law -- real MoE workloads are
    notoriously imbalanced (a few experts take most tokens)."""
    weights = [1.0 / (i + 1) ** zipf_alpha for i in range(num_experts)]
    total = sum(weights)
    experts = []
    for _ in range(num_tokens):
        chosen = set()
        while len(chosen) < top_k:
            u, acc = rng.random() * total, 0.0
            for e, w in enumerate(weights):
                acc += w
                if u <= acc:
                    chosen.add(e)
                    break
            else:
                chosen.add(rng.randrange(num_experts))
        experts.extend(chosen)
    return experts


def simulate_moe_layer(
    num_tokens: int = 2048,
    num_experts: int = 64,
    top_k: int = 6,
    num_gpus: int = 4,
    hidden_size: int = 4096,
    capacity_factor: float = 1.25,
    zipf_alpha: float = 1.2,
    placement: Optional[Dict[int, int]] = None,   # expert -> gpu
    seed: int = 0,
) -> MoEResult:
    rng = random.Random(seed)
    if placement is None:
        placement = {e: e * num_gpus // num_experts for e in range(num_experts)}
    pairs = route_tokens(num_tokens, num_experts, top_k, zipf_alpha, rng)
    expert_loads = [0] * num_experts
    for e in pairs:
        expert_loads[e] += 1
    cap = int(capacity_factor * num_tokens * top_k / num_experts)
    dropped = sum(max(0, c - cap) for c in expert_loads)
    gpu_load = [0] * num_gpus
    for e, c in enumerate(expert_loads):
        gpu_load[placement[e]] += min(c, cap)
    mean = sum(gpu_load) / num_gpus
    imb = (max(gpu_load) / mean) if mean > 0 else 0.0
    return MoEResult(
        imbalance_factor=imb,
        dropped_token_frac=dropped / max(1, len(pairs)),
        all2all_tokens=len(pairs),
        per_gpu_load=gpu_load,
        expert_loads=expert_loads,
    )


def greedy_rebalance(expert_loads: List[int], num_gpus: int,
                     placement: Optional[Dict[int, int]] = None) -> Dict[int, int]:
    """Baseline placement policy: sort experts by load, greedily bin-pack
    onto the currently-lightest GPU. Replace with your learned policy here."""
    placement = placement or {e: 0 for e in range(len(expert_loads))}
    gpu_load = [0] * num_gpus
    for e in sorted(range(len(expert_loads)), key=lambda e: -expert_loads[e]):
        g = min(range(num_gpus), key=lambda g: gpu_load[g])
        placement[e] = g
        gpu_load[g] += expert_loads[e]
    return placement


def rebalanced_sim(num_tokens: int = 2048, num_experts: int = 64, top_k: int = 6,
                   num_gpus: int = 4, capacity_factor: float = 1.25,
                   zipf_alpha: float = 1.2, seed: int = 0) -> Tuple[MoEResult, MoEResult]:
    rng = random.Random(seed)
    pairs = route_tokens(num_tokens, num_experts, top_k, zipf_alpha, rng)
    expert_loads = [0] * num_experts
    for e in pairs:
        expert_loads[e] += 1
    base = simulate_moe_layer(num_tokens, num_experts, top_k, num_gpus,
                              capacity_factor=capacity_factor, seed=seed,
                              placement=None)
    better_placement = greedy_rebalance(expert_loads, num_gpus)
    better = simulate_moe_layer(num_tokens, num_experts, top_k, num_gpus,
                                capacity_factor=capacity_factor, seed=seed,
                                placement=better_placement)
    return base, better
