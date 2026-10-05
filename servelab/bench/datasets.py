"""Prompt datasets for end-to-end benchmarks."""

import json
import random
from typing import List


def sample_synthetic_prompts(num_prompts: int, input_mean: int = 200,
                             input_std: int = 40, output_mean: int = 100,
                             seed: int = 42) -> List[dict]:
    """vLLM-bench-style synthetic workload: random token ids."""
    rng = random.Random(seed)
    prompts = []
    for i in range(num_prompts):
        n_in = max(8, int(rng.gauss(input_mean, input_std)))
        # ids in a plausible range; the engine only needs token ids
        ids = [rng.randrange(1000, 50000) for _ in range(n_in)]
        prompts.append({"prompt_token_ids": ids,
                        "expected_output_len": max(1, int(rng.gauss(output_mean, 20)))})
    return prompts


def sample_sharegpt_prompts(path: str, num_prompts: int,
                            tokenizer=None) -> List[dict]:
    """First human turn of each ShareGPT conversation."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = []
    for conv in data:
        turns = conv.get("conversations", [])
        if turns and turns[0].get("from") == "human":
            text = turns[0]["value"]
            if tokenizer is not None:
                ids = tokenizer.encode(text)
            else:
                ids = None
            prompts.append({"prompt": text, "prompt_token_ids": ids})
        if len(prompts) >= num_prompts:
            break
    return prompts
