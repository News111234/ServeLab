"""Logits -> token sampler (temperature / top-k / top-p, greedy)."""

from typing import List, Sequence

import torch

from .sampling_params import SamplingParams


class Sampler:
    def __init__(self, seed: int = 0):
        self.gen = torch.Generator()
        self.gen.manual_seed(seed)

    def sample(self, logits: torch.Tensor,          # [B, V] float32
               params: Sequence[SamplingParams]) -> List[int]:
        tokens = []
        for row, sp in zip(logits, params):
            tokens.append(self._sample_one(row, sp))
        return tokens

    def _sample_one(self, logits: torch.Tensor, sp: SamplingParams) -> int:
        if sp.temperature == 0.0:
            return int(torch.argmax(logits).item())
        gen = self.gen
        if sp.seed >= 0:                       # per-request deterministic seed
            gen = torch.Generator()
            gen.manual_seed(sp.seed)
        probs = torch.softmax(logits / max(sp.temperature, 1e-6), dim=-1)
        if sp.top_k > 0:
            k = min(sp.top_k, probs.shape[-1])
            topv, topi = torch.topk(probs, k)
            topv = topv / topv.sum()
            idx = topi[torch.multinomial(topv, 1, generator=gen)]
            return int(idx.item())
        if sp.top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cum > sp.top_p
            cutoff[..., 0] = False             # always keep the first token
            sorted_probs[cutoff] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum()
            idx = sorted_idx[torch.multinomial(sorted_probs, 1, generator=gen)]
            return int(idx.item())
        return int(torch.multinomial(probs, 1, generator=gen).item())
