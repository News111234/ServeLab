"""Sampling parameters (subset of the OpenAI/vLLM surface, torch-free)."""

from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1                    # -1: disabled
    max_tokens: int = 16
    ignore_eos: bool = False
    seed: int = -1                     # -1: nondeterministic
    stop_token_ids: tuple = ()

    def __post_init__(self):
        if self.temperature < 0.0:
            raise ValueError("temperature must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k != -1 and self.top_k < 1:
            raise ValueError("top_k must be -1 or >= 1")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

    @staticmethod
    def greedy(max_tokens: int = 16) -> "SamplingParams":
        return SamplingParams(temperature=0.0, max_tokens=max_tokens)
