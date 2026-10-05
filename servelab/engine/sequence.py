"""Request / sequence state machine (torch-free)."""

import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .sampling_params import SamplingParams


class SequenceStatus(str, Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    FINISHED_EOS = "FINISHED_EOS"
    FINISHED_LENGTH = "FINISHED_LENGTH"
    FINISHED_ABORTED = "FINISHED_ABORTED"


FINISHED_STATES = {
    SequenceStatus.FINISHED_EOS,
    SequenceStatus.FINISHED_LENGTH,
    SequenceStatus.FINISHED_ABORTED,
}


@dataclass
class RequestMetrics:
    arrival_time: float = 0.0
    first_scheduled_time: float = 0.0
    first_token_time: float = 0.0
    finished_time: float = 0.0
    num_preemptions: int = 0

    @property
    def ttft(self) -> float:
        return self.first_token_time - self.arrival_time

    @property
    def e2e_latency(self) -> float:
        return self.finished_time - self.arrival_time

    def to_dict(self) -> dict:
        return {
            "ttft": self.ttft,
            "latency": self.e2e_latency,
            "preemptions": self.num_preemptions,
        }


class Sequence:
    """One generation stream (prompt + generated tokens). No beam search:
    one sequence per request (see roadmap for n-best extension)."""

    def __init__(
        self,
        request_id: str,
        prompt_token_ids: List[int],
        sampling_params: SamplingParams,
        priority: int = 0,
        arrival_time: Optional[float] = None,
        prefix_token_ids: Optional[List[int]] = None,
    ):
        self.request_id = request_id
        self.priority = priority
        self.arrival_time = time.monotonic() if arrival_time is None else arrival_time
        self.sampling_params = sampling_params
        self.prefix_token_ids = prefix_token_ids or []
        self.prompt_token_ids = list(prompt_token_ids)
        self.token_ids: List[int] = self.prefix_token_ids + self.prompt_token_ids
        self.num_prompt_tokens = len(self.token_ids)  # prefix counts as prompt
        self.num_computed_tokens = 0                  # tokens whose KV is stored
        self.status = SequenceStatus.WAITING
        self.metrics = RequestMetrics(arrival_time=self.arrival_time)
        self.sampled_token: Optional[int] = None

    # ------------------------------------------------------------- accessors
    @property
    def num_output_tokens(self) -> int:
        return len(self.token_ids) - self.num_prompt_tokens

    @property
    def output_token_ids(self) -> List[int]:
        return self.token_ids[self.num_prompt_tokens:]

    def get_len(self) -> int:
        return len(self.token_ids)

    def is_finished(self) -> bool:
        return self.status in FINISHED_STATES

    def is_prefill_done(self) -> bool:
        return self.num_computed_tokens >= self.num_prompt_tokens

    def remaining_prompt_tokens(self) -> int:
        return self.num_prompt_tokens - self.num_computed_tokens

    # ------------------------------------------------------------- mutations
    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        self.sampled_token = token_id

    def on_preempt_recompute(self) -> None:
        """Recompute-mode preemption: forget computed state; all tokens
        (prompt + generated) become the new prompt. KV blocks were released;
        prefix caching will likely recover them on re-run."""
        self.num_computed_tokens = 0
        self.num_prompt_tokens = len(self.token_ids)
        self.metrics.num_preemptions += 1
        self.status = SequenceStatus.WAITING

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Sequence({self.request_id}, len={self.get_len()}, "
                f"computed={self.num_computed_tokens}, status={self.status.value})")
