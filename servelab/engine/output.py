"""Engine outputs (mirrors vLLM's RequestOutput shape)."""

from dataclasses import dataclass, field
from typing import List, Optional

from .sequence import Sequence


@dataclass
class CompletionOutput:
    index: int = 0
    text: str = ""
    token_ids: List[int] = field(default_factory=list)
    finish_reason: Optional[str] = None    # "stop" | "length" | "abort"


@dataclass
class RequestOutput:
    request_id: str
    prompt: str = ""
    prompt_token_ids: List[int] = field(default_factory=list)
    outputs: List[CompletionOutput] = field(default_factory=list)
    finished: bool = False
    metrics: Optional[dict] = None

    @classmethod
    def from_sequence(cls, seq: Sequence, detokenize=None) -> "RequestOutput":
        out = CompletionOutput(
            index=0,
            token_ids=list(seq.output_token_ids),
            finish_reason=seq.status.value.replace("FINISHED_", "").lower(),
        )
        if detokenize is not None:
            out.text = detokenize(out.token_ids)
        return cls(
            request_id=seq.request_id,
            prompt_token_ids=list(seq.prompt_token_ids),
            outputs=[out],
            finished=seq.is_finished(),
            metrics=seq.metrics.to_dict(),
        )
