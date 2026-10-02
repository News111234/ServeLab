"""Engine package. torch-backed modules (engine.py) are imported lazily so
the torch-free parts (scheduler, sequence, sampling_params, output) can be
used by the simulator on machines without torch.
"""

from .sampling_params import SamplingParams
from .sequence import Sequence, SequenceStatus, RequestMetrics
from .output import RequestOutput, CompletionOutput
from .scheduler import (
    Scheduler, SchedulerOutput, SchedulingPolicy, FCFSPolicy,
    PriorityPolicy, SJFPredictedPolicy, build_scheduler_policy,
)

_LAZY = {
    "LLMEngine": ("servelab.engine.engine", "LLMEngine"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod_name, attr = _LAZY[name]
        return getattr(importlib.import_module(mod_name), attr)
    raise AttributeError(name)


__all__ = [
    "SamplingParams", "Sequence", "SequenceStatus", "RequestMetrics",
    "RequestOutput", "CompletionOutput",
    "Scheduler", "SchedulerOutput", "SchedulingPolicy", "FCFSPolicy",
    "PriorityPolicy", "SJFPredictedPolicy", "build_scheduler_policy",
    "LLMEngine",
]
