"""Tensor parallelism: Megatron-style TP layers, sharding, simulated + gloo
execution backends."""

from .layers import (
    Collective, ColumnParallelLinear, GlooCollective, RowParallelLinear,
    SimulatedCollective, VocabParallelEmbedding, VocabParallelLMHead,
    shard_state_dict_tp, slice_shard,
)
from .tp_model import TPDecoderModel, TPDecoderLayer
from .simulated import SimulatedTPModelGroup, run_rank_threads

__all__ = [
    "Collective", "ColumnParallelLinear", "RowParallelLinear",
    "VocabParallelEmbedding", "VocabParallelLMHead", "shard_state_dict_tp",
    "slice_shard", "SimulatedCollective", "GlooCollective",
    "TPDecoderModel", "TPDecoderLayer", "SimulatedTPModelGroup",
    "run_rank_threads",
]
