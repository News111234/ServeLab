"""Tensor-parallel primitives (Megatron/vLLM-style, reference implementation).

Two execution backends behind one `Collective` interface:
    - SimulatedCollective : W rank-threads in ONE process, synchronized at each
      collective (works on CPU / CI, validates the sharding MATH)
    - GlooCollective      : real torch.distributed process group (NCCL would be
      a drop-in on GPU machines)

Sharding scheme (identical to Megatron TP):
    Column-parallel (output-dim split, no comm):  q/k/v_proj, gate/up_proj, embed(vocab)
    Row-parallel  (input-dim split, all-reduce):  o_proj, down_proj
    Attention heads and KV heads are sharded per rank; attention itself needs
    NO communication (heads are independent) -- the only collectives per layer
    are the two row-parallel all-reduces.
"""

import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Collective(ABC):
    """Minimal collective surface the TP layers need."""

    world_size: int

    @abstractmethod
    def all_reduce_sum(self, t: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def all_gather_cat(self, t: torch.Tensor, dim: int) -> torch.Tensor: ...

    def sync_object(self, obj, src: int = 0):
        """Return `obj` from rank `src` on every rank (default impl: local)."""
        return obj if self.rank == src else None

    @property
    def rank(self) -> int:
        raise NotImplementedError


class SimulatedCollective(Collective):
    """W worker threads in one process meet at each collective call.

    Rank is thread-local: bind it with `rank_scope(r)` inside each worker
    thread. Each (step, op) slot collects one partial per rank; the last
    arriving thread computes the result and every thread takes a copy.
    Deterministic, CPU-only, runs in CI.
    """

    def __init__(self, world_size: int):
        assert world_size >= 1
        self.world_size = world_size
        self._tls = threading.local()
        self._slots: Dict[str, Dict[int, torch.Tensor]] = {"_last": {}}
        self._cv = threading.Condition()
        self._step = 0
        self._error: Optional[BaseException] = None

    def abort(self, err: BaseException) -> None:
        """Poison the collective: one rank crashed, so the others must raise
        instead of waiting forever at their next meet point."""
        with self._cv:
            self._error = err
            self._cv.notify_all()

    @contextmanager
    def rank_scope(self, rank: int):
        self._tls.rank = rank
        try:
            yield self
        finally:
            self._tls.rank = None

    @property
    def rank(self) -> int:
        r = getattr(self._tls, "rank", None)
        if r is None:
            raise RuntimeError("SimulatedCollective used outside rank_scope()")
        return r

    def all_reduce_sum(self, t: torch.Tensor) -> torch.Tensor:
        return self._meet("rs", t, lambda parts: torch.stack(parts).sum(0))

    def all_gather_cat(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        return self._meet("ga", t,
                          lambda parts: torch.cat(list(parts), dim=dim))

    def sync_object(self, obj, src: int = 0):
        if self.rank == src:
            with self._cv:
                self._slots["_last"][f"obj{self._step}"] = ("obj", obj)
                self._cv.notify_all()
                return obj
        with self._cv:
            key = f"obj{self._step}"
            self._cv.wait_for(lambda: key in self._slots["_last"])
            return self._slots["_last"][key][1]

    def _meet(self, op: str, t: torch.Tensor, combine) -> torch.Tensor:
        key = f"{self._step}:{op}"
        with self._cv:
            if self._error is not None:
                raise RuntimeError("collective aborted: a peer rank crashed") \
                    from self._error
            slot = self._slots.setdefault(key, {})
            slot[self.rank] = t.detach().clone()
            if len(slot) < self.world_size:
                self._cv.wait_for(lambda: key not in self._slots
                                  or self._error is not None)
                if self._error is not None and key in self._slots:
                    raise RuntimeError("collective aborted: a peer rank crashed") \
                        from self._error
                return self._slots["_last"][key].clone()
            result = combine([slot[r] for r in range(self.world_size)])
            self._slots["_last"][key] = result
            del self._slots[key]
            self._step += 1
            self._cv.notify_all()
        return result.clone()


class GlooCollective(Collective):
    """torch.distributed process group (gloo on CPU; swap for NCCL on GPU)."""

    def __init__(self, rank: int, world_size: int, init_method: str):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="gloo", rank=rank, world_size=world_size,
                init_method=init_method)
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:            # override the plain attribute
        return self._world_size

    def all_reduce_sum(self, t: torch.Tensor) -> torch.Tensor:
        t = t.contiguous()
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        return t

    def all_gather_cat(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        parts = [torch.empty_like(t.contiguous())
                 for _ in range(self.world_size)]
        torch.distributed.all_gather(parts, t.contiguous())
        return torch.cat(parts, dim=dim)

    def sync_object(self, obj, src: int = 0):
        box = [obj if self._rank == src else None]
        torch.distributed.broadcast_object_list(box, src=src)
        return box[0]


class ColumnParallelLinear(nn.Module):
    """Y = X W^T with W [out, in] split along `out`. No communication.

    NOTE: takes RANK-LOCAL dims -- global->local slicing is
    `shard_state_dict_tp`'s job, the layer never divides by world itself.
    """

    def __init__(self, in_features: int, out_features_local: int,
                 collective: Collective, bias: bool = True):
        super().__init__()
        self.collective = collective
        self.weight = nn.Parameter(torch.empty(out_features_local, in_features))
        self.bias = (nn.Parameter(torch.zeros(out_features_local))
                     if bias else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Y = X W^T with W [out, in] split along `in`. All-reduce on output.
    Takes RANK-LOCAL in_features."""

    def __init__(self, in_features_local: int, out_features: int,
                 collective: Collective, bias: bool = True):
        super().__init__()
        self.collective = collective
        self.weight = nn.Parameter(torch.empty(out_features, in_features_local))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        partial = F.linear(x, self.weight)
        out = self.collective.all_reduce_sum(partial)
        return out + self.bias if self.bias is not None else out


class VocabParallelEmbedding(nn.Module):
    """Embedding split along the vocab dim; out-of-shard ids contribute 0.
    Takes the RANK-LOCAL vocab count."""

    def __init__(self, num_embeddings_local: int, embedding_dim: int,
                 collective: Collective):
        super().__init__()
        self.collective = collective
        self.shard = num_embeddings_local
        self.weight = nn.Parameter(torch.empty(self.shard, embedding_dim))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        rank = self.collective.rank            # resolved at call time: the
        start = rank * self.shard              # sim runs ranks in threads,
        end = start + self.shard               # gloo runs them in processes
        mask = (ids >= start) & (ids < end)
        local = (ids - start).clamp(0, self.shard - 1)
        emb = F.embedding(local, self.weight)
        emb = emb * mask.unsqueeze(-1).to(emb.dtype)
        return self.collective.all_reduce_sum(emb)


class VocabParallelLMHead(nn.Module):
    """Logits computed per vocab shard, then all-gathered along vocab.
    Takes the RANK-LOCAL vocab count."""

    def __init__(self, vocab_local: int, hidden: int, collective: Collective):
        super().__init__()
        self.collective = collective
        self.weight = nn.Parameter(torch.empty(vocab_local, hidden))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        local = F.linear(hidden, self.weight)
        return self.collective.all_gather_cat(local, dim=-1)


def slice_shard(tensor: torch.Tensor, dim: int, rank: int, world: int) -> torch.Tensor:
    return tensor.tensor_split(world, dim=dim)[rank].contiguous()


def shard_state_dict_tp(state: Dict[str, torch.Tensor], rank: int, world: int,
                        num_heads: int, num_kv_heads: int, head_dim: int,
                        vocab_size: int) -> Dict[str, torch.Tensor]:
    """Full HF-style state dict -> this rank's shard (Megatron split rules)."""
    out = {}
    for name, t in state.items():
        short = name[len("model."):] if name.startswith("model.") else name
        if short.endswith("rotary_emb.inv_freq"):
            continue
        if short == "embed_tokens.weight":
            out[short] = slice_shard(t, 0, rank, world)          # vocab split
        elif short == "lm_head.weight":
            out[short] = slice_shard(t, 0, rank, world)          # vocab split
        elif short.endswith("q_proj.weight"):
            out[short] = slice_shard(t, 0, rank, world)          # heads split
        elif short.endswith("q_proj.bias"):
            out[short] = slice_shard(t, 0, rank, world)
        elif short.endswith("k_proj.weight") or short.endswith("k_proj.bias"):
            out[short] = slice_shard(t, 0, rank, world)          # kv heads split
        elif short.endswith("v_proj.weight") or short.endswith("v_proj.bias"):
            out[short] = slice_shard(t, 0, rank, world)
        elif short.endswith("o_proj.weight"):
            out[short] = slice_shard(t, 1, rank, world)          # input(heads) split
        elif short.endswith("gate_proj.weight") or short.endswith("up_proj.weight"):
            out[short] = slice_shard(t, 0, rank, world)          # column parallel
        elif short.endswith("down_proj.weight"):
            out[short] = slice_shard(t, 1, rank, world)          # row parallel
        else:                                                    # norms etc: replicate
            out[short] = t.contiguous()
    return out
