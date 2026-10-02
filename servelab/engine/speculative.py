"""Speculative decoding (Leviathan et al. '23 / Chen et al. '23).

Algorithm per round (greedy variant):
    1. a cheap proposer suggests k tokens conditioned on the context;
    2. the TARGET model teacher-forces [last_token + k drafts] in ONE pass,
       producing k+1 next-token distributions;
    3. tokens are accepted while the target agrees with the draft; on the
       first disagreement the target's own token is emitted (the correction),
       and if everything matched, the (k+1)-th target token is a free bonus.

Correctness contract (tested): greedy speculative output is IDENTICAL to
plain greedy decoding of the same target, for ANY proposer quality.

Implementation notes:
    - `SimpleKVRunner` is a single-sequence, non-paged KV runner reusing the
      same DecoderModel weights. Scheduler-level batched speculation (spec
      decoding under continuous batching, cf. vLLM V1 spec-decode) is a
      roadmap item; this module isolates the algorithm.
    - temperature>0 uses the standard rejection sampler with residual
      distribution norm(max(p - q, 0)) and requires a model-based proposer;
      the n-gram proposer is greedy-only (same as vLLM prompt-lookup).
"""

from dataclasses import dataclass
from typing import List, Optional, Protocol

import torch

from ..models.decoder import DecoderModel, apply_rope


class KVRunner(Protocol):
    """Single-sequence incremental inference over a fixed decoder model."""

    cache_len: int

    def prefill(self, token_ids: List[int]) -> torch.Tensor: ...  # [T, V]
    def decode_step(self, token_id: int) -> torch.Tensor: ...     # [V]
    def rewind(self, new_cache_len: int) -> None: ...


class SimpleKVRunner:
    """Non-paged growable per-layer KV cache for ONE sequence. Works with any
    decoder exposing the DecoderModel interface (incl. TPDecoderModel)."""

    def __init__(self, model: DecoderModel):
        self.model = model
        self.cache_len = 0
        self.n_layers = model.cfg.num_layers
        self.k_cache: List[List[torch.Tensor]] = [[] for _ in range(self.n_layers)]
        self.v_cache: List[List[torch.Tensor]] = [[] for _ in range(self.n_layers)]

    @torch.no_grad()
    def _forward(self, token_ids: List[int]) -> torch.Tensor:
        m = self.model
        device = next(m.parameters()).device
        ids = torch.tensor(token_ids, dtype=torch.long, device=device)
        start = self.cache_len
        positions = torch.arange(start, start + len(token_ids), device=device)
        hidden = m.embed_tokens(ids)
        groups = m.cfg.num_heads // m.cfg.num_kv_heads
        T = len(token_ids)
        for i, layer in enumerate(m.layers):
            residual = hidden
            x = layer.input_layernorm(hidden)
            q, k, v = layer.self_attn.qkv(x)
            q, k = apply_rope(q, k, positions, m.inv_freq)
            self.k_cache[i].append(k)                    # chunk [T, kvH, D]
            self.v_cache[i].append(v)
            k_all = torch.cat(self.k_cache[i], dim=0)    # [ctx, kvH, D]
            v_all = torch.cat(self.v_cache[i], dim=0)
            if groups > 1:
                k_all = k_all.repeat_interleave(groups, dim=1)
                v_all = v_all.repeat_interleave(groups, dim=1)
            scores = torch.einsum("qhd,rhd->hqr", q.float(), k_all.float()) \
                * layer.self_attn.scale
            q_pos = positions[:, None]
            k_pos = torch.arange(k_all.shape[0], device=device)[None, :]
            scores = scores.masked_fill((q_pos < k_pos).unsqueeze(0),
                                        float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            o = torch.einsum("hqr,rhd->qhd", probs, v_all.float())
            hidden = residual + layer.self_attn.out(
                o.reshape(T, -1).to(residual.dtype))
            residual = hidden
            hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
        return m.lm_head_forward(m.norm(hidden)).float()             # [T, V]

    def prefill(self, token_ids: List[int]) -> torch.Tensor:
        logits = self._forward(token_ids)
        self.cache_len += len(token_ids)
        return logits

    def decode_step(self, token_id: int) -> torch.Tensor:
        logits = self._forward([token_id])
        self.cache_len += 1
        return logits[-1]

    def rewind(self, new_cache_len: int) -> None:
        assert new_cache_len <= self.cache_len
        drop = self.cache_len - new_cache_len
        if drop == 0:
            return
        for i in range(self.n_layers):
            for cache in (self.k_cache, self.v_cache):
                chunks, kept = [], self.cache_len - drop
                for c in cache[i]:
                    if kept <= 0:
                        break
                    take = min(kept, c.shape[0])
                    chunks.append(c[:take])
                    kept -= take
                cache[i] = chunks
        self.cache_len = new_cache_len


class DraftProposer(Protocol):
    def propose(self, context: List[int], k: int) -> List[int]: ...


class PromptLookupProposer:
    """n-gram lookup in the prompt + generated text (no model, greedy-only).
    Strong on repetitive/extractive workloads (code, RAG quotes, editing)."""

    def __init__(self, ngram_size: int = 4):
        self.ngram_size = ngram_size

    def propose(self, context: List[int], k: int) -> List[int]:
        n = self.ngram_size
        if len(context) < n + 1:
            return []
        suffix = tuple(context[-n:])
        for i in range(len(context) - n):
            if tuple(context[i:i + n]) == suffix:
                return list(context[i + n:i + n + k])
        return []


class ModelDraftProposer:
    """A smaller (or the same) decoder used as an autoregressive draft.
    Keeps its own KV runner and re-syncs it to the accepted context."""

    def __init__(self, runner: KVRunner, k: int):
        self.runner = runner
        self.k = k
        self.proposal_probs: List[torch.Tensor] = []   # q dists of last round

    def _sync_context(self, context: List[int]) -> None:
        want = len(context) - 1                        # KV covers context[:-1]
        if self.runner.cache_len > want:
            self.runner.rewind(want)
        elif self.runner.cache_len < want:
            self.runner.prefill(context[self.runner.cache_len:want])

    def propose(self, context: List[int], k: int,
                temperature: float = 1.0) -> List[int]:
        self._sync_context(context)
        logits = self.runner.decode_step(context[-1])
        self.proposal_probs = []
        drafts = []
        for _ in range(k):
            self.proposal_probs.append(torch.softmax(logits / temperature, -1))
            drafts.append(int(torch.argmax(logits).item()))
            if len(drafts) < k:
                logits = self.runner.decode_step(drafts[-1])
        return drafts


@dataclass
class SpecStats:
    rounds: int = 0
    draft_proposed: int = 0
    accepted: int = 0
    target_positions: int = 0       # tokens teacher-forced through the target
    target_calls: int = 0           # forward invocations (the wall-clock unit:
                                    # a k+1-token verify pass costs ~one decode
                                    # step on a memory-bound GPU)
    emitted: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / max(1, self.draft_proposed)

    @property
    def target_amortization(self) -> float:
        """emitted tokens per target forward CALL (>1 = fewer launches than
        plain autoregressive decoding; this is the speculative speedup source)."""
        return self.emitted / max(1, self.target_calls)


class SpeculativeDecoder:
    def __init__(self, target: KVRunner, proposer: DraftProposer, k: int = 4):
        self.target = target
        self.proposer = proposer
        self.k = k
        self.stats = SpecStats()

    # ------------------------------------------------------------------ util
    def _verify(self, tokens: List[int], drafts: List[int]) -> List[int]:
        """Teacher-force [last + drafts]; returns emitted tokens and rewinds
        the target cache to match what was emitted."""
        verify_in = [tokens[-1]] + drafts
        rows = self.target.prefill(verify_in)                  # [k+1, V]
        self.stats.target_positions += len(verify_in)
        self.stats.target_calls += 1
        emitted: List[int] = []
        for i, d in enumerate(drafts):
            pred = int(torch.argmax(rows[i]).item())
            emitted.append(pred)
            if pred == d:
                self.stats.accepted += 1
            else:
                break                                           # correction
        else:
            emitted.append(int(torch.argmax(rows[len(drafts)]).item()))  # bonus
        self.target.rewind(len(tokens) - 1 + len(emitted))
        return emitted

    # ---------------------------------------------------------------- greedy
    @torch.no_grad()
    def generate_greedy(self, prompt: List[int], max_tokens: int) -> List[int]:
        t, st = self.target, self.stats
        t.rewind(0)                        # reusable across generate() calls
        tokens = list(prompt)              # `tokens` is the FULL context
        logits = t.prefill(prompt)[-1]
        st.target_positions += len(prompt)
        st.target_calls += 1
        tokens.append(int(torch.argmax(logits).item()))
        st.emitted += 1
        while len(tokens) - len(prompt) < max_tokens:
            drafts = self.proposer.propose(tokens, self.k)
            st.rounds += 1
            if not drafts:
                logits = t.decode_step(tokens[-1])
                st.target_positions += 1
                st.target_calls += 1
                tokens.append(int(torch.argmax(logits).item()))
                st.emitted += 1
                continue
            st.draft_proposed += len(drafts)
            emitted = self._verify(tokens, drafts)
            tokens.extend(emitted)
            st.emitted += len(emitted)
        return tokens[len(prompt):len(prompt) + max_tokens]

    # -------------------------------------------------------------- sampling
    @torch.no_grad()
    def generate_sampling(self, prompt: List[int], max_tokens: int,
                          temperature: float = 1.0,
                          generator: Optional[torch.Generator] = None) -> List[int]:
        """Standard rejection sampling; requires ModelDraftProposer (needs a
        draft distribution q). Output distribution == target distribution."""
        assert isinstance(self.proposer, ModelDraftProposer)
        t = self.target
        t.rewind(0)
        tokens = list(prompt)
        logits = t.prefill(prompt)[-1]
        self.stats.target_positions += len(prompt)
        self.stats.target_calls += 1
        tokens.append(int(torch.multinomial(
            torch.softmax(logits / temperature, -1), 1,
            generator=generator).item()))
        self.stats.emitted += 1
        while len(tokens) - len(prompt) < max_tokens:
            drafts = self.proposer.propose(tokens, self.k,
                                           temperature=temperature)
            self.stats.rounds += 1
            self.stats.draft_proposed += len(drafts)
            if not drafts:
                logits = t.decode_step(tokens[-1])
                self.stats.target_positions += 1
                self.stats.target_calls += 1
                tokens.append(int(torch.multinomial(
                    torch.softmax(logits / temperature, -1), 1,
                    generator=generator).item()))
                self.stats.emitted += 1
                continue
            verify_in = [tokens[-1]] + drafts
            rows = t.prefill(verify_in)
            self.stats.target_positions += len(verify_in)
            self.stats.target_calls += 1
            p = torch.softmax(rows / temperature, dim=-1)       # [k+1, V]
            r = torch.rand(len(drafts), generator=generator)
            emitted: List[int] = []
            for i, d in enumerate(drafts):
                q = self.proposer.proposal_probs[i]
                if r[i] < (p[i, d] / q[d]).item():
                    emitted.append(d)
                    self.stats.accepted += 1
                else:
                    residual = torch.clamp(p[i] - q, min=0.0)
                    if residual.sum() <= 0:
                        residual = p[i]        # p == q everywhere: sample target
                    emitted.append(int(torch.multinomial(
                        residual / residual.sum(), 1,
                        generator=generator).item()))
                    break
            else:
                emitted.append(int(torch.multinomial(
                    p[len(drafts)], 1, generator=generator).item()))
            t.rewind(len(tokens) - 1 + len(emitted))
            tokens.extend(emitted)
            self.stats.emitted += len(emitted)
        return tokens[len(prompt):len(prompt) + max_tokens]
