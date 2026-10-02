"""Speculative decoding tests.

Hard contract: greedy spec output == plain greedy output of the same target,
regardless of proposer quality (self-draft, n-gram, random draft).
"""

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from helpers_tiny_model import build_tiny_qwen2  # noqa: E402

from servelab.config import ModelConfig  # noqa: E402
from servelab.engine.engine import LLMEngine  # noqa: E402
from servelab.engine.sampling_params import SamplingParams  # noqa: E402
from servelab.engine.speculative import (  # noqa: E402
    ModelDraftProposer, PromptLookupProposer, SimpleKVRunner,
    SpeculativeDecoder,
)
from servelab.models.decoder import DecoderModel  # noqa: E402
from servelab.models.loader import load_model  # noqa: E402

PROMPT = list(range(10, 30))


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = tmp_path_factory.mktemp("spec") / "tiny"
    cfg = build_tiny_qwen2(str(path))
    mc = ModelConfig.from_hf(cfg, path=str(path))
    _, state = load_model(str(path))
    m = DecoderModel(mc, device="cpu", dtype=torch.float32)
    m.load_state_dict_hf(state)
    m.eval()
    return mc, m


def plain_greedy(m, prompt, max_tokens):
    runner = SimpleKVRunner(m)
    out = []
    logits = runner.prefill(prompt)[-1]        # prefill -> [T, V]
    for _ in range(max_tokens):
        tok = int(torch.argmax(logits).item())
        out.append(tok)
        logits = runner.decode_step(tok)       # decode -> [V] (no extra [-1]!)
    return out


def test_self_draft_matches_greedy_and_full_acceptance(model):
    mc, m = model
    baseline = plain_greedy(m, PROMPT, 16)

    target = SimpleKVRunner(m)
    draft = ModelDraftProposer(SimpleKVRunner(m), k=4)   # same weights
    spec = SpeculativeDecoder(target, draft, k=4)
    got = spec.generate_greedy(PROMPT, 16)
    assert got == baseline
    assert spec.stats.acceptance_rate == 1.0
    assert spec.stats.target_amortization > 2.0     # ~k+1 tokens per forward


def test_prompt_lookup_matches_greedy(model):
    mc, m = model
    baseline = plain_greedy(m, PROMPT, 16)
    target = SimpleKVRunner(m)
    spec = SpeculativeDecoder(target, PromptLookupProposer(ngram_size=3), k=4)
    got = spec.generate_greedy(PROMPT, 16)
    assert got == baseline


def test_random_draft_matches_greedy_and_exercises_rewind(model):
    """A mismatching draft forces the rewind path on target AND draft."""
    mc, m = model
    baseline = plain_greedy(m, PROMPT, 16)

    # draft with perturbed weights -> argmax differs somewhere
    draft_m = DecoderModel(mc, device="cpu", dtype=torch.float32)
    _, state = load_model(mc.path)
    for name in list(state):
        if "lm_head" in name or name.startswith("model.norm"):
            continue
        state[name] = state[name] * 1.9
    draft_m.load_state_dict_hf(state)
    draft_m.eval()

    target = SimpleKVRunner(m)
    spec = SpeculativeDecoder(target, ModelDraftProposer(SimpleKVRunner(draft_m), k=4), k=4)
    got = spec.generate_greedy(PROMPT, 16)
    assert got == baseline
    assert spec.stats.acceptance_rate < 1.0 or spec.stats.draft_proposed == 0


def test_greedy_matches_llm_engine(model):
    mc, m = model
    from servelab.config import CacheConfig, SchedulerConfig
    engine = LLMEngine(mc, CacheConfig(num_blocks=32),
                       SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=64),
                       seed=0, device="cpu")
    ref = engine.generate([PROMPT], SamplingParams.greedy(max_tokens=16))
    assert plain_greedy(m, PROMPT, 16) == ref[0].outputs[0].token_ids


def test_sampling_runs_and_is_deterministic(model):
    mc, m = model
    target = SimpleKVRunner(m)
    spec = SpeculativeDecoder(target, ModelDraftProposer(SimpleKVRunner(m), k=3), k=3)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = spec.generate_sampling(PROMPT, 12, temperature=1.0, generator=g1)
    b = spec.generate_sampling(PROMPT, 12, temperature=1.0, generator=g2)
    assert a == b and len(a) == 12
    assert all(0 <= t < mc.vocab_size for t in a)


def test_max_tokens_one(model):
    mc, m = model
    target = SimpleKVRunner(m)
    spec = SpeculativeDecoder(target, PromptLookupProposer(), k=4)
    assert len(spec.generate_greedy(PROMPT, 1)) == 1
