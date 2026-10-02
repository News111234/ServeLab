"""Step-by-step trace of one self-draft speculation round."""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from helpers_tiny_model import build_tiny_qwen2
from servelab.config import ModelConfig
from servelab.models.decoder import DecoderModel
from servelab.models.loader import load_model
from servelab.engine.speculative import (
    ModelDraftProposer, SimpleKVRunner, SpeculativeDecoder,
)

d = tempfile.mkdtemp()
cfg = build_tiny_qwen2(os.path.join(d, "m"))
mc = ModelConfig.from_hf(cfg, path=os.path.join(d, "m"))
_, state = load_model(os.path.join(d, "m"))
PROMPT = list(range(10, 30))

dense = DecoderModel(mc, device="cpu", dtype=torch.float32)
dense.load_state_dict_hf(state)

target = SimpleKVRunner(dense)
draft_runner = SimpleKVRunner(dense)
proposer = ModelDraftProposer(draft_runner, k=4)
spec = SpeculativeDecoder(target, proposer, k=4)

# baseline step by step
base_runner = SimpleKVRunner(dense)
bl = []
logits = base_runner.prefill(PROMPT)[-1]
for _ in range(8):
    tok = int(torch.argmax(logits).item())
    bl.append(tok)
    logits = base_runner.decode_step(tok)

# spec round by round, manual
tokens = []
logits = target.prefill(PROMPT)[-1]
tokens.append(int(torch.argmax(logits).item()))
print("baseline:", bl)
for round_i in range(3):
    drafts = proposer.propose(tokens, 4)
    print(f"round {round_i}: tokens_tail={tokens[-3:]} drafts={drafts}")
    verify_in = [tokens[-1]] + drafts
    rows = target.prefill(verify_in)
    print("  verify argmax per row:",
          [int(torch.argmax(rows[i]).item()) for i in range(rows.shape[0])])
    emitted = spec._verify(tokens, drafts)
    print("  emitted:", emitted)
    tokens.extend(emitted)
print("spec tokens:", tokens)
