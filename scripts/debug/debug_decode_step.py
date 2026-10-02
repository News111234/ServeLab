"""Isolate the decode-step bug: incremental runner vs full recompute."""
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
from servelab.engine.speculative import SimpleKVRunner

d = tempfile.mkdtemp()
cfg = build_tiny_qwen2(os.path.join(d, "m"))
mc = ModelConfig.from_hf(cfg, path=os.path.join(d, "m"))
_, state = load_model(os.path.join(d, "m"))
PROMPT = list(range(10, 30))

dense = DecoderModel(mc, device="cpu", dtype=torch.float32)
dense.load_state_dict_hf(state)

runner = SimpleKVRunner(dense)
with torch.no_grad():
    lg_prompt = runner.prefill(PROMPT)[-1]
    tok1 = int(torch.argmax(lg_prompt).item())
    print("tok1 =", tok1)

    # incremental decode
    lg_inc = runner.decode_step(tok1)

    # full recompute over prompt + tok1 (fresh runner)
    full = SimpleKVRunner(dense)
    lg_full = full.prefill(PROMPT + [tok1])[-1]

    print("incremental vs full recompute max diff:",
          (lg_inc - lg_full).abs().max().item())
    print("incremental top5:", torch.topk(lg_inc, 5).indices.tolist())
    print("full       top5:", torch.topk(lg_full, 5).indices.tolist())
    print("nan in incremental:", bool(torch.isnan(lg_inc).any()))
