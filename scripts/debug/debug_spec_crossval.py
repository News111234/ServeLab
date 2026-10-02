"""Cross-validate SimpleKVRunner vs dense paged forward vs LLMEngine."""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from helpers_tiny_model import build_tiny_qwen2
from servelab.config import CacheConfig, ModelConfig, SchedulerConfig
from servelab.engine.engine import LLMEngine
from servelab.engine.sampling_params import SamplingParams
from servelab.engine.speculative import SimpleKVRunner
from servelab.kv_cache.manager import BlockManager
from servelab.kv_cache.pool import KVCachePool
from servelab.models.decoder import DecoderModel, SeqMeta
from servelab.models.loader import load_model

d = tempfile.mkdtemp()
cfg = build_tiny_qwen2(os.path.join(d, "m"))
mc = ModelConfig.from_hf(cfg, path=os.path.join(d, "m"))
_, state = load_model(os.path.join(d, "m"))
PROMPT = list(range(10, 30))

# --- dense paged forward
dense = DecoderModel(mc, device="cpu", dtype=torch.float32)
dense.load_state_dict_hf(state)
dense.block_size = 16
pool = KVCachePool(8, 16, mc.num_layers, mc.num_kv_heads, mc.head_dim,
                   dtype=torch.float32, device="cpu", kv_cache_dtype="none")
bm = BlockManager(8, 16, enable_prefix_caching=False)
bm.seq_blocks["s"] = []
bm.seq_matched_nodes["s"] = []
bm.allocate_slots("s", len(PROMPT))
metas = [SeqMeta(0, len(PROMPT), len(PROMPT), bm.get_block_table("s"))]
slots = [bm.slot_for_token("s", i) for i in range(len(PROMPT))]
with torch.no_grad():
    h = dense.forward_packed(torch.tensor(PROMPT), torch.arange(len(PROMPT)),
                             torch.tensor(slots), metas, pool)
    lg_paged = dense.lm_head_forward(h[-1:]).float()

# --- SimpleKVRunner
runner = SimpleKVRunner(dense)
lg_run = runner.prefill(PROMPT)[-1]
print("runner-vs-dense prefill max diff:",
      (lg_paged - lg_run).abs().max().item())

# --- engine prefill logits (one step, captured manually)
eng = LLMEngine(mc, CacheConfig(num_blocks=32),
                SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=64),
                seed=0, device="cpu")
eng.add_request(prompt_token_ids=PROMPT,
                sampling_params=SamplingParams(max_tokens=1, temperature=0.0,
                                               ignore_eos=True))
so = eng.scheduler.schedule()
tokens, positions, slots2, metas2 = [], [], [], []
for ps in so.prefills:
    s, start, n = ps.seq, ps.num_computed, ps.num_new_tokens
    tokens.extend(s.token_ids[start:start + n])
    positions.extend(range(start, start + n))
    slots2.extend(eng.bm.slot_mapping(s.request_id, start, n))
    metas2.append(SeqMeta(0, n, start + n,
                          eng.bm.get_block_table(s.request_id)))
with torch.no_grad():
    h2 = eng.model.forward_packed(torch.tensor(tokens), torch.tensor(positions),
                                  torch.tensor(slots2), metas2, eng.pool)
    lg_eng = eng.model.lm_head_forward(h2[-1:]).float()
print("engine-vs-dense prefill max diff:",
      (lg_eng - lg_paged).abs().max().item())

# --- full generation: plain greedy (runner) vs engine greedy
runner2 = SimpleKVRunner(dense)
out = []
logits = runner2.prefill(PROMPT)[-1]
for _ in range(8):
    tok = int(torch.argmax(logits).item())
    out.append(tok)
    logits = runner2.decode_step(tok)[-1]
eng2 = LLMEngine(mc, CacheConfig(num_blocks=32),
                 SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=64),
                 seed=0, device="cpu")
ref = eng2.generate([PROMPT], SamplingParams.greedy(max_tokens=8))
print("runner greedy:", out)
print("engine greedy:", ref[0].outputs[0].token_ids)
print("greedy match:", out == ref[0].outputs[0].token_ids)
