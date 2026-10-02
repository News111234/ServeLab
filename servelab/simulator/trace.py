"""Request traces for the cluster simulator (torch-free).

Formats supported:
    - synthetic       : Poisson arrivals + lognormal lengths + Zipf prefixes
    - sharegpt        : multi-turn conversations (prefix reuse per conversation)
    - azure           : Azure LLM Inference Trace (csv/parquet, TIMESTAMP /
                        ContextTokens / GeneratedTokens)
    - burstgpt        : BurstGPT csv log

Only lengths and arrival times are needed by the simulator; token ids are
synthesized deterministically from the prefix key so the REAL RadixCache
implementation can be reused for cache-hit accounting (same code as engine).
"""

import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TraceRequest:
    req_id: str
    arrival_time: float
    prompt_tokens: int
    output_tokens: int
    prefix_key: str = ""          # requests sharing a key share its prefix
    prefix_tokens: int = 0        # length of the shared prefix
    priority: int = 0

    def pseudo_tokens(self) -> List[int]:
        """Deterministic token ids for prefix-cache accounting."""
        rng = random.Random(hashlib.md5(self.prefix_key.encode()).hexdigest())
        prefix = [rng.randrange(1000, 50000) for _ in range(self.prefix_tokens)]
        rng2 = random.Random(f"{self.prefix_key}:{self.req_id}")
        suffix = [rng2.randrange(1000, 50000)
                  for _ in range(self.prompt_tokens - self.prefix_tokens)]
        return prefix + suffix


def make_synthetic_trace(
    num_requests: int = 500,
    request_rate: float = 4.0,
    prompt_mean: float = 256.0,
    prompt_cv: float = 0.8,
    output_mean: float = 128.0,
    output_cv: float = 1.0,
    num_prefix_groups: int = 16,
    prefix_len: int = 128,
    zipf_alpha: float = 1.2,
    seed: int = 42,
) -> List[TraceRequest]:
    """Poisson arrivals, lognormal lengths, Zipf-distributed shared prefixes."""
    rng = random.Random(seed)
    zipf = ZipfSampler(num_prefix_groups, zipf_alpha, rng)
    trace = []
    t = 0.0
    for i in range(num_requests):
        t += rng.expovariate(request_rate)
        p = int(round(rng.lognormvariate(math.log(prompt_mean), prompt_cv)))
        o = int(round(rng.lognormvariate(math.log(output_mean), output_cv)))
        g = zipf.sample()
        trace.append(TraceRequest(
            req_id=f"syn-{i}",
            arrival_time=t,
            prompt_tokens=max(8, p),
            output_tokens=max(1, o),
            prefix_key=f"group-{g}" if prefix_len > 0 else "",
            prefix_tokens=prefix_len if prefix_len > 0 else 0,
        ))
    return trace


class ZipfSampler:
    def __init__(self, n: int, alpha: float, rng: random.Random):
        weights = [1.0 / (i + 1) ** alpha for i in range(n)]
        total = sum(weights)
        self.cum = []
        acc = 0.0
        for w in weights:
            acc += w / total
            self.cum.append(acc)
        self.rng = rng

    def sample(self) -> int:
        u = self.rng.random()
        for i, c in enumerate(self.cum):
            if u <= c:
                return i
        return len(self.cum) - 1


def load_sharegpt(path: str, max_requests: int = 2000,
                  words_per_token: float = 0.75) -> List[TraceRequest]:
    """Approximate token counts from word counts (no tokenizer dependency).
    Multi-turn conversations become requests sharing a growing prefix."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    trace: List[TraceRequest] = []
    t = 0.0
    for ci, conv in enumerate(data[: max_requests]):
        turns = conv.get("conversations", [])
        history = 0
        for ti in range(0, len(turns) - 1, 2):
            human, gpt = turns[ti], turns[ti + 1]
            prompt_len = max(8, int(len(human["value"].split()) / words_per_token))
            out_len = max(1, int(len(gpt["value"].split()) / words_per_token))
            t += 2.0  # fixed 2s spacing; replace with real timestamps if needed
            trace.append(TraceRequest(
                req_id=f"sharegpt-{ci}-{ti}",
                arrival_time=t,
                prompt_tokens=prompt_len,
                output_tokens=out_len,
                prefix_key=f"conv-{ci}",
                prefix_tokens=history,
            ))
            history += prompt_len + out_len
    return trace


def load_azure_trace(path: str, max_requests: int = 5000) -> List[TraceRequest]:
    """Azure LLM Inference Trace: TIMESTAMP, ContextTokens, GeneratedTokens.
    Accepts .csv or .parquet (parquet needs pandas+pyarrow)."""
    rows = []
    if path.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(path)
        rows = list(zip(df["TIMESTAMP"], df["ContextTokens"], df["GeneratedTokens"]))
    else:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append((r["TIMESTAMP"], int(float(r["ContextTokens"])),
                             int(float(r["GeneratedTokens"]))))
    import datetime
    base = None
    trace = []
    for i, (ts, ctx, gen) in enumerate(rows[:max_requests]):
        if isinstance(ts, str):
            t0 = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
            ts = t0.timestamp()
        base = ts if base is None else base
        trace.append(TraceRequest(
            req_id=f"azure-{i}", arrival_time=ts - base,
            prompt_tokens=int(ctx), output_tokens=max(1, int(gen)),
        ))
    return trace


def load_burstgpt_trace(path: str, max_requests: int = 5000) -> List[TraceRequest]:
    """BurstGPT log csv: Request timestamp, Input tokens, Output tokens."""
    trace = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            if i >= max_requests:
                break
            try:
                ts = float(r.get("Request timestamp", i))
            except (TypeError, ValueError):
                ts = float(i)
            trace.append(TraceRequest(
                req_id=f"burstgpt-{i}", arrival_time=ts,
                prompt_tokens=int(float(r.get("Input tokens", 0) or 0)) or 8,
                output_tokens=int(float(r.get("Output tokens", 0) or 0)) or 1,
            ))
    return trace


def load_trace(path_or_kind: str, **kwargs) -> List[TraceRequest]:
    if path_or_kind == "synthetic":
        return make_synthetic_trace(**kwargs)
    if path_or_kind.endswith(".json"):
        return load_sharegpt(path_or_kind, **kwargs)
    if "azure" in path_or_kind.lower():
        return load_azure_trace(path_or_kind, **kwargs)
    return load_burstgpt_trace(path_or_kind, **kwargs)
