"""Minimal OpenAI-compatible server (non-streaming) over ServeLab engine.

    python -m servelab.serving.openai_server --model /path/to/Qwen2.5-0.5B-Instruct
    curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
        -d '{"model":"servelab","messages":[{"role":"user","content":"hi"}],"max_tokens":32}'

Streaming (SSE) is deliberately left as an exercise/roadmap item -- the
engine's step() loop already yields per-token deltas to build it on.
"""

import argparse
import uuid
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from ..config import CacheConfig, SchedulerConfig
from ..engine.engine import LLMEngine
from ..engine.sampling_params import SamplingParams
from ..models.loader import load_model


class Message(BaseModel):
    role: str = "user"
    content: str = ""


class ChatRequest(BaseModel):
    model: str = "servelab"
    messages: List[Message]
    max_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 1.0
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str = "servelab"
    prompt: str
    max_tokens: int = 64
    temperature: float = 0.7


def create_app(model_path: str, kv_dtype: str = "auto",
               enable_prefix_cache: bool = True) -> FastAPI:
    app = FastAPI(title="ServeLab")
    mc, _ = load_model(model_path)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    engine = LLMEngine(
        mc,
        CacheConfig(enable_prefix_caching=enable_prefix_cache,
                    kv_cache_dtype=kv_dtype),
        SchedulerConfig(max_num_seqs=32, max_num_batched_tokens=4096),
        tokenizer=tok,
    )

    def _generate(prompt: str, max_tokens: int, temperature: float, top_p: float):
        rid = f"req-{uuid.uuid4().hex[:8]}"
        engine.add_request(prompt=prompt,
                           sampling_params=SamplingParams(
                               max_tokens=max_tokens, temperature=temperature,
                               top_p=top_p), request_id=rid)
        out = None
        while engine.has_unfinished():
            for o in engine.step():
                if o.request_id == rid:
                    out = o
        return out

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        prompt = tok.apply_chat_template([m.model_dump() for m in req.messages],
                                         tokenize=False, add_generation_prompt=True)
        out = _generate(prompt, req.max_tokens, req.temperature, req.top_p)
        comp = out.outputs[0]
        return {
            "id": out.request_id, "object": "chat.completion", "model": req.model,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": comp.text},
                         "finish_reason": "stop" if comp.finish_reason == "eos" else "length"}],
            "usage": {"prompt_tokens": len(out.prompt_token_ids),
                      "completion_tokens": len(comp.token_ids)},
        }

    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        out = _generate(req.prompt, req.max_tokens, req.temperature, 1.0)
        comp = out.outputs[0]
        return {"id": out.request_id, "object": "text_completion",
                "model": req.model,
                "choices": [{"index": 0, "text": comp.text,
                             "finish_reason": "stop" if comp.finish_reason == "eos" else "length"}]}

    @app.get("/stats")
    def stats():
        return engine.stats()

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--kv-cache-dtype", default="auto")
    args = ap.parse_args()
    import uvicorn
    app = create_app(args.model, kv_cache_dtype=args.kv_cache_dtype)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
