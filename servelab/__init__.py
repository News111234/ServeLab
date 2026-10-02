"""ServeLab: a research-oriented miniature LLM serving stack.

Layer 1 (torch-free core, fully unit-testable):
    - kv_cache   : paged block pool + radix prefix cache + pluggable eviction policies
    - engine.scheduler : continuous batching / chunked prefill / preemption (pluggable policies)
    - simulator  : trace-driven multi-replica cluster simulator (routing, PD-disagg, MoE)

Layer 2 (torch engine, reference implementation):
    - models     : Qwen2 / LLaMA decoder with paged attention paths
    - attention  : correctness-first torch paged attention + optional Triton kernels
    - quant      : FP8/INT8 KV cache quantization, W8A8 linear (reference)

The design deliberately mirrors vLLM v1 (scheduler/KV manager), SGLang (RadixCache)
and LMCache (offloading) so that policy research done here can be ported back.
"""

__version__ = "0.1.0"
