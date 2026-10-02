# ServeLab

**研究型轻量 LLM 推理服务实验平台** —— 一个可以读完源码、改出论文、写上简历的项目。

ServeLab 用 ~4000 行可读代码复刻了 vLLM / SGLang / LMCache 的核心机制，并在其上构建了一个
**trace 驱动的集群模拟器**，专门用于 serving 算法（调度 / 路由 / 缓存淘汰 / 量化）研究。
所有策略点都是**可插拔接口**，每个接口都对应一个可发表的研究方向（见
[docs/research_roadmap.md](docs/research_roadmap.md)）。

```
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 3  研究: trace 驱动集群模拟器 (torch-free, 秒级迭代)            │
│     router / predictor / PD 分离 / MoE EP / 指标 / 校准器 / sweep     │
├──────────────────────────────────────────────────────────────────────┤
│  Layer 2  引擎: LLMEngine (torch, 参考实现)                           │
│     continuous batching · chunked prefill · 抢占(recompute+SWAP)      │
│     投机解码(greedy+拒绝采样) · 张量并行(SPMD, 模拟/gloo 双后端)       │
│     Qwen2/Llama decoder · paged attention (torch + Triton)            │
├──────────────────────────────────────────────────────────────────────┤
│  Layer 1  核心 (torch-free, 100% 单测覆盖):                           │
│     Paged BlockManager · Radix PrefixCache · 淘汰/准入策略            │
│     KV 量化(FP8/INT8) · KV Offload(驱逐归档+前缀恢复) · W8A8 Linear   │
└──────────────────────────────────────────────────────────────────────┘
```

## 为什么是这个形态

| 你的目标 | ServeLab 怎么支撑 |
|---|---|
| 面试证明"读过框架源码" | 每个核心机制都有**注释了与 vLLM/SGLang 异同**的迷你实现，可白板讲解 |
| 面试证明"做过优化" | kernel bench（Triton paged attention vs torch）、serving bench（TTFT/TPOT/goodput）、profiling 指引 |
| 后续发论文 | 模拟器 + 可插拔策略 + 真实 trace 加载器；策略代码可在模拟器与真实引擎间无缝迁移 |

## 快速开始

```bash
cd ServeLab
# 依赖: numpy + tabulate (模拟器); torch + safetensors (引擎); 建议直接
pip install -e ".[all]"
pytest tests -q          # 62 个测试，其中模拟器/调度器/缓存全部 torch-free
```

```bash
# 1) 引擎端到端（无需下载模型: 自动构建随机小模型）
python examples/run_engine.py

# 2) 真实 checkpoint（如 Qwen2.5-0.5B-Instruct，注意 torch 需 CUDA 版）
python examples/run_engine.py --model D:/models/Qwen2.5-0.5B-Instruct --prompt "你好"

# 3) 模拟器: 对比四种路由策略（论文主战场）
python examples/run_simulator.py --compare-routers --num-requests 300

# 4) PD 分离仿真
python examples/run_simulator.py --pd --replicas 4

# 5) MoE 专家负载均衡仿真
python examples/moe_sim.py --gpus 8

# 6) 张量并行: 线程级模拟 TP（单进程, CI 可跑）
#    （测试: pytest tests/test_parallel.py -q  TP=2/4 与单卡输出逐 token 一致）
#    真多进程 gloo（GPU 机器上可换 nccl）:
python examples/run_tp_gloo.py --tp 2

# 7) OpenAI 兼容 server
python -m servelab.serving.openai_server --model /path/to/model
```

## 与开源项目的对应关系

| ServeLab 模块 | 参考的开源实现 | 本文的取舍 |
|---|---|---|
| `kv_cache/manager.py` | vLLM v0 `BlockSpaceManager` / v1 `KVCacheManager` | 去掉 COW：只缓存完整块（v1 同款取舍）；SWAP 抢占 + 驱逐归档/恢复 |
| `kv_cache/radix.py` | SGLang `RadixCache` | 单块单节点（免分裂），锁计数防驱逐 |
| `kv_cache/policies.py` | SGLang LRU / 各类缓存论文 | **可插拔**：LRU / LFU / cost-aware，论文方向 #1 |
| `engine/scheduler.py` | vLLM v0/v1 scheduler, Sarathi-Serve | continuous batching + chunked prefill + recompute/swap 抢占，策略可插拔 |
| `engine/speculative.py` | vLLM spec-decode / Medusa 论文 | 提议-验证-拒绝采样完整算法，greedy 输出与普通解码严格一致 |
| `parallel/` | Megatron-LM / vLLM TP | 列/行并行 + 词表并行，SPMD 引擎，模拟(线程)与 gloo(进程)双后端 |
| `models/decoder.py` | vLLM `Qwen2` 实现, HF transformers | 打包批 + 每序列 paged attention（正确性优先） |
| `attention/triton/` | vLLM v0 paged attention kernel | 单 pass 在线 softmax；split-K/flash-decoding 是练习题 |
| `kv_cache/quant.py` | KIVI / KVQuant / vLLM fp8-KV | per-(row,head) 对称量化基线，粒度是研究轴 |
| `kv_cache/offload.py` | LMCache / AttentionStore | 驱逐归档 + 前缀未命中自动恢复 |
| `simulator/` | DistServe / Mooncake / Preble 的实验方法论 | 解析式 step-time 模型 + 真实 trace + 参数校准器 + sweep 流水线 |
| `bench/benchmark_serving.py` | vLLM `benchmark_serving.py` | TTFT/TPOT/goodput 指标口径一致 |

致谢与许可：本项目从上述项目的公开设计与论文中学习，代码为独立实现。MIT License。

## 两条使用路径

**面试路径**（1~2 周）：先读 [docs/interview_guide.md](docs/interview_guide.md)，题目逐条过
[docs/interview_qa.md](docs/interview_qa.md)（预设问题+背诵答案）与
[docs/knowledge_checklist.md](docs/knowledge_checklist.md)（领域知识打勾清单）——
JD 每一条都映射到具体文件与"怎么讲"；跑通 62 个测试；用自己的话复述
PagedAttention / chunked prefill / radix cache 三个机制。

**论文路径**（1~3 个月）：先读 [docs/research_roadmap.md](docs/research_roadmap.md) ——
6 个选题（含 baseline、数据集、指标、风险），选一个在模拟器上把策略改掉，
跑出对比表后移植回真实引擎验证。

## 目录

```
servelab/
├── kv_cache/     manager(块管理+swap) radix(前缀树) policies(淘汰) pool(物理池)
│                 quant(FP8/INT8 KV) offload(驱逐归档+恢复) swap(CPU换出)
├── engine/       scheduler(调度+策略) sequence sampling_params sampler engine
│                 speculative(投机解码: 提议/验证/拒绝采样)
├── parallel/     layers(列/行/词表并行) tp_model sharding 模拟执行 gloo 后端
├── models/       loader(HF权重) decoder(Qwen2/Llama)
├── attention/    triton/paged_attention (Triton 算子)
├── quant/        linear(W8A8 参考实现)
├── simulator/    trace(合成/ShareGPT/Azure/BurstGPT) predictor router
│                 cluster(多副本+PD) model_cost(成本模型) metrics moe
│                 calibration(参数校准) sweep(实验流水线)
├── bench/        benchmark_serving datasets kernel_bench
└── serving/      openai_server
tests/            62 tests: 块管理/前缀树/调度/模拟器/量化/TP/投机/swap/校准
docs/             architecture · research_roadmap · interview_guide · interview_qa ·
                  knowledge_checklist · benchmark_guide · TODO · PROGRESS
```
