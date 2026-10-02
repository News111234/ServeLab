# 面试指南：JD 每一条 ↔ 代码位置 ↔ 讲法

> 用法：面试前按 JD 逐条过。每条给出「代码入口 → 你要能讲的一句话 → 可能的追问」。
> **可背诵的完整问答见 [interview_qa.md](interview_qa.md)**，领域知识打勾清单见
> [knowledge_checklist.md](knowledge_checklist.md)。
> 原则：**每个机制都亲手实现过**，所以从"用过"升级到"能写出 v0 并说出与 v1 的差异"。

## 岗位职责 1：vLLM/SGLang 推理系统研发

- **代码**: `engine/engine.py`（step 循环）、`engine/scheduler.py`
- **讲法**: "我按 vLLM 的调度-执行分离结构写了一个引擎：scheduler 产出
  SchedulerOutput（prefill chunks + decode 集合），engine 只负责打包批执行与采样。
  continuous batching 的本质是 decode 步不需要整批同步结束，每步后重算 batch。"
- **追问**: vLLM v0 和 v1 调度器差别？→ v0 整批 prefill/decode 交替 + block queue，
  v1 混批 + hash 块缓存 + 无 COW；我的实现两种模式都有开关。

## 岗位职责 2：GPU 算子与性能优化（CUDA/Triton）

- **代码**: `attention/triton/paged_attention.py`、`servelab/quant/linear.py`、
  `bench/kernel_bench.py`
- **讲法**: "Triton paged attention 是单 pass 在线 softmax：每程序处理一个
  (seq, head)，按块遍历 block table，维护 running max/sum/acc 三元组；
  GQA 通过 `kv_head = head // groups` 映射。正确性用 torch 参考实现对拍
  （test 里有）。"
- **追问**:
  - 为什么 decode 是访存受限？→ 每步每 token 都要把全部权重和 KV 读一遍，
    arithmetic intensity 低， roofline 上贴着带宽线（`model_cost.py` 里就是这个模型）。
  - Kernel launch 瓶颈怎么定位？→ Nsight Systems 时间线上看 gap；
    CUDA Graph 是解法，我引擎里 `enforce_eager` 开关预留了这个扩展。
  - GEMM 优化？→ W8A8 参考实现里讲了 per-token×per-channel 的数值方案，
    CUDA 上换 `torch._int_mm`；算强度/tiling 影响在 kernel_bench 里量。

## 岗位职责 3：分布式与 MoE

- **代码**: `parallel/`（TP 全套）、`simulator/cluster.py`（PD 分离）、
  `simulator/moe.py`、`simulator/router.py`
- **讲法**: "TP 我按 Megatron 规则实现了完整切分：q/k/v、gate/up 列并行
  （输出维切，零通信），o_proj、down_proj 行并行（输入维切，输出 all-reduce），
  embedding/lm_head 词表并行。注意力头按 rank 分片后**注意力本身零通信**，
  每层只有两次 all-reduce。引擎是 SPMD 的：每个 rank 跑相同的确定性调度器，
  rank 0 采样并广播 token。测试里 TP=2/TP=4 和单卡输出逐 token 一致。"
- **追问**:
  - TP=2 时 GQA 的 KV 头怎么切？→ kv_heads/world，每组 query 头跟自己的
    kv 头留在同一 rank（整除约束在引擎构造时 assert）。
  - 通信怎么验证的？→ 两个后端：线程级 SimulatedCollective（CI 可跑，验证
    切分数学）+ torch.distributed gloo（真进程）；接口同一抽象，上 NCCL 零改动。
  - NCCL AllReduce 为什么对 decode TP 是主要开销？→ 每层两次（attention out
    + MLP out），小消息延迟受限，与计算 overlap 是关键；EP 里 DeepEP 把
    all2all 做成 NVLink/RDMA kernel 化（low-latency 模式）。
  - 为什么输出可能和单卡 bitwise 不同？→ all-reduce 求和顺序/不同 batch 形状
    的 GEMM 分块会改变浮点舍入——语义等价、数值不保证 bitwise（我在测试里
    显式文档化了这一点）。

## 岗位职责 4：KV Cache 与量化

- **代码**: `kv_cache/manager.py`、`kv_cache/radix.py`、`kv_cache/quant.py`、
  `kv_cache/offload.py`、`kv_cache/swap.py`
- **讲法**: "PagedAttention 解决外碎片、prefix cache 解决重复 prefill、
  量化解决单 token KV 体积、offload/swap 解决容量上限。我的前缀树是内容寻址的：
  key = blake2b(父链哈希, 块 token)，所以只要完整块序列一致就能共享物理块；
  运行中的块靠锁计数免驱逐。" + "我只缓存完整块，所以不需要 COW——
  vLLM v1 同款取舍，这是我最喜欢讲的 trade-off。" + "抢占我实现了两种模式：
  recompute（丢 KV 重算，靠前缀缓存降低重算成本）和 swap（私有 KV 拷到 host，
  进度零丢失）——vLLM v0 是 swap，v1 砍掉了 swap 只留 recompute，我两种都
  实现了所以能讲清楚取舍。" + "驱逐我做了归档：换出的前缀块进 host offloader，
  之后前缀未命中会自动恢复——驱逐从'丢失'变成'降级'。"
- **追问**: PagedAttention 相对朴素方案省多少？→ 外碎片从 O(batch×seq) 降到
  单块内 ~半块/序列；prefix cache 在多轮对话 trace 上能省 30%+ prefill
  （`examples/run_simulator.py` 里 cache hit 那列可现场演示）。
  swap 什么时候比 recompute 好？→ 重算成本（prompt 长 × 前缀未命中）vs
  PCIe 拷贝成本（KV 字节 / 带宽）的权衡——长输出短 prompt 偏向 swap。

## 投机解码（JD 未明说但 serving 岗位高频追问）

- **代码**: `engine/speculative.py`
- **讲法**: "完整实现了提议-验证-拒绝采样：draft 提 k 个 token，target 一次
  teacher-force 前向 [last + drafts] 得 k+1 个分布，逐位接受、首错纠正、
  全对拿第 k+1 个免费奖励 token。温度 > 0 走标准拒绝采样（残差分布
  norm(max(p-q,0))，含 p==q 的零残差保护）。**greedy 输出与普通解码逐 token
  严格一致是测试钉死的契约**，与提议器质量无关。指标上我区分接受率和
  target 前向调用摊销——投机的收益来自把 k+1 次前向合成 1 次（GPU 上
  decode 是访存受限，batch 内多 token 几乎不增加时延）。"
- **追问**: 什么工作负载适合 prompt-lookup？→ 重复/抽取式（代码编辑、RAG
  引用、多轮改写）；接受率是提议器质量 × 目标分布 sharpness 的函数。
  为什么 draft 通常要小 1~2 个数量级？→ 每轮 draft 自己要跑 k 步，只有
  draft 单步时延 << target 时才净赚。

## 岗位职责 5：大规模集群与 Benchmark 体系

- **代码**: `simulator/metrics.py`、`bench/benchmark_serving.py`
- **讲法**: "指标口径对齐 vLLM benchmark_serving：TTFT、TPOT（每输出 token
  时延）、goodput = 同时满足 TTFT 与 TPOT SLO 的请求占比——比平均延迟诚实。
  我在模拟器里还加了 Jain 指数量多副本负载均衡。"
- **追问**: KV-aware routing 是什么？→ 按前缀哈希做亲和路由让命中率最大化
  （`PrefixAffinityRouter`），代价是负载不均（Jain 指数会掉，对比表可见）；
  我的 `PredictiveRouter` 就是把两者用预测工作量融合——论文方向 #2。

## 任职要求 1：系统与编程基础

- **代码**: 整个 `kv_cache/` 三层拆分 + 三方互斥不变量 + `tests/`
- **讲法**: "我把'物理池/前缀树/块管理'拆成三层，因为淘汰策略研究需要
  整体换掉其中一层而不动另外两层；62 个测试里最值钱的是引擎四种配置组合
  输出逐 token 一致。"

## 任职要求 2：LLM 推理技术栈核心概念速查

| 概念 | 我的实现位置 | 一句话 |
|---|---|---|
| PagedAttention | `kv_cache/manager.py` + `models/decoder.py::_paged_attention` | KV 分块 + block table 间接寻址 |
| FlashAttention | 精神在 Triton kernel（online softmax） | IO-aware：不物化 S 矩阵 |
| Prefill/Decode | `engine/engine.py::step` | 两阶段，计算/访存受限互换 |
| Continuous Batching | `engine/scheduler.py` | 步级重组 batch |
| Chunked Prefill | `scheduler._plan_prefill_chunk` | 长 prompt 切片，消除队头阻塞 |
| PagedAttention prefix 复用 | `kv_cache/radix.py` | 内容寻址块共享 |
| Preemption | `scheduler._handle_alloc_failure` | recompute 模式 + 前缀缓存加速重算 |

## 演示清单（现场可跑）

```bash
pytest tests -q                                    # 62 passed, 2 skipped
python examples/run_simulator.py --compare-routers # 四种路由对比表
python examples/run_engine.py                      # 随机小模型端到端
python examples/moe_sim.py                         # MoE 重平衡前后
pytest tests/test_parallel.py -q                   # TP=2/4 与单卡逐 token 一致
```
