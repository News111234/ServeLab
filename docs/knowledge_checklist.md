# LLM 推理 Infra 知识清单（面试自测版）

> 用法：每条都是"考点一句话 + ServeLab 对应位置"。能**脱稿把一句话展开讲 30 秒**
> 就打勾；讲不出来就回 `docs/architecture.md` 和对应源码复习。
> ⭐ = 出现频率最高，优先背。配 `docs/interview_qa.md`（针对本项目的问题+答案）。

## 1. Transformer 与解码基础

- [ ] ⭐ 自回归解码：每步只新增 1 token，之前所有 token 的 K/V 必须保留 → KV Cache 的由来
- [ ] ⭐ QKV/O 四个投影的参数形状与计算量（2×P×n FLOPs 近似的来源）
- [ ] ⭐ GQA/MQA：多 query 头共享 KV 头，KV Cache 体积缩小 (H_q/H_kv) 倍（LLaMA-3-8B: 32/8=4×）
- [ ] RoPE：旋转位置编码，作用于 Q/K 不作用于 V；NeoX rotate-half 实现形式
      （ServeLab: `models/decoder.py::apply_rope`）
- [ ] RMSNorm 与 LayerNorm 区别：无均值中心化、无 bias，推理期只有缩放
      （ServeLab: `models/decoder.py::RMSNorm`，float32 计算防溢出）
- [ ] SwiGLU MLP：gate/up/down 三矩阵（Silu(gate(x)) * up(x) → down）
- [ ] Word embedding 与 lm_head 权重 tying：省一份 V×H 参数
      （ServeLab: `DecoderModel.load_state_dict_hf` 的 tie 分支）

## 2. 推理两阶段与瓶颈 ⭐⭐

- [ ] ⭐ Prefill：整段 prompt 一次前向，算术强度 ≈ n tokens → **compute-bound**
- [ ] ⭐ Decode：每步 1 token，但要读全部权重+全部 KV → **memory-bound**
      （数字：8B 模型 fp16 权重 16GB / A100 有效 1.4TB/s ≈ 11ms/步，与 batch 关系不大）
- [ ] ⭐ Roofline 模型：t = max(FLOPs/F_eff, Bytes/BW_eff)；机器平衡点 = F_eff/B_eff
      （ServeLab: `simulator/model_cost.py::step_time` 就是这个模型，可校准）
- [ ] Batch 增大时：prefill 时间近似线性涨（compute），decode 时间几乎不变（memory）
      → 这就是 batching 对 decode 吞吐的杠杆
- [ ] TTFT 由 prefill 决定；TPOT 由 decode 决定；两者优化手段冲突 → PD 分离的动机
- [ ] FP16/BF16/FP8 的位宽与表示范围（FP8 E4M3: 4 位指数、max 448）

## 3. KV Cache 与 PagedAttention ⭐⭐⭐

- [ ] ⭐ KV Cache 体积公式：2(K+V) × layers × kv_heads × head_dim × bytes/token
      （7B: 2×32×32×128×2B = 512KB/token → 4096 tokens ≈ 2GB）
- [ ] ⭐ PagedAttention：KV 按 block_size 分块 + block table 间接寻址，
      消除按 seq_len 预分配的**外碎片**，内部碎片降到 ~半块/序列
      （ServeLab: `kv_cache/manager.py`，slot = block_id×bs+offset）
- [ ] ⭐ vLLM v0 vs v1 的块管理：v0 free_block_queue + 内容哈希 + **copy-on-write**；
      v1 简化为整块哈希匹配、无 COW
      （ServeLab 采用 v1 取舍——`PROGRESS.md` 设计决策表）
- [ ] 只缓存完整块 ⇒ 共享块不可变 ⇒ 不需要 COW；代价：部分块内容不共享
- [ ] COW 什么时候才需要：子块（<block_size）粒度共享时，v0 的 fork/beam search 场景
- [ ] KV Cache 显存预算：num_blocks = 可用显存 / 每块字节数
      （ServeLab: `engine.py::_default_num_blocks`，量化时按 1 字节+scale 算）
- [ ] KV 量化（见 §7 量化节）与 KV 稀疏化/丢弃（H2O、SnapKV——了解即可）

## 4. Prefix Caching / 前缀复用 ⭐⭐

- [ ] ⭐ 动机：多轮对话、few-shot、长 system prompt 的前缀 KV 完全相同，重复 prefill 浪费
- [ ] ⭐ 块内容哈希 = blake2b(父链哈希, 块 token ids)：前缀一致 ⇒ 哈希链一致 ⇒ 物理块共享
      （ServeLab: `utils/common.py::hash_block_tokens`）
- [ ] ⭐ RadixCache（SGLang RadixAttention）：树节点=块，LRU 驱逐叶子，**锁计数**保护运行中块
      （ServeLab: `kv_cache/radix.py`，单块单节点免分裂）
- [ ] vLLM v1 的 prefix caching：哈希表版（无树），默认开启
- [ ] 命中收益：TTFT 省 prefill；代价：树维护 + 碎片（块不连续）
- [ ] 淘汰策略研究点：LRU 不看"重算成本×复用概率" → workload-aware 淘汰/准入
      （论文方向 #1，ServeLab: `kv_cache/policies.py` 可插拔）
- [ ] 层级存储：放不下的进 host/SSD（LMCache/AttentionStore/Mooncake）；
      ServeLab 做了驱逐归档+未命中自动恢复（`kv_cache/offload.py` + manager.restore）

## 5. 调度 ⭐⭐

- [ ] ⭐ Continuous batching（Orca, OSDI'22）：**iteration 级**调度，decode 步不整批同步，
      每步重组 batch → 吞吐大幅提升（ServeLab: `engine/scheduler.py::schedule` 三阶段）
- [ ] ⭐ Chunked prefill（Sarathi-Serve/vLLM v1）：长 prompt 切片与 decode 混批，
      消除"一个大 prefill 阻塞全队"的队头阻塞，TPOT 尾部显著变好
- [ ] ⭐ Preemption：内存不够时驱逐运行中序列——recompute（丢 KV 重算，前缀缓存兜底）
      vs swap（KV 拷 host，进度零丢失）；v0 有 swap，v1 只留 recompute
      （ServeLab 两种都实现：`kv_cache/swap.py` + scheduler 两个分支）
- [ ] Admission：waiting 按策略排序（FCFS/priority/SJF-predicted），
      内存不足 head-of-line blocking（ServeLab: `SchedulingPolicy` 可插拔）
- [ ] Scheduling 改进的研究点：输出长度预测驱动 SJF、优先级+公平性、
      locality-aware（论文方向 #2，ServeLab: `simulator/predictor.py`）

## 6. 投机解码 ⭐⭐

- [ ] ⭐ 流程：draft 提 k 个 token → target 一次 teacher-force 前向 [last+drafts] 得 k+1 个
      分布 → 逐位接受、首错纠正、全对拿第 k+1 个免费 token
      （ServeLab: `engine/speculative.py::SpeculativeDecoder._verify`）
- [ ] ⭐ 拒绝采样保证输出分布 = target 分布（Leviathan/Chen 2023）；
      greedy 退化为 argmax 逐位比对——**greedy 输出与普通解码严格一致**
- [ ] 收益来源：GPU 上 decode 是访存受限，一次前向 5 个 token ≈ 一次前向 1 个 token 的时延
      → 前向调用次数减少 ≈ 加速
- [ ] 何时亏：draft 太大/接受率低（draft 自己跑 k 步的成本 > 省下的 target 步）
      经验：draft 比 target 小 1~2 个数量级，接受率 > ~60-70% 才净赚
- [ ] Prompt lookup / n-gram 提议：无 draft 模型，从 context 查重复片段，
      适合代码编辑/RAG/改写类负载（greedy-only，vLLM 同限制）
- [ ] 与 continuous batching 融合是工程难点（vLLM V1 spec-decode、Medusa/EAGLE 多头提议
      ——ServeLab 未做，作为 roadmap 条目）

## 7. 量化 ⭐⭐

- [ ] ⭐ KV Cache 量化：FP8(E4M3) per-tensor（vLLM）vs INT8 per-channel K + per-token V
      （KIVI）；K 的异常值集中在个别通道 → per-channel 更优，需要转置布局
      （ServeLab 基线: per-(row,head) 对称量化，`kv_cache/quant.py`，粒度是研究轴）
- [ ] FP8 vs INT8：FP8 动态范围大、对 outlier 更鲁棒；硬件上 Hopper/Blackwell 有原生
      FP8 指令
- [ ] ⭐ W8A8（权重+激活都 INT8）：per-token 激活 × per-channel 权重，
      SmoothQuant 把激活异常值迁移到权重（ServeLab 参考实现: `quant/linear.py`，
      CUDA 上换 torch._int_mm/CUTLASS）
- [ ] ⭐ Weight-only 量化：**AWQ**（按激活分布保护显著权重通道）、**GPTQ**（基于 OBQ 的
      逐列量化 + 误差补偿，常配 group-wise），只压权重、激活保持 fp16
- [ ] 量化权衡：显存/带宽收益 vs 精度损失（PPL 上涨、长上下文退化）；
      KV 量化能直接增加可容纳的并发/上下文长度

## 8. 并行策略 ⭐⭐⭐

- [ ] ⭐ TP 张量并行：列并行（权重按输出维切，无通信）+ 行并行（按输入维切，输出
      all-reduce）；Megatron 切分规则
      （ServeLab: `parallel/layers.py`，与 Megatron 逐条对应）
- [ ] ⭐ TP 通信量：每层 2 次 all-reduce（attn out + MLP out）；ring all-reduce 通信量
      2(K-1)/K × 消息大小；decode 时消息小 → **延迟受限**，与计算 overlap 是关键
- [ ] ⭐ TP 下注意力：heads/kv_heads 按 rank 切 → 注意力本身**零通信**
      （GQA 整除约束：H_kv % TP == 0，ServeLab 构造时 assert）
- [ ] ⭐ 词表并行：embedding mask 出本 rank 分片 + all-reduce；lm_head 分片算 logits +
      all-gather（ServeLab: `VocabParallelEmbedding/VocabParallelLMHead`）
- [ ] PP 流水线并行：按层切，micro-batch 填 bubble；bubble 比例 (p-1)/(m+p-1)；
      1F1B/interleaved 调度
- [ ] EP 专家并行（见 §10 MoE）；DP 数据并行（推理期主要是多副本）
- [ ] CP/SP 上下文/序列并行：长上下文把 sequence 维切开（Ring Attention、
      DeepSpeed Ulysses）——了解概念即可
- [ ] 多维组合：TP(节点内 NVLink) × PP(节点间) × DP(副本) —— vLLM/SGLang 的常见部署形态
- [ ] SPMD vs driver/worker：ServeLab 是 SPMD（每 rank 跑相同确定性调度器，
      rank0 采样广播）；vLLM 是 driver + workers 分工
      （`parallel/engine_group.py` + rank0 广播）

## 9. 集合通信与网络

- [ ] ⭐ All-Reduce（TP 用）、All-Gather（词表/logits）、All-to-All（MoE/EP 用）的语义与
      通信量公式
- [ ] ⭐ NCCL：GPU 集合通信库，ring/tree 算法，带宽随消息大小变化（小消息 latency-bound）
- [ ] NVLink/NVSwitch：节点内高带宽（H100 NVLink 900GB/s 双向），TP 限制在节点内的原因
- [ ] RDMA/InfiniBand/RoCE：跨节点 KV 传输（Mooncake 用 RDMA 传 KV）
- [ ] 通信与计算 overlap：异步 all-reduce、分块流水（TP 优化的核心手段）
- [ ] DeepEP：MoE 专用的 all-to-all 通信库（normal + low-latency 两模式）

## 10. MoE Serving ⭐

- [ ] MoE 前向：router 选 top-k 专家 → 只激活部分 FFN → FLOPs 省但**显存要装全部专家**
- [ ] 专家负载不均衡（Zipf 分布）→ capacity factor 限流丢 token 或 aux-loss 重平衡；
      DeepSeek-V3 的 aux-loss-free bias 更新
- [ ] EP + All-to-All：dispatch（token 送到专家所在卡）→ 分组计算 → combine
- [ ] Grouped GEMM：按专家分组的批量 GEMM，MoE 计算的核心 kernel
- [ ] Serving 侧研究点：专家放置/重平衡、动态容量、预测驱动的预换入
      （ServeLab: `simulator/moe.py` 模拟 + greedy 重平衡基线）

## 11. GPU 架构与算子

- [ ] ⭐ SIMT/SMEM/寄存器/占用率（occupancy）基本概念；warp 调度
- [ ] ⭐ Triton 编程模型：block 级抽象、tl.load/store、constexpr、JIT 编译
      （ServeLab: `attention/triton/paged_attention.py` 单 pass 在线 softmax decode kernel）
- [ ] ⭐ FlashAttention：online softmax（分块维护 running max/sum）、不物化 S 矩阵、
      IO-aware tiling；FA2 把并行维度扩到 seq
      （ServeLab: torch 路径 + Triton kernel 都用同一 online softmax 数学）
- [ ] CUDA Graph：捕获整段 kernel 序列一次发射，消除 decode 的 launch 开销
      （ServeLab: `enforce_eager` 开关预留，roadmap）
- [ ] Kernel 融合：RMSNorm+RoPE+QKV、bias+act 等；为什么 decode 偏爱大融合（launch 占比高）
- [ ] Blackwell/Hopper 新特性：FP8/FP4 原生、TMA（了解即可）

## 12. Profiling 工具 ⭐

- [ ] ⭐ Nsight Systems：时间线，看 kernel 间 gap、launch 开销、通信重叠
- [ ] ⭐ Nsight Compute：单 kernel 的 SOL 指标——SM% 高=计算受限，DRAM% 高=访存受限
- [ ] torch.profiler：CPU 侧快速定位（ServeLab: `docs/benchmark_guide.md` §3 有模板）
- [ ] 排查流程：吞吐上不去 → nsys 找 gap（launch？通信？）→ ncu 看受限类型 → 针对性
      融合/图捕获/overlap

## 13. Serving 指标与 SLO ⭐⭐

- [ ] ⭐ TTFT：首 token 时延（≈ 排队 + prefill）；TPOT：每输出 token 时延（decode 平均，
      不含首 token）；端到端 = TTFT + TPOT×(n-1)
- [ ] ⭐ Goodput：**同时满足** TTFT 与 TPOT SLO 的请求占比（DistServe 提出的口径，
      比平均延迟诚实）（ServeLab: `simulator/metrics.py` + bench 同口径）
- [ ] 吞吐 vs 时延的 trade-off 曲线；负载过载点行为（队列爆炸、SLO 崩）
- [ ] 负载均衡度量：Jain 公平指数 = (Σx)²/(n·Σx²)，越接近 1 越均衡
      （路由对比表里那列）
- [ ] KV-aware routing：前缀亲和路由（命中率↑、均衡↓）与预测性路由的融合
      （论文方向 #2/#4，ServeLab: `simulator/router.py` 四种实现）

## 14. 开源框架与论文生态 ⭐⭐

- [ ] ⭐ vLLM：PagedAttention 论文（SOSP'23）；v0 引擎 vs v1 引擎（调度/缓存重构、无 COW、
      CUDA Graph、prefix caching 默认开）
- [ ] ⭐ SGLang：RadixAttention（前缀树缓存）、cache-aware 路由、零开销调度器（overlap）
- [ ] TensorRT-LLM：编译式 engine、kernel 高度融合、inflight batching
- [ ] Orca（OSDI'22）：iteration 级调度起源；Sarathi-Serve：chunked prefill 混批
- [ ] DistServe（OSDI'24）：PD 分离 + goodput 对齐配置；Mooncake（FAST'25）：KVCache
      为中心的 RDMA 分离集群；Splitwise、Preble（前缀感知全局调度）——了解核心思想
- [ ] LMCache / AttentionStore：KV 层级存储
- [ ] 投机解码：Leviathan/Chen 2023、Medusa、EAGLE；KV 量化：KIVI、KVQuant、IntactKV
- [ ] 本项目对齐关系表：README「与开源项目的对应关系」——面试前过一遍

## 15. 自测标准（每条都能做到再上考场）

- [ ] 能在白纸上画出：block table ↔ 物理 KV 的映射；radix 树的匹配与驱逐
- [ ] 能报出数字：7B 模型 KV 每千 token ≈ 0.5GB（fp16）；8B decode 单步 ≈ 10ms 量级；
      TP=2 每层 2 次 all-reduce
- [ ] 能讲清 3 个 trade-off：COW vs 不缓存部分块；recompute vs swap；
      prefix-affinity 路由 vs 负载均衡
- [ ] 能讲 2 个调试故事：KV pool fp16 硬编码（交叉验证发现）；
      调度器状态污染（swap 序列溜进普通 admission）——`PROGRESS.md` #9/#12
- [ ] 能 1 分钟讲完项目（`interview_qa.md` 开场白）
