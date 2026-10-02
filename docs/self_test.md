# ServeLab 知识自测卷（选择 + 填空）

> 用法：闭卷做完再对答案（答案与解析在卷尾）。目标是 ≥85 分（每题 2 分，满分 100）。
> 错的题按解析里指的文档/源码回去复习，对应 `docs/knowledge_checklist.md` 的勾。
> 难度：☆ 基础概念 ★ 进阶 ★★ 项目细节/数字。

---

# A 卷 · 选择题（25 题，每题 2 分）

## A1 两阶段与瓶颈

**A1.1**☆ Decode 阶段的主要瓶颈是：
A. 计算单元算力  B. 显存带宽  C. kernel launch 数量  D. 网络通信

**A1.2**☆ 随 decode batch 增大，单步耗时的变化趋势最接近：
A. 线性增长  B. 平方增长  C. 几乎不变  D. 指数下降

**A1.3**★ 8B 模型 fp16 权重 16GB，在有效带宽 1.4TB/s 的卡上，decode 单步（batch=1）读权重大约需要：
A. 0.1ms  B. 1ms  C. 11ms  D. 100ms

**A1.4**★ Prefill 的算术强度（FLOPs/Byte）随 prompt 长度 n 的变化是：
A. 正比于 n  B. 反比于 n  C. 与 n 无关  D. 正比于 n²

## A2 KV Cache 与 PagedAttention

**A2.1**☆⭐ LLaMA-2-7B（32 层、MHA、head_dim=128）fp16 KV Cache 每 token 占：
A. 64KB  B. 128KB  C. 256KB  D. 512KB

**A2.2**☆⭐ PagedAttention 消除的主要是：
A. 计算冗余  B. 按最大长度预分配导致的外碎片  C. 数值精度损失  D. 通信量

**A2.3**★ PagedAttention 下每个序列的内部碎片（预留未用）平均约为：
A. 一个块  B. 半个块  C. 两个块  D. 0

**A2.4**★★ vLLM v1 相比 v0 在块管理上最重要的简化是：
A. 引入 copy-on-write  B. 改为整块哈希匹配、去掉 COW  C. 放弃前缀缓存  D. 改用链表管理块

**A2.5**★★ 为什么 ServeLab/vLLM v1 不需要块级 copy-on-write？
A. 因为块很小  B. 因为只缓存完整块，共享块不可变  C. 因为用了 CAS 原子操作  D. 因为 KV 是只读张量

## A3 前缀缓存与淘汰

**A3.1**☆⭐ ServeLab 前缀块的哈希计算方式是：
A. 对单个块内容独立哈希  B. blake2b(父块哈希, 块 token) 链式哈希  C. 对整条 prompt 哈希  D. md5(块内容+序列号)

**A3.2**★ RadixCache 里"锁计数 > 0"的节点意味着：
A. 正在被写坏  B. 被运行中的序列持有，不可驱逐  C. 待删除  D. 哈希冲突

**A3.3**★ LRU 淘汰策略在 prefix cache 场景的主要缺陷是：
A. 实现太复杂  B. 不考虑重算成本与未来复用概率  C. 不支持并发  D. 内存占用高

## A4 调度

**A4.1**☆⭐ Continuous batching（Orca）的核心思想是：
A. 把 batch 拆成 micro-batch  B. iteration 级调度，每步重组 batch  C. 请求按长度排序  D. 用 CUDA Graph 加速

**A4.2**☆⭐ Chunked prefill 主要改善的指标是：
A. 模型精度  B. 长 prefill 阻塞导致的 decode 尾延迟  C. 权重加载速度  D. KV 占用

**A4.3**★ 抢占模式下，"生成进度零丢失"的是：
A. recompute  B. swap  C. 两者都是  D. 两者都不是

**A4.4**★★ vLLM v1 只保留 recompute 抢占、去掉了 swap，最可能的理由是：
A. swap 需要 RDMA  B. 前缀缓存使重算成本大幅下降，swap 的 PCIe 拷贝不划算  C. swap 有精度损失  D. 显存太大装不下

## A5 投机解码

**A5.1**☆⭐ 投机解码一轮全接受时，target 一次前向能产出多少个 token（提议 k 个）？
A. k  B. k+1  C. 2k  D. 1

**A5.2**★ 采样温度>0 时，draft token 被接受的概率是：
A. p(d)  B. min(1, p(d)/q(d))  C. q(d)/p(d)  D. 1-q(d)

**A5.3**★ 投机解码拒绝后用于出纠正 token 的分布是：
A. 均匀分布  B. p  C. norm(max(p-q, 0))  D. q

**A5.4**★★ 投机解码"净赚"的条件最接近：
A. draft 与 target 参数量相同  B. draft 单步成本 << target 且接受率足够高  C. k 越大越好  D. 接受率 > 10% 即可

## A6 量化

**A6.1**☆ FP8 E4M3 格式的可表示最大值是：
A. 127  B. 256  C. 448  D. 65504

**A6.2**★ KIVI 对 K 和 V 分别采用：
A. K per-token，V per-channel  B. K per-channel，V per-token  C. 都 per-tensor  D. 都二值化

**A6.3**★ SmoothQuant 的核心操作是：
A. 只量化权重  B. 把激活的异常值迁移到权重，实现 W8A8  C. 用 Hessian 补偿误差  D. 按激活保护显著权重

**A6.4**★ GPTQ 量化误差补偿依赖的是：
A. 激活统计  B. 逐层的 Hessian 信息  C. 蒸馏损失  D. 梯度检查点

## A7 并行与通信

**A7.1**☆⭐ Megatron TP 中 o_proj / down_proj 属于：
A. 列并行  B. 行并行（输出需 all-reduce）  C. 词表并行  D. 不切分

**A7.2**☆⭐ TP 切分注意力头之后，注意力计算本身需要的集合通信次数（每层）是：
A. 0  B. 1  C. 2  D. 4

**A7.3**★ ring all-reduce 的总通信量因子是：
A. (K-1)/K  B. 2(K-1)/K  C. K  D. 2K

**A7.4**★★ GQA 模型在 TP=4 下正常工作的必要条件是：
A. hidden_size % 4 == 0  B. kv_heads % 4 == 0  C. 层数 % 4 == 0  D. 词表 % 4 == 0

**A7.5**★★ PP（流水线并行）有 p 个流水级、m 个 micro-batch 时的 bubble 比例：
A. (p-1)/(m+p-1)  B. (m-1)/(m+p)  C. p/m  D. 1/p

## A8 MoE 与通信原语

**A8.1**☆ MoE 专家并行中，token 分发与聚合使用的通信原语是：
A. all-reduce  B. all-gather  C. all-to-all  D. broadcast

**A8.2**★ MoE token 被丢弃（dropped）的直接原因是：
A. 通信超时  B. 超过 capacity factor 限制的专家容量  C. 精度不足  D. 路由 softmax 为零

## A9 算子与工具

**A9.1**☆⭐ FlashAttention 的核心技巧是：
A. 低秩分解  B. 分块 + online softmax（不物化完整 S 矩阵）  C. 稀疏注意力  D. INT8 计算

**A9.2**☆ CUDA Graph 主要解决 decode 阶段的：
A. 显存碎片  B. kernel launch 开销  C. 数值精度  D. 通信延迟

**A9.3**★ 想看"kernel 之间的空隙与 launch 开销"应该用：
A. Nsight Systems  B. Nsight Compute  C. py-spy  D. cProfile

**A9.4**★ Nsight Compute 里 DRAM% 很高、SM% 很低说明该 kernel：
A. 计算受限  B. 访存受限  C. launch 受限  D. 通信受限

## A10 指标与框架

**A10.1**☆⭐ TPOT 的定义是：
A. 首 token 时延  B. 输出阶段平均每 token 时延（不含首 token）  C. 端到端总时延  D. 每秒 token 数

**A10.2**☆⭐ Goodput 的定义是：
A. 有效吞吐 tok/s  B. 同时满足 TTFT 与 TPOT SLO 的请求占比  C. GPU 利用率  D. 去重后的吞吐

**A10.3**★ Jain 负载均衡指数的计算式是：
A. max/mean  B. (Σx)²/(n·Σx²)  C. Σx²/n  D. mean/max

**A10.4**★ 提出 goodput 对齐 PD 分离配置方法的系统是：
A. Orca  B. Sarathi-Serve  C. DistServe  D. FlashAttention

**A10.5**★★ "KVCache 为中心、用 RDMA 在分离集群传输 KV"的系统是：
A. Mooncake  B. vLLM  C. TensorRT-LLM  D. SGLang

---

# B 卷 · 填空题（25 空，每空 2 分）

**B1.**☆ PagedAttention 中，某 token 的物理行号（slot）= ______ × block_size + ______。

**B2.**☆⭐ KV Cache 每 token 字节 = 2（K+V）× ______ × kv_heads × head_dim × dtype 字节数。

**B3.**★ LLaMA-3-8B（32 层、8 个 KV 头、head_dim=128）fp16 下 KV Cache 每 token = ______ KB。

**B4.**☆⭐ Prefill 是 ______ 受限，Decode 是 ______ 受限。

**B5.**★ Roofline 单步时间模型：t = max(______/F_eff, ______/BW_eff) + launch 开销。

**B6.**☆⭐ ServeLab 只缓存______块，因此共享块不可变、无需 COW。

**B7.**★ ServeLab 前缀块哈希 = blake2b(______，块 token ids)，前缀一致 ⇒ 整条 ______ 一致。

**B8.**★ RadixCache 驱逐只从 ______ 为 0 的 ______ 节点里选（策略可插拔：lru/lfu/costaware）。

**B9.**☆⭐ Continuous batching 提出自 ______（系统名，OSDI'22）。

**B10.**★ ServeLab 调度器每步三阶段顺序：先 decode 预留 slot，再续跑 in-flight ______，最后按策略从 ______ 队列准入。

**B11.**★ SWAP 抢占恢复时要求前缀覆盖与换出时______，否则回退 ______ 模式。

**B12.**☆⭐ 投机解码一轮提议 k 个、全接受时共发射 ______ 个 token；greedy 输出与普通解码______（关系）。

**B13.**★ 投机解码的净收益来自把 k+1 次 target 前向合成 ______ 次，decode 访存受限时多 token 前向的单步时延几乎______。

**B14.**☆ SmoothQuant 实现 W8A8 时，激活的异常值被______到权重侧。

**B15.**☆ AWQ 是______-only 量化，用______分布决定保护哪些权重通道。

**B16.**☆⭐ Megatron TP 里，QKV/gate/up 是______并行（无通信），o_proj/down_proj 是______并行（输出 all-reduce）。

**B17.**★ TP 下每层共 ______ 次 all-reduce；ring all-reduce 通信量因子是 ______（用 K 表示）。

**B18.**★ ServeLab TP 引擎是 ______ 架构：每个 rank 跑相同确定性调度器，采样结果由 rank ______ 广播。

**B19.**☆ ServeLab 的 TP 有两个通信后端：线程级 ______ 后端（CI 可跑）和真 torch.distributed ______ 后端。

**B20.**☆⭐ 模拟器与真实引擎共用同一个 ______ 类，保证策略代码两边行为一致。

**B21.**★ ServeLab 成本校准器拟合的三个参数是 flops_eff、______ 和 ______。

**B22.**☆⭐ MoE 推理显存必须容纳______专家，但每 token 只激活 ______ 个。

**B23.**★ ServeLab 三大一致性验收：四种调度/缓存配置组合输出一致、TP=2/4 与单卡一致、投机解码与______一致。

**B24.**☆ Nsight Systems 看时间线与______，Nsight Compute 看单 kernel 的 ______ 指标（SM%/DRAM%）。

**B25.**★★ ServeLab 用交叉验证发现过一个真 bug：KV pool 的 auto dtype 被硬编码成 ______ 存储，导致 float32 模型的 KV 舍入误差逐层放大。

---
---

# 答案与解析

## A 卷

| 题 | 答案 | 解析与复习位置 |
|---|---|---|
| A1.1 | **B** | decode 每步要重读全部权重+KV，8B 模型 ≈ 11ms（`knowledge_checklist.md` §2） |
| A1.2 | **C** | 访存受限：权重/KV 只读一遍，batch 大小几乎不影响单步时延 |
| A1.3 | **C** | 16GB / 1.4TB/s ≈ 11.4ms（§2） |
| A1.4 | **A** | FLOPs=2Pn、Bytes≈2P，强度 ∝ n（§2） |
| A2.1 | **D** | 2×32×32×128×2B = 512KB/token（§3） |
| A2.2 | **B** | 按最大长度预分配的外碎片；内碎片降到约半块/序列（§3） |
| A2.3 | **B** | 平均半个块（SOSP'23 论文口径） |
| A2.4 | **B** | v1 整块哈希匹配、无 COW（§3；README 对照表） |
| A2.5 | **B** | 完整块不可变 → 序列只写私有块（`architecture.md` §1） |
| A3.1 | **B** | 链式哈希保证前缀一致 ⇒ 哈希链一致（`utils/common.py`） |
| A3.2 | **B** | 锁计数保护运行中序列的块（`kv_cache/radix.py`） |
| A3.3 | **B** | 不看重算成本×复用概率 → 论文方向 #1（`policies.py`） |
| A4.1 | **B** | Orca 的 iteration 级调度（§5） |
| A4.2 | **B** | 长 prefill 不再独占引擎步，TPOT 尾部显著改善（§5） |
| A4.3 | **B** | swap 拷回 KV，进度零丢失（`kv_cache/swap.py`） |
| A4.4 | **B** | 前缀缓存让重算大概率命中，swap 收益变小 |
| A5.1 | **B** | k 个接受 + 第 k+1 个免费奖励 token（§6） |
| A5.2 | **B** | 标准拒绝采样 min(1, p/q)（`speculative.py::generate_sampling`） |
| A5.3 | **C** | 残差分布 norm(max(p-q,0))，保证输出分布等于 target |
| A5.4 | **B** | draft 小 1~2 个数量级、接受率 60~70%+ 才净赚 |
| A6.1 | **C** | E4M3 max=448（`kv_cache/quant.py`） |
| A6.2 | **B** | KIVI：K 的异常值集中在固定通道 → per-channel（§7） |
| A6.3 | **B** | SmoothQuant 平移异常值，两边都好量化 |
| A6.4 | **B** | GPTQ 基于 OBQ，用 Hessian 补偿逐列误差 |
| A7.1 | **B** | 行并行按输入维切，部分和 all-reduce（`parallel/layers.py`） |
| A7.2 | **A** | 头维度切分后各头独立 → 注意力零通信（`tp_model.py`） |
| A7.3 | **B** | ring all-reduce = 2(K-1)/K × 消息大小 |
| A7.4 | **B** | GQA 整除约束，引擎构造时 assert（`tp_model.py::TPAttention`） |
| A7.5 | **A** | PP bubble = (p-1)/(m+p-1) |
| A8.1 | **C** | MoE dispatch/combine 用 all-to-all（DeepEP 专为此设计） |
| A8.2 | **B** | 超过 capacity factor 的容量上限即丢弃（`simulator/moe.py`） |
| A9.1 | **B** | online softmax + IO-aware tiling，不物化 S 矩阵 |
| A9.2 | **B** | 一次捕获整段 kernel 序列，消除 launch gap（`enforce_eager` 开关） |
| A9.3 | **A** | Nsight Systems 看时间线；Compute 看单 kernel（§12） |
| A9.4 | **B** | DRAM% 高 = 访存受限，正是 decode attention 的正常形态 |
| A10.1 | **B** | TTFT + TPOT×(n-1) = 端到端（§13） |
| A10.2 | **B** | 同时满足双 SLO 的占比（DistServe 口径，`simulator/metrics.py`） |
| A10.3 | **B** | Jain = (Σx)²/(n·Σx²)，越接近 1 越均衡 |
| A10.4 | **C** | DistServe（OSDI'24） |
| A10.5 | **A** | Mooncake（FAST'25） |

## B 卷

| 空 | 答案 | 备注 |
|---|---|---|
| B1 | block_id；块内偏移 offset | `kv_cache/manager.py::slot_for_token` |
| B2 | 层数（num_layers） | §3 公式 |
| B3 | **128** | 2×32×8×128×2B = 128KB（GQA 4 倍压缩） |
| B4 | 计算（compute）；访存（memory） | §2 |
| B5 | FLOPs；Bytes | `simulator/model_cost.py::step_time` |
| B6 | 完整 | `architecture.md` §1 |
| B7 | 父块哈希（父链哈希）；哈希链 | `utils/common.py::hash_block_tokens` |
| B8 | 锁计数；叶子 | `kv_cache/radix.py::evict` |
| B9 | Orca | OSDI'22 |
| B10 | chunked prefill 分片；waiting | `engine/scheduler.py::schedule` 三阶段 |
| B11 | 严格一致（块数相等）；recompute | `manager.py::swap_in` |
| B12 | k+1；逐 token 严格一致（相同） | 测试钉死的契约 |
| B13 | 1；不变（几乎不增加） | §6 收益来源 |
| B14 | 迁移（平滑） | SmoothQuant |
| B15 | 权重（weight）；激活 | AWQ |
| B16 | 列（column）；行（row） | Megatron 规则 |
| B17 | 2；2(K-1)/K | 每层 attn-out + MLP-out 各一次 |
| B18 | SPMD；0 | rank0 采样并广播 |
| B19 | 模拟（SimulatedCollective）；gloo（GlooCollective） | `parallel/layers.py` |
| B20 | RadixCache（前缀缓存） | 模拟器与引擎同一实现 |
| B21 | bw_eff；launch 开销（overhead） | `simulator/calibration.py` |
| B22 | 全部；top-k | MoE 显存与算力解耦 |
| B23 | 普通解码（greedy 普通生成） | `tests/test_speculative.py` |
| B24 | kernel 间空隙（launch 开销）；SOL | §12 |
| B25 | fp16 | `PROGRESS.md` 缺陷 #9 |

---

## 评分与行动建议

- **≥90**：可以直接上考场，把 `interview_qa.md` 的 ⭐⭐⭐ 题背熟即可
- **75~89**：按错题的复习位置回 `knowledge_checklist.md` 对应小节过一遍，隔天重做错题
- **<75**：先精读 `docs/architecture.md` 全文 + 重跑一遍 `pytest tests -q` 对照测试名回忆机制，再回来重做整卷
