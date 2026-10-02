# ServeLab 项目进展

> 状态：**v0.1.0 · 全部计划内功能已实现并通过本机验证**
> 环境：Windows 11 / RTX 4050 Laptop 6GB / Anaconda Python 3.12.7（torch 2.13 CPU 版）
> 配套文档：[TODO.md](TODO.md)（下一步）· [architecture.md](architecture.md)（代码导读）·
> [research_roadmap.md](research_roadmap.md)（论文选题）· [interview_guide.md](interview_guide.md)（面试映射）

## 1. 里程碑

| # | 里程碑 | 内容 | 状态 |
|---|---|---|---|
| M1 | 项目骨架 | pyproject/目录结构/配置 dataclass（`config.py`） | ✅ 完成 |
| M2 | KV Cache 核心（torch-free） | Paged BlockManager、Radix PrefixCache（内容寻址、锁保护）、可插拔淘汰策略（LRU/LFU/CostAware）、FP8/INT8 KV 量化、CPU Offload 原语 | ✅ 完成 |
| M3 | 调度器（torch-free） | Continuous Batching、Chunked Prefill、Recompute 抢占、可插拔准入策略（FCFS/Priority/SJF-predicted）、前缀匹配-撤销事务 | ✅ 完成 |
| M4 | torch 引擎 | 打包批执行、Qwen2/LLaMA decoder（HF 权重直载）、paged attention（torch 参考）、sampler、vLLM 风格 API（`LLMEngine.generate`） | ✅ 完成 |
| M5 | 量化 | KV per-(row,head) INT8/FP8（CPU 上经 uint8 视图存 fp8）、W8A8 Linear 参考实现 | ✅ 完成 |
| M6 | 集群模拟器（torch-free） | 解析式 step-time 成本模型（GPU/Model presets）、4 种 Router、4 种预测器、PD 分离模式、MoE EP 负载仿真、TTFT/TPOT/goodput/Jain 指标 | ✅ 完成 |
| M7 | Triton 算子 | PagedAttention decode kernel（在线 softmax、GQA）、RMSNorm kernel（需 Linux+GPU 验证） | ✅ 代码完成 ⚠️ 待 GPU 验证 |
| M8 | Bench 与应用 | benchmark_serving（fixed/poisson 负载）、kernel_bench、三份示例脚本、OpenAI 兼容 server（非流式） | ✅ 完成 |
| M9 | 测试体系 | 62 tests（43→62）+ 2 个 GPU-gated skip；引擎四配置一致性 + TP 一致性 + 投机解码一致性作为验收标准 | ✅ 完成 |
| M10 | 文档 | README、代码导读、论文选题 ×6、面试映射、bench 指南、trace 清单、TODO/进展（本文件） | ✅ 完成 |
| M11 | **张量并行** | Megatron 切分规则（列/行/词表并行）、SPMD 引擎、Simulated(线程)+Gloo(进程) 双后端、集合通信毒化机制；TP=2/4 与单卡输出逐 token 一致 | ✅ 完成（gloo 后端待 GPU 机器验证） |
| M12 | **投机解码** | SimpleKVRunner、Model/PromptLookup 提议器、greedy 验证 + 拒绝采样、SpecStats（接受率/前向调用摊销）；greedy 输出与普通解码严格一致 | ✅ 完成 |
| M13 | **KV 层级** | SWAP 抢占模式（CPU 换出/换回，进度零丢失）、驱逐归档（CPUKVOffloader）+ 前缀未命中自动恢复（radix.attach） | ✅ 完成 |
| M14 | **研究工具链** | 成本模型参数校准器（三参数交替最小二乘）、sweep 实验流水线（网格→CSV）、调试脚本归档 scripts/debug/ | ✅ 完成 |

## 2. 验证状态

**本机已验证（Windows，CPU）：**

- `pytest tests -q` → **62 passed, 2 skipped**（skip 的是 Triton 对拍，需 Linux+GPU）：
  - 块管理 6（分配/释放/前缀复用/锁保护驱逐/AllocStatus/无缓存模式）
  - 前缀树 5（匹配往返/LRU 顺序/锁驱逐/策略注册/LFU 语义）
  - 调度器 8（完成性/chunk 预算/FCFS 顺序/优先级/前缀命中/内存压力抢占/并发上限/中止）
  - 模拟器 7（全请求完成/指标口径/四种路由/前缀亲和提升命中/PD 分离/预测器差异/MoE 重平衡）
  - 量化 7（INT8/FP8 往返误差/池读写/块行映射/W8A8 误差）
  - 引擎 10（端到端/四配置一致性/前缀命中输出一致/批与单条一致/长 prompt 切片一致性/INT8 KV 一致性/抢占路径/KV 增长不变量/metrics/abort）
  - **张量并行 5**（分片规则往返/切片/TP 模型对拍/TP=2 引擎一致/TP=4 引擎一致）
  - **投机解码 6**（自投机全接受+摊销/prompt-lookup/随机 draft/引擎一致/采样确定性/边界）
  - **swap+offload 5**（swap 往返/块表恢复/调度器进度保持/驱逐归档+恢复/引擎 swap e2e）
  - **校准+sweep 3**（真值恢复/最小样本/网格行）
- 示例脚本实跑通过：`run_engine.py`、`run_simulator.py --compare-routers`、`moe_sim.py`、
  模拟 TP 引擎组（TP=2/4 与单卡逐 token 一致）

**待验证（需要其他环境）：**

- Triton kernel 对拍（`tests/test_triton_kernels.py`，需 Linux+GPU+triton）
- CUDA torch 实机吞吐（本机 torch 为 CPU 版；4050 装 CUDA 版后即可）
- 真实大模型端到端（Qwen2.5-0.5B/7B）与 `benchmark_serving` 实测
- OpenAI server（需 `pip install fastapi uvicorn`）

## 3. 关键设计决策（ADR 摘要）

| 决策 | 选择 | 理由 / 代价 |
|---|---|---|
| 前缀缓存粒度 | 只缓存完整块，**不做 COW** | 共享块不可变 → 免拷贝路径；与 vLLM v1 同款取舍。代价：部分块内容不共享（命中率略降） |
| 缓存架构 | 物理池 / 前缀树 / 块管理三层拆开（SGLang 风格） | 淘汰策略可整体替换（论文插槽 #1）；代价：三层需维护互斥不变量 |
| 前缀树结构 | 单块单节点，key=blake2b(父链,内容) | 免节点分裂，代码减半；代价：树深=前缀块数 |
| 引擎注意力 | 打包批 + 每序列 gather 的 float32 softmax（正确性优先） | 可读可对拍；性能路径交 Triton kernel |
| 模拟器缓存 | 直接复用引擎 RadixCache 类 | 策略改动在 sim 与实机行为一致（论文方法论卖点） |
| 成本模型 | roofline 解析式（可达 FLOPs/BW 常数） | 秒级迭代；校准接口留给方向 #5 |
| 事件循环 | 副本过期事件直接丢弃（不重调度） | 修复 O(重复事件) 级联；忙副本必有 busy_until 时刻的真实事件 |
| KV 量化基线 | per-(row,head) 对称量化 | 无跨行依赖、免重校准；粒度本身是研究轴（方向 #3） |

## 4. 开发期缺陷记录（面试调试素材）

开发-验证过程中发现并修复的真 bug，每一个都对应一条测试：

1. **decode 路径不更新 `num_computed_tokens`**（`engine/engine.py`）
   现象：长生成时 `slot_for_token` 越界。根因：prefill 分支更新了 computed，
   decode 分支漏更新，`computed == len-1` 不变量被破坏；此前测试因块容量富余未暴露。
   修复：decode 时置 `computed = get_len()`；新增不变量回归测试
   `test_kv_growth_invariant`。教训：**不变量要有专门的测试盯着**。
2. **调度器 admission 中断未撤销前缀匹配**（`engine/scheduler.py`）
   现象：preemption 压力测试中 `maybe_match_prefix` 重复断言失败。
   根因：admission 失败路径只 break，已锁定的树节点没有解锁/移除条目。
   修复：新增 `BlockManager.undo_match`（只解锁、不动池），并区分
   `release`（有 KV 内容要回收）与 `undo_match`（纯锁操作）。
3. **release 的内容与块表不一致**（`engine/scheduler.py` + `kv_cache/manager.py`）
   现象：radix.insert 长度断言失败。根因：刚采样、还没写入 KV 的最后一个
   token 被计入回收内容。修复：所有 release 调用点改为
   `token_ids[:num_computed_tokens]`。
4. **模拟器重复事件级联**（`simulator/cluster.py`）
   现象：200k 事件只推进 29 秒仿真时间（应 ~8k 事件）。
   根因：每步给所有副本 push 事件，与堆中已有 pending 事件重复，重复事件
   又自我重调度，事件数乘性膨胀。修复：过期事件直接丢弃；只"踢醒"空闲副本。
5. **PD 分离的 computed 语义错位**（`simulator/cluster.py`）
   现象：decode 副本上请求永不产出 token。根因：照抄引擎的
   `computed = prompt-1` 技巧（引擎里用于给最后 prompt token 产 logits），
   但模拟器 decode 分支只追加输出 token，序列永远"prefill 未完成"。
   修复：PD 语义下整个 prompt KV 已随传输到达，`computed = prompt_len`。
6. **HF 权重名前缀未剥离**（`models/decoder.py`）
   现象：所有权重被当 unexpected 丢弃，模型为随机初始化。
   修复：加载时剥离 `model.` 前缀；并对缺失权重改为显式报错。
7. **CPU 上 fp8 `index_copy_` 未实现**（`kv_cache/pool.py`）
   修复：fp8 载荷经 uint8 视图存储（同 element size），读写时 `.view()` 往返。
8. **policies ↔ radix 循环导入**：类型引用改 `TYPE_CHECKING` +
   `from __future__ import annotations`。
9. **KV pool "auto" dtype 硬编码 fp16**（`kv_cache/pool.py` + `quant.py`）——
   由投机解码模块的交叉验证发现：CPU float32 模型的 KV 被静默降到 fp16 存储，
   每次写读 2.4e-4 舍入误差被逐层放大后导致 argmax 相位翻转、输出发散。
   修复：auto 跟随 compute dtype，`build_kv_quantizer` 补 float32/bfloat16 分支。
   教训：**两套独立实现的交叉验证（cross-validation）是找数值 bug 的利器**。
10. **并行层双重切分**（`parallel/layers.py`）：构造函数内部又除以 world，
    而调用方已传入 rank 局部尺寸 → 形状减半。修复：明确"层接收局部维度，
    全局切分是 shard_state_dict_tp 的职责"。
11. **模拟 TP 的两处并发 bug**：rank 存在线程局部（构造期读 rank 崩溃 → 改为
    前向时解析）；lm_head all-gather 只有 rank0 干活 → 永久等待（all-gather
    必须全员到场）。另加集合通信"毒化"机制：任一 rank 崩溃，其余线程立刻
    失败而非死等——否则一个异常就变成测试挂死。
12. **swap 模式的两个状态污染 bug**（`scheduler.py`）：被换出序列溜进普通
    admission 路径（其 swap 元数据被无视、状态被重建）→ admission 循环对
    `_swapped` 序列阻塞；decode 循环遍历 running 快照时访问本轮已被抢占的
    序列（块表已弹）→ 快照遍历加守卫。两个 bug 都被引擎 e2e swap 测试抓住。
13. **投机解码 context 缺 prompt**（`engine/speculative.py`）：generate 的
    tokens 列表从首个采样 token 起步，proposer 拿到的 context 只有 1 个 token，
    提议全错。修复：tokens 从 prompt 起步、返回时切片输出部分；顺带修复
    跨 max_tokens 边界的截断与 runner 复用时的缓存重置。

## 5. 已知限制（= TODO 的来源）

- Triton kernel 未在 GPU 实测；flash-decoding split-K、CUDA Graph 未实现
- SWAP 换出/换回走 float32 语义（经 read/write 往返），量化 KV 的 swap 有一次
  反量化-重量化；swap-in 失败率与前缀驱逐的交互已处理但未做系统化压测
- TP 引擎为 SPMD 确定性调度（无独立 driver/worker 进程拆分）；无 PP/EP 的
  引擎级实现（EP 有模拟器层）
- 投机解码是单序列模块（未与 continuous batching 融合，cf. vLLM V1 spec-decode）
- server 非流式；无 beam search / parallel sampling；不支持 AWQ/GPTQ 权重
- radix 驱逐用线性查找（研究规模够用，生产规模需叶子堆）
- 模拟器成本模型已可校准但未接真实测量数据；PD 传输是常带宽模型（无分块流水）
- 引擎注意力为逐序列循环（正确性优先），无 flash-attn 库接入

## 6. 变更记录

- **v0.2.0（2026-10-02）**：四大机制升级——① 张量并行（Megatron 切分规则 +
  SPMD 引擎 + 模拟/gloo 双后端 + 集合通信毒化，TP=2/4 与单卡逐 token 一致）；
  ② 投机解码（提议/验证/拒绝采样完整算法，greedy 输出严格一致，含交叉验证
  发现并修复的 KV pool fp16 真 bug）；③ KV 层级闭环（SWAP 抢占 + 驱逐归档 +
  前缀自动恢复）；④ 研究工具链（成本模型校准器 + sweep 实验流水线）。
  测试 43→62。新缺陷记录 #9~#13。
- **v0.1.0（2026-10-02）**：首次完整交付。三层架构（核心/引擎/模拟器）+ Triton
  算子 + bench + 示例 + server 骨架 + 43 测试全绿 + 六份文档。
  已知缺陷 8 项全部修复并有测试锚定。
