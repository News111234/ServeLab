# ServeLab 架构：每个模块"是什么、像谁、差在哪"

> 读代码前先读这篇。每个小节给出：机制一句话 → 对应源码文件 → 与上游开源实现的差异
> （面试官最爱问"你的和 vLLM 的有什么区别"）。

## 1. Paged KV Cache 与块管理

**文件**: `kv_cache/manager.py`、`kv_cache/pool.py`

机制：KV cache 按 `block_size`（默认 16 token）分块，逻辑上每条序列持有一个
block table，物理行号 = `block_id * block_size + offset`（"slot"）。
物理池是扁平张量 `[num_layers, num_rows, num_kv_heads, head_dim]`，K/V 各一份。

与 vLLM 的差异（重要，面试高频）：
- vLLM v0 的 `BlockSpaceManager` 用 free_block_queue + hash 表 + copy-on-write；
  v1 简化为 hash 全块匹配、无 COW。ServeLab 采用 **v1 的取舍**：只缓存完整块，
  部分 block 永不进树，因此**共享块不可变，天然免 COW**。
- vLLM 的块分配与 prefix 匹配耦合在一个管理器里；ServeLab 把
  **物理池（pool） / 前缀树（radix） / 块管理（manager） 三层拆开**，
  淘汰策略因此可以整体替换（见 §2）。

关键不变量（测试在盯着）：
1. 任意 running 序列 `num_computed_tokens == len(token_ids) - 1`（decode 时）；
2. 块要么在 free 池、要么在树里、要么在某序列的 table 里，三者互斥（树上锁定的块
   可以同时被多个 table 引用，锁计数保护）。

## 2. Radix 前缀缓存与可插拔淘汰

**文件**: `kv_cache/radix.py`、`kv_cache/policies.py`

机制：树的一个节点 = 一个完整块，key = `blake2b(父链哈希, 块内容)`。两个请求只要
前缀的完整块序列一致，就共享物理 KV。匹配时对沿途节点 `lock += 1`（运行期不被
驱逐）；序列结束/被抢占时 `unlock` 并把新增块 `insert` 回树，内容重复的块作为
dup 返回并释放。

与 SGLang 的差异：
- SGLang 节点可以携带多块并支持分裂（insert 时 split）；ServeLab 单块单节点，
  无需分裂，代码少一半，代价是树的深度 = 前缀块数（查找 O(块数)，可以接受）。
- SGLang 的叶子驱逐是堆实现；ServeLab 把"挑谁驱逐"抽象成 `EvictionPolicy`
  接口（LRU / LFU / CostAware 三个内置实现），**这是论文方向 #1 的插槽**。

## 3. 调度器

**文件**: `engine/scheduler.py`（torch-free，配 `tests/test_scheduler.py` 的
FakeDriver 可独立驱动）

一步（`schedule()`）做三件事：
1. **decode**：给所有 prefill 完成的 running 序列预留下一 token 的 slot；
2. **in-flight chunked prefill**：没做完的 prefill 优先续块（vLLM v1 混批风格，
   token 预算 `max_num_batched_tokens` 内 decode 与 prefill chunk 混合）；
3. **admission**：按策略排序 waiting 队列（FCFS / Priority / SJF-predicted），
   先做前缀匹配（锁块）再规划 chunk；内存不够则 head-of-line blocking。

**Preemption**：decode 预留 slot 失败 → 按策略从 running 里挑牺牲者
（`recompute` 模式：释放块进树 → token 全部重算 → 前缀缓存会大量命中，
重算成本远低于首次）。`swap` 模式和 `offload` 联动是预留的扩展点。

与 vLLM 差异：v0 不混批（有 waiting 时 decode/prefill 交替整批），v1 混批；
ServeLab 两种都实现了（`chunked_prefill` 开关），并额外暴露
**调度策略本身**（`SchedulingPolicy.order/preempt_order`）为插槽。

## 4. 引擎与模型

**文件**: `engine/engine.py`、`models/loader.py`、`models/decoder.py`

- forward 采用 vLLM v0 的**打包批**：所有序列的 token 摊平成 `[T, H]`，
  每层做一次 QKV/MLP 投影，注意力按序列循环（gather paged KV + float32
  softmax + GQA repeat_interleave）。正确性优先；性能路径是 Triton kernel。
- RoPE 用 NeoX rotate-half（LLaMA/Qwen 同款）；RMSNorm float32 计算。
- 权重直接吃 HF checkpoint（safetensors 单文件 / 分片 / pytorch_model.bin），
  名字带 `model.` 前缀自动剥离，`tie_word_embeddings` 支持。
- 一致性是引擎的验收标准（`tests/test_engine_smoke.py`）：
  {prefix cache on/off} × {chunked prefill on/off} × {batch/单条} 输出必须逐 token 一致。

## 5. KV 量化

**文件**: `kv_cache/quant.py`、`quant/linear.py`

- KV：`none / int8 / fp8(e4m3)`，统一 **per-(row, head) 对称量化**——
  一行 = 一个 token 一个 kv head 的向量，写时量化、读时反量化，
  不需要重校准也不会有跨行依赖。粒度本身是研究轴：
  per-tensor（vLLM fp8-KV）→ per-block → per-token（KIVI 的 V）→ per-channel
  （KIVI 的 K，需要转置布局，roadmap #3）。
- fp8 在 CPU 上 `index_copy_` 未实现，所以物理池用 **uint8 视图**存储 fp8
  载荷（同 element size），读写时 `.view()` 往返——细节见 `pool.py` 注释。
- W8A8 Linear：per-token 激活 × per-channel 权重（SmoothQuant/compressed-tensors
  同款数值方案）。CUDA 上应替换成 `torch._int_mm`/CUTLASS；这里保留反量化
  参考实现用于验证数值误差（< 5% rel err，有测试）。

## 6. 模拟器（论文主战场）

**文件**: `simulator/*`（torch-free）

- **成本模型** `model_cost.py`：一步耗时 = max(FLOPs/FLOPs_eff, Bytes/BW_eff) +
  launch 开销。FLOPs = 2·P·(prefill tokens + decode batch)，Bytes = 权重 + 全量 KV。
  `GPU_PRESETS` 给的是**可达值**（不是 datasheet 峰值），并预留了用真实
  benchmark 校准的接口（roadmap #5：模拟器-实机 gap 校准本身就是可发的工作）。
- **副本** `cluster.py::Replica`：直接复用 `RadixCache` 做前缀命中统计——
  模拟器与真实引擎跑同一份缓存代码，策略改动两边行为一致。
- **PD 分离**：`pd_mode=True` 时副本分为 prefill/decode 两池，prefill 完成 →
  KV 传输时间 = KV字节/带宽 + 固定延迟 → 进入 decode 副本队列（DistServe/Mooncake
  的最简抽象）。
- **事件循环**：heapq 事件驱动；**过期副本事件直接丢弃**（忙副本必有
  `busy_until` 时刻的真实事件在堆里），这是修过一个 O(重复事件级联) 的性能坑
  之后的结论，注释留在代码里。

## 7. 测试地图

| 测试文件 | 盯什么 |
|---|---|
| `test_block_manager.py` | 分配/释放/驱逐/锁/slot 映射 |
| `test_radix.py` | 匹配/插入/去重/锁保护驱逐/LRU vs LFU 语义 |
| `test_scheduler.py` | 预算切分/准入顺序/优先级/抢占/死锁检测 |
| `test_simulator.py` | 全请求完成/指标口径/路由对比/PD/MoE 重平衡 |
| `test_quant.py` | 量化往返误差/池读写/块行映射/W8A8 |
| `test_engine_smoke.py` | 端到端一致性（4 配置组合）/前缀命中/抢占路径/不变量 |

## 8. 张量并行（TP）

**文件**: `parallel/layers.py`、`parallel/tp_model.py`、`parallel/simulated.py`、
`parallel/engine_group.py`、`examples/run_tp_gloo.py`

- 切分规则与 Megatron 一致：q/k/v、gate/up 为**列并行**（输出维切，无通信）；
  o_proj、down_proj 为**行并行**（输入维切，输出 all-reduce）；embedding 与
  lm_head 为**词表并行**（mask+reduce、all-gather）。注意力头/KV 头按 rank
  分片——**注意力本身零通信**，每层只有两次 row-parallel all-reduce。
- 两个执行后端共用一个 `Collective` 接口：`SimulatedCollective`（单进程内 W 个
  rank 线程在集合通信点会合，CPU/CI 可跑）与 `GlooCollective`（真 torch.distributed
  进程组，GPU 机器换 nccl 即可）。集合通信带"毒化"机制：任一 rank 崩溃，其余
  rank 立即失败而不是死等（这是修过一次线程死锁后的结论）。
- 引擎是 **SPMD**：每个 rank 跑相同的确定性调度器，rank 0 采样并广播 token
  （vLLM 的 driver-only sampler 同款设计）。
- 验收标准：TP=2 / TP=4 与单卡输出**逐 token 一致**（`tests/test_parallel.py`）。
  实现时踩过的坑：并行层必须接收 rank 局部维度（全局切分是 `shard_state_dict_tp`
  的职责），否则会双重切分——形状断言第一时间就能抓到。

## 9. 投机解码

**文件**: `engine/speculative.py`、`scripts/debug/`（当时的对拍脚本）

- 每轮：提议器给出 k 个 draft → target 一次 teacher-force 前向 [last + drafts]
  得到 k+1 个分布 → 逐位接受/纠正（全对则第 k+1 个 token 是免费奖励）。
- 三种提议器：`ModelDraftProposer`（小模型自回归）、`PromptLookupProposer`
  （n-gram 查表，无模型，greedy-only，与 vLLM 同限制）。
- 采样温度 > 0 时走标准拒绝采样（残差分布 norm(max(p-q,0))，带 p==q 全零
  残差的回退保护）。
- **正确性契约（测试钉死）**：greedy 投机输出与同 target 的普通 greedy **逐
  token 一致**，与提议器质量无关；self-draft 时接受率 = 1.0、每次前向发射
  ~k+1 个 token。
- 开发这模块时通过交叉验证发现并修复了 KV pool 的一个真 bug（"auto" dtype
  把 fp32 模型的 KV 硬编码成 fp16 存储）——见 `PROGRESS.md` 缺陷 #9。
  `scripts/debug/` 留着当时的逐层对拍脚本，面试可以讲这个调试过程。

## 10. KV 层级：SWAP 抢占与驱逐归档

**文件**: `kv_cache/swap.py`、`manager.py`（swap_out/swap_in/restore）、
`scheduler.py`（swap 分支）

- **SWAP 抢占**（`preemption_mode="swap"`）：被抢占序列的私有 KV 块拷到
  host（`CPUSwapSpace`），matched 前缀块留在树里；恢复时重新匹配前缀 +
  原样拷回，**生成进度零丢失**。前缀覆盖数必须与换出时严格一致，否则回退
  recompute（防止私有数据错位）。
- **驱逐归档 + 恢复**（`CPUKVOffloader`）：radix 驱逐叶子时把 (内容哈希,
  token_ids, KV) 归档到 host；之后的前缀匹配在树 miss 处自动查档，命中则
  分配新块写回并挂回树（`RadixCache.attach`）——冷前缀"驱逐"变成"降级"。
- 调度器保证：换出的序列**只能**经 swap-in 通道回到 running（普通 admission
  会跳过它们），decode 循环遍历快照时跳过本轮已被抢占的序列——这两个不变量
  是修过两个状态污染 bug 后钉进代码和测试的。
