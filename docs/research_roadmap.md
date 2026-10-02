# 论文选题 Backlog（每条都留了插槽，选一条做深）

> 选题原则：(a) 在 ServeLab 的现有插槽上改动量 < 500 行；(b) 有明确的可跑
> baseline（本仓库已内置）；(c) 有公开 trace 可用；(d) 结论可先在模拟器上
> 验证、再移植回真实引擎复现。按"性价比"排序。

## #1 Workload-aware Prefix-Cache 淘汰/准入策略 ⭐推荐首做

- **插槽**: `kv_cache/policies.py::EvictionPolicy`（+ 新增 admission 接口）
- **动机**: LRU/SGLang 只看"最近"，不看"这条前缀重算多贵、再被请求的概率多大"。
  长系统提示词被挤掉的代价远高于短闲聊前缀。
- **做法**: 价值函数 `V(prefix) = P(reuse | 特征) × 重算成本 × 剩余生命`，
  特征：命中频率、长度、到达间隔、抖动度；P 可以是参数化模型或小在线学习器。
  再加一层**准入**控制（one-shot 超长前缀直接不缓存，省内存给热门前缀）。
- **Baseline**: LRU（内置）、LFU（内置）、CostAware（内置占位，就是让你替换的）、
  vLLM v1 hash-cache、SGLang RadixCache。
- **数据集**: ShareGPT（多轮强前缀）、Azure LLM Inference Trace、Mooncake trace。
- **指标**: 前缀命中率、TTFT p50/p99（命中省 prefill）、吞吐、内存占用。
- **风险**: 命中率提升可能被"驱逐长前缀的内存腾挪"抵消——需要报告
  命中率×成本联合指标。

## #2 输出长度预测驱动的调度与路由 ⭐与 #1 二选一

- **插槽**: `simulator/predictor.py::OutputLengthPredictor` +
  `engine/scheduler.py::SJFPredictedPolicy` + `simulator/router.py::PredictiveRouter`
- **动机**: SJF 和预测性路由都吃"预测的输出长度"；预测误差如何在
  TTFT/负载均衡上退化，没人给出系统性的测量与鲁棒化方案。
- **做法**: 在线预测器（回归 / 分位数 / 小 GRU / S3 式状态机）→
  预测置信度进入调度键（如 SJF 变成 expected-completion + 方差惩罚）；
  路由侧融合 cache 亲和与预测负载（`PredictiveRouter.prefix_bonus` 就是那个权重旋钮）。
- **Baseline**: FCFS、least-loaded、round-robin、prefix-affinity（全部内置，
  `examples/run_simulator.py --compare-routers` 一键出对比表）。
- **数据集**: BurstGPT（真实波动）、Azure（企业负载）、ShareGPT。
- **指标**: TTFT/TPOT 分位数、goodput、Jain 公平指数、预测器校准曲线。
- **风险**: 审稿人会说"S3 做过"——差异化点在**调度×路由联合**与误差鲁棒性。

## #3 KV 量化粒度/异常值处理研究

- **插槽**: `kv_cache/quant.py`（`KVQuantizer` 接口）+ 引擎 `kv_cache_dtype`
- **动机**: per-(row,head) 是最简单的正确基线；KIVI 的 per-channel K 需要转置
  存储，vLLM 的 fp8 需要 online calibration——粒度×校准×旋转（QuaRot 式）
  的三角权衡在统一框架下对比的工作不多。
- **做法**: 在本框架实现 3~4 种量化器（per-channel K 需要给 pool 加转置 K 池）、
  在 WikiText/Pile 上测 perplexity 增量 + 在引擎上测吞吐/显存节省。
- **Baseline**: float16（内置）、int8 per-row（内置）、fp8 per-row（内置）。
- **指标**: KV 显存字节、perplexity 增量、长上下文任务准确率、吞吐。
- **风险**: 这条线 KIVI/KVQuant/IntactKV 已经很卷，需要找新角度（如
  量化粒度与 prefix cache 复用的交互：共享块被不同后缀访问时的尺度适配）。

## #4 Chunked-Prefill 与 PD 分离的联合调度

- **插槽**: `engine/scheduler.py`（chunk 预算策略）+ `simulator/cluster.py`（PD）
- **动机**: Sarathi-Serve 证明混批好、DistServe 证明分离好——但"哪些请求该
  走分离、哪些该混批"的**混合部署调度**还是空白。
- **做法**: 在模拟器里同时放 mixed 副本与 PD 对，按请求特征（长度/时延 SLO）
  动态选池；调度器侧调整 chunk 预算在 decode/大 prefill 间的分配。
- **Baseline**: 全 mixed（内置）、全 PD（内置 `--pd`）。
- **指标**: goodput(TTFT SLO × TPOT SLO)、吞吐、副本利用率。
- **风险**: 需要把 PD 传输模型做细（分块流水传输），否则审稿人质疑保真度。

## #5 模拟器-实机 gap 校准（方法论/benchmark 论文）

- **插槽**: `simulator/model_cost.py::GpuSpec`（flops_eff/bw_eff 可拟合）+
  `bench/benchmark_serving.py`（提供实测数据）
- **动机**: 所有 serving 论文都有模拟器，但没人交代"模拟器误差多少、怎么校准"。
  做一套"在同一 trace 上跑模拟器与真机 vLLM/ServeLab，拟合有效算力/带宽/
  开销三参数，报告各指标的预测误差"就是一篇实打实的 benchmark 短文。
- **Baseline**: 未校准的 roofline 常数（内置 presets）。
- **数据集**: 自采（4090/A100 各一轮）+ 公开 trace。
- **指标**: TTFT/TPOT/吞吐的 MAPE、按负载强度的误差曲线。
- **风险**: 需要稳定的 GPU 环境；本机 4050(6G) 可以跑小模型起步。

## #6 MoE 服务：专家放置 / 动态容量

- **插槽**: `simulator/moe.py::greedy_rebalance`（替换成你的策略）
- **动机**: DeepSeek-V3 的 aux-loss-free 重排说明 expert balance 可以做得很轻；
  serving 侧的**在线放置 + 容量因子联动**仍有空间。
- **做法**: 用 token 级 trace 统计专家热度序列 → 预测下一窗口热度 →
  热度感知放置（对比内置 greedy 重平衡）；或动态容量因子 + 溢出 token 的
  二次路由。
- **Baseline**: 静态均匀放置、greedy 重平衡（内置）。
- **指标**: 负载不均衡因子、token 丢弃率、all2all 通信量、端到端 TPOT。
- **风险**: 需要真实 MoE 路由 trace（可以用开源小 MoE 模型自采）。

---

## 执行建议（如果目标是一篇会议短文/研讨会论文）

1. 第 1 周：跑通 `--compare-routers`，读懂 scheduler + radix + 模拟器三条线。
2. 第 2~3 周：在选中方向上实现你的策略 + 2 个消融变体。
3. 第 4 周：三个 trace × 三个 baseline 的对比表 + 图；写 evaluation。
4. 第 5~6 周：把策略移植回真实引擎（接口已对齐），小规模实机复现。
5. 写作：系统类 workshop（EuroSys/ATC/OSDI 短文、APSys、MLSys）起步。

> 提示：所有 baseline 命令都能在 `docs/benchmark_guide.md` 找到；
> 每个实验记得固定 `--seed`，trace 落盘成 CSV 保证可复现。
