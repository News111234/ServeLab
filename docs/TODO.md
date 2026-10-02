# ServeLab 待办事项

> 优先级：P0 = 下一步就做（面试/开题前必须）；P1 = 论文主线；P2 = 工程完善（每条也是面试谈资）。
> 完成后勾掉并在 `docs/PROGRESS.md` 的变更记录里加一行。

## P0 环境与基础设施

- [x] ~~`pip install -e ".[all]"`~~（anaconda 的 Python 3.12 环境）
- [ ] 安装 CUDA 版 torch（本机 RTX 4050 6GB）：
      `pip install torch --index-url https://download.pytorch.org/whl/cu126`
      然后 `python -c "import torch; print(torch.cuda.is_available())"` 应为 True
- [x] ~~初始化 git 仓库并做首次提交~~（v0.2.0，commit 31a9163）
- [x] ~~上传 GitHub~~：https://github.com/News111234/ServeLab （main 分支）
- [ ] 下载真实小模型并跑通引擎：`python examples/run_engine.py --model <Qwen2.5-0.5B路径>`
- [ ] 下载 trace 数据放入 `traces/`（清单见 `traces/README.md`）：
      ShareGPT json、Azure LLM Inference Trace、BurstGPT csv
- [ ] 在 Linux + GPU 机器（实验室/Colab/AutoDL）跑 `pytest tests/test_triton_kernels.py -v`
      与 `python -m servelab.bench.kernel_bench`，验证 Triton kernel 对拍误差
- [ ] 远程方案（4×5090 服务器，经 Omnissa Horizon 跳板）：Horizon 客户端打开后由我接管 GUI，
      进入跳板机后转 SSH 操作；先跑 `bash scripts/server_probe.sh` 摸底
      （GPU/驱动/模型路径/网络），再跑 `bash scripts/gpu_env_setup.sh` 装 cu128 torch
      与 ServeLab（**5090 = Blackwell sm_120，必须 CUDA 12.8+ 的 torch ≥ 2.7**）

## P0 面试准备

- [ ] 按 `docs/interview_guide.md` 逐条过 JD，每条能脱稿讲 1 分钟
- [ ] 通读 `docs/architecture.md` + 对应源码，重点：kv_cache 三层、scheduler 三阶段、
      radix 锁与驱逐
- [ ] 白板练习：画 block table / slot 公式 `block_id*block_size+offset`；
      画 radix 树匹配与锁
- [ ] 三个现场 demo 彩排：`pytest tests -q`（62 passed, 2 skipped）、
      `python examples/run_simulator.py --compare-routers`、
      `python examples/moe_sim.py`
- [ ] 准备 2 个调试故事（素材见 `docs/PROGRESS.md` 的缺陷记录节）：
      decode 路径 `num_computed_tokens` 不变量 bug；模拟器重复事件级联

## P1 论文实验主线（对应 research_roadmap #1 / #2）

- [ ] 固定实验矩阵并把结果落盘 CSV（保证可复现，全部 `--seed 42`）：
      prefix cache on/off × chunked prefill on/off × 并发 {1,4,16,32}
- [ ] 路由对比基线表：`--compare-routers` × 3 个 trace × 请求率扫描
      {2,4,6,8,10}，画 goodput-负载曲线
- [ ] **方向 #1**：实现 workload-aware `EvictionPolicy` v2
      （价值函数 = 重用概率 × 重算成本）+ 准入控制（one-shot 长前缀不缓存），
      对比 LRU/LFU/CostAware 的命中率与 TTFT
- [ ] **方向 #2**：给 `PredictiveRouter` 加置信度加权，
      测四种预测器（oracle/constant/online-mean/prompt-reg）的
      预测误差 → 调度退化曲线
- [ ] **方向 #5（校准）**：用 `bench/benchmark_serving.py` 实测数据拟合
      `model_cost.py` 的 `flops_eff/bw_eff/launch_overhead`，
      报告模拟器对各指标的 MAPE
- [ ] 结果整理：每张表注明 trace、seed、参数；图用 matplotlib 统一风格

## P2 工程完善（每条 = 一个可讲的扩展点）

- [x] ~~SWAP 抢占模式~~：v0.2.0 已实现（`preemption_mode="swap"`，
      `kv_cache/swap.py`，进度零丢失 + 失败回退 recompute，e2e 测试覆盖）
- [x] ~~引擎接入 offload~~：v0.2.0 已实现驱逐归档 + 前缀未命中自动恢复
      （`kv_cache/offload.py` + `manager.py::maybe_match_prefix`，测试覆盖）
      —— 剩余：异步预取（prefetch on admission）作为论文扩展
- [x] ~~SWAP 模式与调度器集成~~（含换出序列只经 swap-in 通道恢复的不变量）
- [ ] OpenAI server 流式输出（SSE）：`engine.step()` 已逐 token 产出，补上即可
- [ ] radix 驱逐用叶子堆替换 `_find_node` 线性扫描（大缓存性能）
- [ ] CUDA Graph 捕获 decode 步（`EngineConfig.enforce_eager` 开关已预留）
- [ ] flash-decoding：Triton paged attention 加 split-K 两阶段归约（长上下文）
- [ ] per-channel K 量化器（KIVI 式，需转置 K 池）：roadmap #3
- [ ] parallel sampling / beam search（`Sequence.fork`）
- [ ] AWQ/GPTQ 权重加载（`models/loader.py` 扩展）
- [ ] 投机解码融合进 continuous batching（batched spec-decode，cf. vLLM V1）
- [ ] TP 引擎拆分独立 driver/worker 进程 + NCCL 后端（GPU 机器）
- [ ] GitHub Actions CI：CPU 环境跑 `pytest tests -q`

## 完成定义（DoD）

- 主线实验：每个结论 = ≥3 trace × ≥2 baseline × 固定 seed，图表脚本入 `scripts/`
- 面试：JD 六条 + 追问清单全部有代码锚点；demo 机器上跑一遍不翻车
