# Benchmark 与 Profiling 实操指南

## 1. 引擎端到端 bench（实机）

```bash
# 固定并发（closed loop）
python -m servelab.bench.benchmark_serving --model /path/to/Qwen2.5-0.5B-Instruct \
    --num-prompts 64 --concurrency 8 --output-tokens 64

# 泊松到达（open loop，真实节奏）
python -m servelab.bench.benchmark_serving --model ... --load-pattern poisson \
    --request-rate 4 --num-prompts 128
```

标准实验矩阵（论文/面试都够用）：
- prefix cache on/off × chunked prefill on/off（4 组）
- 并发扫描 {1, 4, 16, 32} 画吞吐-时延曲线
- `--kv-cache-dtype int8/fp8` 对照 float16：显存（`engine.stats()` 里的
  num_blocks 等效 token 数）与时延的权衡

## 2. Kernel 微基准（Linux + GPU）

```bash
python -m servelab.bench.kernel_bench
```
输出 torch 参考实现 vs Triton kernel 的耗时与对拍误差。
改 kernel 后先看误差（应 < 1e-2 fp16），再看耗时。

## 3. Profiling

```python
# engine 层：把 step 包进 torch.profiler
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(20):
        engine.step()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("trace.json")   # chrome://tracing 打开
```

Nsight（Linux）：
```bash
nsys profile -o servelab python -m servelab.bench.benchmark_serving --model ... --num-prompts 16
nsys stats servelab.nsys-rep                    # 看每 kernel 耗时与 gap
ncu --set full -kregex "paged|_rmsnorm" python -m servelab.bench.kernel_bench
```
看什么：Nsight Systems 时间线上的 kernel 间 gap（launch 瓶颈 → CUDA Graph）；
Nsight Compute 里 SOL 指标：SM% 低 + DRAM% 高 = 访存受限（decode attention
的正常形态），SM% 高 = 计算受限（大 batch prefill GEMM）。

## 4. 模拟器实验（无 GPU 也能跑，论文主战场）

```bash
# 一键对比四种路由（固定 seed 可复现）
python examples/run_simulator.py --compare-routers --num-requests 500 --seed 42

# 扫描请求率画 goodput-负载曲线（负载过载点的对比是论文核心图）
for r in 2 4 6 8 10; do
  python examples/run_simulator.py --request-rate $r --replicas 4 \
      --router prefix-affinity | tail -8
done

# 真实 trace
#  ShareGPT:      https://huggingface.co/datasets/anon8231489/sharegpt_vicuna
#  Azure trace:   https://github.com/Azure/AzurePublicDataset (LLM Inference Trace)
#  BurstGPT:      https://github.com/ImingB/BurstGPT (csv log)
python examples/run_simulator.py --trace Azure_LLM_Inference_Trace_2024.parquet
```

结果落盘：`compute_metrics(...).to_dict()` 是纯 dict，直接 `csv.DictWriter`
或 pandas 存表，保证实验可复现（脚本化在 `examples/run_simulator.py` 基础上
加个循环即可）。

## 5. 与 vLLM/SGLang 实机对拍（把结论做扎实）

- 同一张卡、同一个模型（如 Qwen2.5-7B-Instruct）、同一份 prompt 采样
  （用 `bench/datasets.py` 生成后落盘，两个框架喂同一份）。
- 口径对齐：vLLM `benchmark_serving.py` 的 TTFT/TPOT 定义与
  `servelab/bench/benchmark_serving.py` 一致，可直接对比。
- 典型结论表述："ServeLab 参考实现的绝对吞吐为 vLLM 的 X%（预期内，
  参考实现），但策略趋势一致：prefix cache 开启后 TTFT p50 下降 Y%，
  与模拟器预测误差 Z%。" —— 这正是 roadmap #5 的校准故事。

## 6. 模拟器批量实验与成本模型校准（v0.2.0 新增）

```bash
# 网格扫描：4 种路由 × 3 个请求率 × 2 种副本数 → results/sweep.csv
python -m servelab.simulator.sweep --out results/routing_sweep.csv     --num-requests 500 --seed 42

# 换 trace 只改 trace 参数源（见 examples/run_simulator.py 的 load_trace）
```

CSV 每行一次仿真（固定 seed 可复现），字段含 TTFT/TPOT 分位数、goodput、
Jain 指数、cache hit、preemptions——直接可画 goodput-负载曲线与路由对比图。

成本模型校准（roadmap #5 的工具）：

```python
from servelab.simulator.calibration import CostCalibrator, Measurement
from servelab.simulator.model_cost import MODEL_PRESETS

cal = CostCalibrator(MODEL_PRESETS["qwen2.5-7b"])
# measurement 来源：benchmark_serving 实测或逐 step 计时落盘的 CSV
# cal.add(Measurement(prefill_tokens, decode_seqs, kv_tokens), measured_seconds)
fit = cal.fit()
print(fit)                      # flops_eff / bw_eff / launch_overhead_ms / mape
sim_gpu = fit.gpu_spec()        # 直接喂给 SimConfig.gpu
```

拟合是三参数交替最小二乘（regime 归类 → 各自最小二乘 → 中位数残差更新
overhead）。用真实测量时，务必让样本同时覆盖 prefill 主导与 decode 主导
两类 batch，否则对应 regime 的常数不可辨识。
