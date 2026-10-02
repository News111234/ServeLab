# Trace 数据集放置说明

把下载的 trace 放在本目录（gitignore 已忽略大文件）：

| 文件 | 格式 | 下载 | 用法 |
|---|---|---|---|
| `sharegpt.json` | json | huggingface.co/datasets/anon8231489/sharegpt_vicuna | `--trace sharegpt.json` |
| `Azure_LLM_Inference_Trace_*.csv/parquet` | csv/parquet | github.com/Azure/AzurePublicDataset | `--trace <文件名>` |
| `BurstGPT_log_*.csv` | csv | github.com/ImingB/BurstGPT | `--trace <文件名>` |

没有网络/数据时用 `--trace synthetic`（内置 Poisson + lognormal + Zipf 前缀
生成器），参数见 `examples/run_simulator.py --help`。

字段要求：
- ShareGPT: `conversations[].from/value`（多轮对话 → 前缀复用组）
- Azure: `TIMESTAMP, ContextTokens, GeneratedTokens`（parquet 需 pandas+pyarrow）
- BurstGPT: `Request timestamp, Input tokens, Output tokens`
