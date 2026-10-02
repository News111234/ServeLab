#!/usr/bin/env bash
# 在 GPU 服务器上运行: bash scripts/server_probe.sh
# 目的: 一次摸清 4x5090 环境，为后续 benchmark 做准备
set -uo pipefail

echo "=== GPU ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>/dev/null \
  || echo "nvidia-smi 不可用"
nvcc --version 2>/dev/null | tail -1 || echo "(nvcc 不在 PATH，正常)"

echo; echo "=== Python ==="
which python3 python 2>/dev/null
python3 --version 2>/dev/null
python3 - <<'EOF' 2>/dev/null || echo "(默认 python3 无 torch，稍后用 venv 装)"
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
      "| devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("capability:", torch.cuda.get_device_capability(0), "(5090 应为 (12, 0))")
EOF

echo; echo "=== 搜索已下载的 LLaMA 模型 (config.json) ==="
find "$HOME" /data /data*/ /mnt/ /workspace /opt 2>/dev/null -maxdepth 5 -name config.json \
  | while read -r f; do
      t=$(grep -o '"model_type"[^,}]*' "$f" 2>/dev/null | head -1)
      case "$t" in *llama*) echo "$f  [$t]";; esac
    done | head -20
echo "(若没搜到，把模型路径告诉我即可)"

echo; echo "=== 磁盘 / 内存 ==="
df -h "$HOME" . 2>/dev/null | head -5
free -g 2>/dev/null | head -2

echo; echo "=== 外网连通性 (pip 用) ==="
curl -sI -m 5 https://pypi.org/simple/ 2>/dev/null | head -1 || echo "pypi 不可达(可能需要校内源)"
