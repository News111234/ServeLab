#!/usr/bin/env bash
# 在 GPU 服务器上运行: bash scripts/gpu_env_setup.sh
# 5090 是 Blackwell (sm_120)，必须用 CUDA 12.8+ 构建的 torch (torch >= 2.7)
set -e
cd "$(dirname "$0")/.."        # 进入 ServeLab 根目录

python3 -m venv .venv-gpu
source .venv-gpu/bin/activate
pip install -U pip wheel

echo "== 安装 Blackwell 版 torch (cu128) =="
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, \
    'capability', torch.cuda.get_device_capability(0))"

echo "== 安装 ServeLab =="
pip install -e ".[engine,tokenizer,dev]"

echo "== 跑测试 (CPU 部分直接全绿; GPU 部分跑 Triton 对拍) =="
pytest tests -q

echo "== 30 秒冒烟: LLaMA 真模型 + 引擎 (把 MODEL_DIR 换成 probe 脚本搜到的路径) =="
# MODEL_DIR=/path/to/llama3-8b python examples/run_engine.py --model $MODEL_DIR --max-tokens 16

echo "完成。长跑实验用: tmux new -s bench  (断线不丢)"
