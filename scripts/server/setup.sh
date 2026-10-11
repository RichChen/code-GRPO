#!/usr/bin/env bash
# Install this repo on a single 4090. Does not train and does not touch the held-out set.
# The pinned wheels are torch 2.13.0+cu130 (requirements-train.txt) and vLLM 0.31.0.
# If the driver CUDA is older than 13.0, stop. Do not install a different torch build.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
REV=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e

if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi is missing; stop" >&2
  exit 1
fi
drv=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
if [ -z "$drv" ]; then
  echo "could not read the driver CUDA version; stop" >&2
  exit 1
fi
python3 - "$drv" <<'PY'
import sys
major, minor = (int(x) for x in sys.argv[1].split(".")[:2])
if (major, minor) < (13, 0):
    raise SystemExit(f"driver CUDA {sys.argv[1]} is older than 13.0; the pinned torch wheel is cu130. Stop.")
print(f"driver CUDA {sys.argv[1]} >= 13.0")
PY

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --frozen

uv venv ~/.venvs/train --python 3.12
uv pip install --python ~/.venvs/train/bin/python -r requirements-train.txt
~/.venvs/train/bin/python - <<'PY'
import torch
print("train torch", torch.__version__, "cuda", torch.version.cuda)
if not torch.__version__.startswith("2.13.0"):
    raise SystemExit(f"train torch is {torch.__version__}, expected 2.13.0. Stop.")
PY

uv venv ~/.venvs/vllm --python 3.12
uv pip install --python ~/.venvs/vllm/bin/python "vllm==0.31.0"
uv pip install --python ~/.venvs/vllm/bin/python "pytest==9.1.1"
~/.venvs/vllm/bin/python - <<'PY'
import torch
import vllm
print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda)
if vllm.__version__ != "0.31.0":
    raise SystemExit(f"vLLM is {vllm.__version__}, expected 0.31.0. Stop.")
PY

export HF_HUB_DISABLE_XET=1
# HF_ENDPOINT, when the operator has set it, is read by huggingface_hub. This script does not pick a mirror.
~/.venvs/vllm/bin/hf download Qwen/Qwen3-1.7B --revision "$REV"
echo "setup ok; base revision $REV"
