#!/usr/bin/env bash
# Install this repo on a single GPU server (phase 3 used an A800 40GB). Does not train and does not touch the held-out set.
# The pinned wheels are torch 2.13.0+cu130 (requirements-train.txt) and vLLM 0.31.0.
# If the driver CUDA is older than 13.0, stop. Do not install a different torch build.
# Before anything is installed it checks three things that cost time on the phase 3 server, and stops with the fix:
# not running as root, Python.h present for the system python3.12, bwrap able to create a user namespace.
# It never changes system configuration itself; the fixes that need root are printed for the operator.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
REV=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e

if [ "$(id -u)" -eq 0 ]; then
  cat >&2 <<'EOF'
running as root; stop.
The sandbox limits the process count with RLIMIT_NPROC, which root ignores, so the fork bomb tests fail.
Create an ordinary user and run everything as that user (root is only for apt and the AppArmor rule):
  adduser --disabled-password --gecos "" arl && su - arl
EOF
  exit 1
fi

if command -v python3.12 >/dev/null; then
  inc=$(python3.12 -c 'import sysconfig; print(sysconfig.get_config_var("INCLUDEPY") or "")')
  if [ ! -f "$inc/Python.h" ]; then
    cat >&2 <<EOF
python3.12 has no Python.h (looked in '$inc'); stop.
The venvs use this interpreter, and triton compiles a C helper against it at run time (GradScaler path).
Install the headers as root:
  apt-get install -y python3.12-dev
EOF
    exit 1
  fi
  echo "python3.12 headers ok: $inc/Python.h"
else
  echo "no system python3.12; uv will use its own build, which ships the headers"
fi

if ! command -v bwrap >/dev/null; then
  echo "bwrap is missing; stop. Install it as root: apt-get install -y bubblewrap" >&2
  exit 1
fi
if ! bwrap --die-with-parent --unshare-user --unshare-pid --unshare-net --ro-bind / / true 2>/dev/null; then
  cat >&2 <<EOF
bwrap cannot create a user namespace for $(id -un); stop.
On Ubuntu 23.10 and later AppArmor blocks unprivileged user namespaces unless a profile allows them.
Install the rule in this repo as root, then rerun this script:
  install -m 644 $ROOT/scripts/server/bwrap-userns.apparmor /etc/apparmor.d/bwrap-userns
  apparmor_parser -r /etc/apparmor.d/bwrap-userns
EOF
  exit 1
fi
echo "bwrap can create a user namespace"

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
