#!/usr/bin/env bash
# Fail closed. bwrap must work; do not fall back to an unisolated grader.
# The 13 bwrap checks are the bwrap case of the nine shared sandbox tests plus the four bwrap-only tests
# in tests/test_sandbox.py. A skip means one of them did not run.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

uv run python - <<'PY'
from arl.env.sandbox import bwrap_available
if not bwrap_available():
    raise SystemExit("bwrap cannot create a sandbox here. Stop. Do not fall back to rlimit_only.")
print("bwrap ok")
PY

out=$(uv run pytest -q tests/test_sandbox.py)
printf '%s\n' "$out"
if printf '%s\n' "$out" | grep -q skipped; then
  echo "sandbox tests skipped a case; the bwrap checks did not all run. Stop." >&2
  exit 1
fi

uv run pytest -q
PYTHONPATH=src ~/.venvs/vllm/bin/python -m pytest -q
PYTHONPATH=src ~/.venvs/train/bin/python -m pytest -q tests/train tests/rl

~/.venvs/train/bin/python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Stop.")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("this GPU does not support bf16. Stop.")
props = torch.cuda.get_device_properties(0)
print(f"gpu {props.name} {props.total_memory / 2**30:.1f} GiB bf16 yes")
PY
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "selfcheck ok"
