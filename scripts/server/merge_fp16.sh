#!/usr/bin/env bash
# Merge SFT-v1 into the base loaded as fp16, then score it on the dev set with the token client.
# vLLM serves with --dtype half. Strict solve rate must fall inside the audited 95% interval of 53.5%,
# [0.455, 0.6125]. Otherwise stop.
# Does not evaluate the held-out set. Does not use runs/sft-preview/merged.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
OUT=$ROOT/runs/sft-v1/merged-fp16
EVAL=$ROOT/runs/sft-eval
NAME=dev-sft-fp16
VLLM=~/.venvs/vllm/bin/vllm
PY=~/.venvs/vllm/bin/python
mkdir -p "$EVAL"

PYTHONPATH=src HF_HUB_OFFLINE=1 ~/.venvs/train/bin/python -m arl.sft.merge_fp16 --out "$OUT"

python3 - "$OUT" "$EVAL/fp16-client.json" <<'PY'
import json, sys
cfg = json.load(open("configs/sft/v1-eval.json", encoding="utf-8"))
cfg["model"] = sys.argv[1]
json.dump(cfg, open(sys.argv[2], "w"), indent=2)
print(cfg)
PY

HF_HUB_OFFLINE=1 "$VLLM" serve "$OUT" --dtype half --max-model-len 8192 --enable-auto-tool-choice \
  --tool-call-parser hermes --host 127.0.0.1 --port 8000 --gpu-memory-utilization 0.85 --seed 0 \
  > "$EVAL/$NAME-serve.log" 2>&1 &
PID=$!
ready=0
for _ in $(seq 1 180); do
  if curl -s -m 2 http://127.0.0.1:8000/v1/models > /dev/null; then ready=1; break; fi
  if ! kill -0 $PID 2>/dev/null; then break; fi
  sleep 5
done

status=3
if [ $ready = 1 ]; then
  PYTHONPATH=src HF_HUB_OFFLINE=1 "$PY" -m arl.rollout.calibrate run --split train \
    --ids data/tasks/dev_ids.txt --k 4 --config "$EVAL/fp16-client.json" --runs "$EVAL" --name "$NAME" \
    > "$EVAL/$NAME-run.log" 2>&1
  status=$?
  tail -5 "$EVAL/$NAME-run.log"
else
  echo "vLLM did not come up; see $EVAL/$NAME-serve.log" >&2
fi

kill -INT $PID 2>/dev/null || true
for _ in $(seq 1 30); do kill -0 $PID 2>/dev/null || break; sleep 2; done
kill -9 $PID 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true

if [ $status != 0 ]; then
  exit $status
fi
PYTHONPATH=src "$PY" - "$EVAL/$NAME-summary.json" <<'PY'
import json, sys
from arl.sft.evaluate import SFT_DEV_STRICT_CI, in_sft_dev_interval
summary = json.load(open(sys.argv[1], encoding="utf-8"))
rate = summary["solve_rate"]
print(f"strict solve rate {rate}, interval {SFT_DEV_STRICT_CI}")
if not in_sft_dev_interval(rate):
    raise SystemExit(4)
PY
