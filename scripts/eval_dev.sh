#!/usr/bin/env bash
# Phase 2A dev evaluation: serve a model with the diagnostic-scan vLLM settings, run the 100 dev tasks x 4, stop.
#   scripts/eval_dev.sh <hf-model-id | path> <run-name> <client-config.json>
# Unlike diag_model.sh, weights are never deleted.
set -uo pipefail
MODEL=$1 NAME=$2 CONFIG=$3
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/runs/sft-eval
VLLM=~/.venvs/vllm/bin/vllm
PARSER=hermes
GPU_UTIL=0.85
mkdir -p "$OUT"
cd "$ROOT"

CMD=(env HF_HUB_OFFLINE=1 "$VLLM" serve "$MODEL" --dtype half --max-model-len 8192 --enable-auto-tool-choice
     --tool-call-parser $PARSER --host 127.0.0.1 --port 8000 --gpu-memory-utilization $GPU_UTIL --seed 0)
echo "[eval] ${CMD[*]}"
t0=$(date +%s)
"${CMD[@]}" > "$OUT/$NAME-serve.log" 2>&1 &
PID=$!
ready=0
for _ in $(seq 1 180); do
  if curl -s -m 2 http://127.0.0.1:8000/v1/models > /dev/null; then ready=1; break; fi
  if ! kill -0 $PID 2> /dev/null; then break; fi
  sleep 5
done
startup=$(( $(date +%s) - t0 ))
gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
kv=$(grep -o "GPU KV cache size: [0-9,]* tokens" "$OUT/$NAME-serve.log" | head -1)
python3 - "$OUT/$NAME-serve.json" <<EOF
import json, sys
json.dump({"model": "$MODEL", "parser": "$PARSER", "gpu_memory_utilization": $GPU_UTIL, "command": "${CMD[*]}",
           "ready": bool($ready), "startup_s": $startup, "gpu_mem_used_mib": $gpu_mem, "kv_cache": "$kv",
           "vllm": "$($VLLM --version 2>/dev/null | tail -1)"}, open(sys.argv[1], "w"), indent=2)
EOF
cat "$OUT/$NAME-serve.json"

status=0
if [ $ready = 1 ]; then
  uv run python -m arl.rollout.calibrate run --split train --ids data/tasks/dev_ids.txt --k 4 --config "$CONFIG" \
    --runs "$OUT" --name "$NAME" > "$OUT/$NAME-run.log" 2>&1
  status=$?
  tail -3 "$OUT/$NAME-run.log"
else
  echo "[eval] server did not come up; see $OUT/$NAME-serve.log"
  status=3
fi

kill -INT $PID 2> /dev/null
for _ in $(seq 1 30); do kill -0 $PID 2> /dev/null || break; sleep 2; done
kill -9 $PID 2> /dev/null
pkill -9 -f "VLLM::EngineCore" 2> /dev/null
exit $status
