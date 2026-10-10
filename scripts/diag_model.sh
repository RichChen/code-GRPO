#!/usr/bin/env bash
# Diagnostic scan for one local model: serve it with vLLM, run the 100 pilot tasks x 4, stop, delete weights.
#   scripts/diag_model.sh <hf-model-id> <run-name> <client-config.json>
set -uo pipefail
MODEL=$1 NAME=$2 CONFIG=$3
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/runs/diag
VLLM=~/.venvs/vllm/bin/vllm
PARSER=hermes
GPU_UTIL=0.85  # Windows keeps ~0.7 GiB of the 11 GiB, so 0.90 does not fit
CACHE=~/.cache/huggingface/hub/models--${MODEL//\//--}
mkdir -p "$OUT"
cd "$ROOT"

snap=$(ls -d "$CACHE"/snapshots/* 2>/dev/null | head -1)
if [ -z "$snap" ]; then echo "[diag] $MODEL not downloaded"; exit 2; fi
template_has_tool_call=$(cat "$snap"/tokenizer_config.json "$snap"/chat_template.jinja 2>/dev/null | grep -c "<tool_call>")

CMD=(env HF_HUB_OFFLINE=1 "$VLLM" serve "$MODEL" --dtype half --max-model-len 8192 --enable-auto-tool-choice
     --tool-call-parser $PARSER --host 127.0.0.1 --port 8000 --gpu-memory-utilization $GPU_UTIL --seed 0)
echo "[diag] ${CMD[*]}"
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
json.dump({"model": "$MODEL", "revision": "$(basename "$snap")", "parser": "$PARSER", "gpu_memory_utilization": $GPU_UTIL,
           "command": "${CMD[*]}", "ready": bool($ready), "startup_s": $startup, "gpu_mem_used_mib": $gpu_mem,
           "kv_cache": "$kv", "template_has_tool_call": $template_has_tool_call > 0,
           "vllm": "$($VLLM --version 2>/dev/null | tail -1)"}, open(sys.argv[1], "w"), indent=2)
EOF
cat "$OUT/$NAME-serve.json"

status=0
if [ $ready = 1 ]; then
  uv run python -m arl.rollout.calibrate pilot --config "$CONFIG" --runs "$OUT" --name "$NAME" > "$OUT/$NAME-run.log" 2>&1
  status=$?
  tail -3 "$OUT/$NAME-run.log"
else
  echo "[diag] server did not come up; see $OUT/$NAME-serve.log"
  status=3
fi

kill -INT $PID 2> /dev/null
for _ in $(seq 1 30); do kill -0 $PID 2> /dev/null || break; sleep 2; done
kill -9 $PID 2> /dev/null
pkill -9 -f "VLLM::EngineCore" 2> /dev/null
if [ $status = 0 ]; then
  # large files live in the shared hub/blobs store; the model dir only links to them
  find "$CACHE" -type l -print0 | while IFS= read -r -d '' link; do
    target=$(readlink -f "$link")
    while [ -L "$target" ]; do target=$(readlink -f "$target"); done
    case "$target" in */hub/blobs/*) rm -f "$target" "$target.refs" "$target.lock" ;; esac
  done
  find ~/.cache/huggingface/hub/blobs -mindepth 1 -type d -empty -delete 2> /dev/null
  rm -rf "$CACHE"
  echo "[diag] $NAME done, weights deleted: $CACHE"
else
  echo "[diag] $NAME failed (status $status), weights kept: $CACHE"
fi
exit $status
