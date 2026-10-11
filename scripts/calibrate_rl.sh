#!/usr/bin/env bash
# Phase 2C step 2: calibrate SFT-v1 on rl_train_ids with the token client (T=1.0, top_k=0), K=4.
#   scripts/calibrate_rl.sh pilot   # 50 tasks, stratified by bug source function
#   scripts/calibrate_rl.sh full    # all 1853 tasks; resumes, so the pilot's episodes are kept
# Serves the fp16 runs/sft-preview/merged with the diagnostic-scan vLLM settings, then stops the server.
set -uo pipefail
MODE=$1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/runs/grpo-calib
MODEL=runs/sft-preview/merged
CONFIG=configs/grpo/calibrate-v1.json
NAME=rl-train
VLLM=~/.venvs/vllm/bin/vllm
PY=~/.venvs/vllm/bin/python
mkdir -p "$OUT"
cd "$ROOT"
case $MODE in
  pilot) EXTRA=(--sample 50) ;;
  full) EXTRA=() ;;
  *) echo "usage: $0 pilot|full"; exit 2 ;;
esac

HF_HUB_OFFLINE=1 "$VLLM" serve "$MODEL" --dtype half --max-model-len 8192 --enable-auto-tool-choice \
  --tool-call-parser hermes --host 127.0.0.1 --port 8000 --gpu-memory-utilization 0.85 --seed 0 \
  >> "$OUT/serve.log" 2>&1 &
PID=$!
ready=0
for _ in $(seq 1 180); do
  if curl -s -m 2 http://127.0.0.1:8000/v1/models > /dev/null; then ready=1; break; fi
  if ! kill -0 $PID 2> /dev/null; then break; fi
  sleep 5
done

status=3
if [ $ready = 1 ]; then
  PYTHONPATH=src HF_HUB_OFFLINE=1 "$PY" -m arl.rollout.calibrate run --split train --ids data/tasks/rl_train_ids.txt \
    "${EXTRA[@]}" --k 4 --config "$CONFIG" --runs "$OUT" --name "$NAME" >> "$OUT/$MODE.log" 2>&1
  status=$?
  [ $MODE = pilot ] && cp "$OUT/$NAME-summary.json" "$OUT/pilot-summary.json"
else
  echo "[calibrate_rl] server did not come up; see $OUT/serve.log"
fi

kill -INT $PID 2> /dev/null
for _ in $(seq 1 30); do kill -0 $PID 2> /dev/null || break; sleep 2; done
kill -9 $PID 2> /dev/null
pkill -9 -f "VLLM::EngineCore" 2> /dev/null
exit $status
