#!/usr/bin/env bash
# The one preregistered held-out run of one model (docs/plan/HELDOUT-PREREG.zh-CN.md). Phase 3 step 5 only.
#   scripts/server/eval_heldout.sh <model: HF id or merged fp16 dir> <name: base | sft | grpo> [--resume]
# vLLM serves with --dtype half and the same arguments as merge_fp16.sh. Client: configs/sft/v1-eval.json with the
# model swapped (token client, temperature 0.8, top_p 1.0, top_k 20, verify_tokenize). 1500 tasks x K=4,
# environment phase1-env-v1. Refuses to start if records for <name> already exist, unless --resume continues the
# same interrupted run (finished episodes are kept, never re-sampled).
set -euo pipefail
MODEL=$1 NAME=$2 RESUME=${3:-}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
OUT=$ROOT/runs/heldout
VLLM=~/.venvs/vllm/bin/vllm
PY=~/.venvs/vllm/bin/python
REV=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
export PATH="$HOME/.venvs/vllm/bin:$PATH"
mkdir -p "$OUT"
case $NAME in base|sft|grpo) ;; *) echo "name must be base, sft or grpo" >&2; exit 2 ;; esac
if [ -s "$OUT/$NAME.jsonl" ] && [ "$RESUME" != "--resume" ]; then
  echo "$OUT/$NAME.jsonl exists: each model is evaluated once. Stop." >&2
  exit 2
fi

python3 - "$MODEL" "$OUT/$NAME-client.json" <<'PY'
import json, sys
cfg = json.load(open("configs/sft/v1-eval.json", encoding="utf-8"))
cfg["model"] = sys.argv[1]
assert (cfg["endpoint"], cfg["temperature"], cfg["top_p"], cfg["top_k"]) == ("tokens", 0.8, 1.0, 20), cfg
json.dump(cfg, open(sys.argv[2], "w"), indent=2)
print(cfg)
PY

REVARG=()
if [ "$MODEL" = "Qwen/Qwen3-1.7B" ]; then REVARG=(--revision "$REV"); fi
HF_HUB_OFFLINE=1 "$VLLM" serve "$MODEL" "${REVARG[@]}" --dtype half --max-model-len 8192 --enable-auto-tool-choice \
  --tool-call-parser hermes --host 127.0.0.1 --port 8000 --gpu-memory-utilization 0.85 --seed 0 \
  >> "$OUT/$NAME-serve.log" 2>&1 &
PID=$!
ready=0
for _ in $(seq 1 180); do
  if curl -s -m 2 http://127.0.0.1:8000/v1/models > /dev/null; then ready=1; break; fi
  if ! kill -0 $PID 2>/dev/null; then break; fi
  sleep 5
done

status=3
if [ $ready = 1 ]; then
  set +e
  PYTHONPATH=src HF_HUB_OFFLINE=1 "$PY" -m arl.rollout.calibrate run --split heldout --k 4 \
    --concurrency "${CONCURRENCY:-16}" --config "$OUT/$NAME-client.json" --runs "$OUT" --name "$NAME" \
    >> "$OUT/$NAME-run.log" 2>&1
  status=$?
  set -e
  tail -3 "$OUT/$NAME-run.log"
else
  echo "vLLM did not come up; see $OUT/$NAME-serve.log" >&2
fi

kill -INT $PID 2>/dev/null || true
for _ in $(seq 1 30); do kill -0 $PID 2>/dev/null || break; sleep 2; done
kill -9 $PID 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
exit $status
