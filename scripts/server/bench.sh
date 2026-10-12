#!/usr/bin/env bash
# On the 4090: probe sleep mode, probe whether training and vLLM can stay resident together, then run 3 GRPO
# steps and scale the step time to 200 steps. Records failures; does not change the learning rate or the algorithm.
# Not for the held-out set.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
OUT=$ROOT/runs/server-bench
PY=~/.venvs/train/bin/python
mkdir -p "$OUT"

probe() {
  local mode=$1 level=$2 dest=$3
  mkdir -p "$dest"
  set +e
  if [ -n "$level" ]; then
    PYTHONPATH=src HF_HUB_OFFLINE=1 "$PY" -m arl.rl.vllm_server probe --config configs/grpo/v1.json \
      --memory-mode "$mode" --sleep-level "$level" --out "$dest" > "$dest/stdout.txt" 2>&1
  else
    PYTHONPATH=src HF_HUB_OFFLINE=1 "$PY" -m arl.rl.vllm_server probe --config configs/grpo/v1.json \
      --memory-mode "$mode" --out "$dest" > "$dest/stdout.txt" 2>&1
  fi
  echo $? > "$dest/exit"
  set -e
  echo "probe $mode level=${level:-none} exit=$(cat "$dest/exit")"
}

# SKIP_PROBES=1 reuses sleep-l1/ and sleep-l2/ from an earlier run (their exit files pick the fallback mode).
if [ "${SKIP_PROBES:-0}" = 1 ] && [ -f "$OUT/sleep-l1/exit" ] && [ -f "$OUT/sleep-l2/exit" ]; then
  echo "probes skipped: sleep-l1 exit=$(cat "$OUT/sleep-l1/exit"), sleep-l2 exit=$(cat "$OUT/sleep-l2/exit")"
else
  probe sleep 1 "$OUT/sleep-l1"
  probe sleep 2 "$OUT/sleep-l2"
fi

write_cfg() {
  local mode=$1 steps=$2 dest=$3
  PYTHONPATH=src "$PY" - "$mode" "$steps" "$dest" <<'PY'
import json, sys
mode, steps, dest = sys.argv[1], int(sys.argv[2]), sys.argv[3]
json.dump({"base": "configs/grpo/v1.json", "overrides": {
    "steps": steps, "out_dir": dest, "eval": {"every": 0}, "vllm": {"memory_mode": mode},
}}, open(dest + "/bench-config.json", "w"), indent=2)
PY
}

run_steps() {
  local mode=$1 steps=$2 dest=$3
  mkdir -p "$dest"
  write_cfg "$mode" "$steps" "$dest"
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 30 > "$dest/gpu.csv" &
  local mon=$!
  set +e
  PYTHONPATH=src HF_HUB_OFFLINE=1 "$PY" -m arl.rl.grpo --config "$dest/bench-config.json" --steps "$steps" \
    > "$dest/stdout.txt" 2>&1
  echo $? > "$dest/exit"
  set -e
  kill $mon 2>/dev/null || true
  wait $mon 2>/dev/null || true
  echo "grpo $mode steps=$steps exit=$(cat "$dest/exit")"
}

# One step with both processes resident. A non-zero exit means that mode does not fit.
run_steps colocate 1 "$OUT/colocate-1"
mode=colocate
result=$OUT/colocate-1
if [ "$(cat "$OUT/colocate-1/exit")" != 0 ]; then
  if [ "$(cat "$OUT/sleep-l1/exit")" = 0 ] || [ "$(cat "$OUT/sleep-l2/exit")" = 0 ]; then
    mode=sleep
  else
    mode=restart
  fi
  echo "colocate did not fit; 3-step bench uses $mode"
  run_steps "$mode" 3 "$OUT/steps-3"
  result=$OUT/steps-3
else
  echo "colocate fits; continuing that run to 3 steps"
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 30 > "$OUT/colocate-1/gpu-rest.csv" &
  mon=$!
  set +e
  PYTHONPATH=src HF_HUB_OFFLINE=1 "$PY" -m arl.rl.grpo --config "$OUT/colocate-1/bench-config.json" \
    --resume --steps 3 > "$OUT/colocate-1/stdout-rest.txt" 2>&1
  echo $? > "$OUT/colocate-1/exit-rest"
  set -e
  kill $mon 2>/dev/null || true
  wait $mon 2>/dev/null || true
fi
PYTHONPATH=src "$PY" - "$result/log.jsonl" "$mode" <<'PY'
import json, statistics, sys
path, mode = sys.argv[1], sys.argv[2]
steps = []
with open(path, encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        if rec.get("type") == "step" and "step_s" in rec:
            steps.append(rec["step_s"])
if not steps:
    raise SystemExit("no finished steps to scale")
mean = statistics.fmean(steps)
print(json.dumps({
    "mode": mode,
    "steps": len(steps),
    "step_s": steps,
    "mean_step_s": round(mean, 1),
    "estimate_200_steps_hours": round(mean * 200 / 3600, 2),
}, indent=2))
PY
echo "bench done; see $OUT"
