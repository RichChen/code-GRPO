#!/usr/bin/env bash
# Phase 3 step 5: the three preregistered held-out runs, one after another, each once.
cd ~/work/agentic-rl
T=runs/phase3/heldout.time
for spec in "Qwen/Qwen3-1.7B base" "runs/sft-v1/merged-fp16 sft" "runs/grpo-v1/merged-fp16-step-150 grpo"; do
  set -- $spec
  echo "$2 start $(date '+%F %T')" >> $T
  bash scripts/server/eval_heldout.sh "$1" "$2"
  rc=$?
  echo "$2 exit=$rc end $(date '+%F %T') gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> $T
  [ $rc = 0 ] || { echo "chain stopped at $2" >> $T; exit $rc; }
done
echo "all done $(date '+%F %T')" >> $T
