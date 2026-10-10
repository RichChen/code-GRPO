#!/usr/bin/env bash
# Download models one by one through the proxy; restart a download that writes nothing for STALL_S seconds.
#   scripts/fetch_models.sh <log-file> <hf-model-id>...
set -uo pipefail
LOG=$1; shift
HF=~/.venvs/vllm/bin/hf
STALL_S=180
export HTTPS_PROXY=${HTTPS_PROXY:-http://172.29.64.1:7890}
export HF_HUB_DISABLE_XET=1  # the xet transfer path stalls behind the proxy; plain HTTP resumes fine
for m in "$@"; do
  s=$(date +%s); tries=0
  while :; do
    tries=$((tries + 1))
    "$HF" download "$m" >> "$(dirname "$LOG")/download-$(basename "$m").log" 2>&1 &
    pid=$!
    last=-1; idle=0
    while kill -0 $pid 2> /dev/null; do
      sleep 15
      now=$(awk '/write_bytes/ && !/cancelled/ {print $2}' /proc/$pid/io 2> /dev/null || echo 0)
      if [ "$now" = "$last" ]; then idle=$((idle + 15)); else idle=0; last=$now; fi
      if [ $idle -ge $STALL_S ]; then kill $pid; echo "$m stalled, restarting (try $tries)" >> "$LOG"; break; fi
    done
    wait $pid; rc=$?
    [ $rc = 0 ] && break
    [ $tries -ge 10 ] && break
  done
  echo "$m exit=$rc $(( $(date +%s) - s ))s tries=$tries" >> "$LOG"
done
