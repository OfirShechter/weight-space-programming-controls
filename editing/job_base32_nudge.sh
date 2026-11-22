#!/usr/bin/env bash
# Base at the full 32k with the answer nudge: the like-for-like partner of "edit + nudge". Skips finished work.
#   1. base thinking at 32k, 500 x 4, every answer saved (the first 500 x 4 base run did not save them)
#   2. nudge the samples cut at 32k (budget_nudge.py think35)
# Usage: job_base32_nudge.sh <gpu_index> <root>
set -uo pipefail
GPU="$1"; ROOT="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH="$(dirname "$PY"):$PATH"
LOG="$ROOT/logs"; mkdir -p "$LOG"
step() {
  local name="$1"; shift
  echo "[$(date '+%F %T')] start $name" | tee -a "$LOG/job.log"
  if "$@" >"$LOG/$name.log" 2>&1; then
    echo "[$(date '+%F %T')] done  $name: $(grep -v -E 'INFO|WARNING|it/s|s/it|warn' "$LOG/$name.log" | tail -1 | cut -c1-300)" | tee -a "$LOG/job.log"
  else
    echo "[$(date '+%F %T')] FAIL  $name (see $LOG/$name.log)" | tee -a "$LOG/job.log"; return 1
  fi
}
step B_R0_think_b "$PY" "$HERE/eval_math.py" "$ROOT" R0_base think35 --subset data/math500_sub500.jsonl --n 4 --tag __full500x4b && \
step B_nudge32k "$PY" "$HERE/budget_nudge.py" "$ROOT" think35 --model R0_base --tag __full500x4b
echo "[$(date '+%F %T')] base32-nudge job finished" | tee -a "$LOG/job.log"
