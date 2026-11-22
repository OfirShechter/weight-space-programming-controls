#!/usr/bin/env bash
# Follow-up runs: budget forcing from the saved thinking, and B2 tuned at 32k. Every step skips finished work.
#   1. thinking budgets with the standard answer nudge, from the saved thinking (budget_nudge.py)
#   2. sweep stage C: the 5 stage-B configs rescored at 32k on practice problems -> models/B2_tuned32
#   3. B2_tuned32 on all 500 MATH-500 problems x 4
# Usage: job_more.sh <gpu_index> <root>
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
for b in 4k 8k 16k; do step M_nudge$b "$PY" "$HERE/budget_nudge.py" "$ROOT" budget$b; done
step M_sweep32 "$PY" "$HERE/sweep_b2.py" "$ROOT" --stage32 && \
step M_B2T32_think "$PY" "$HERE/eval_math.py" "$ROOT" B2_tuned32 think35 --subset data/math500_sub500.jsonl --n 4 --tag __full500x4
echo "[$(date '+%F %T')] more job finished" | tee -a "$LOG/job.log"
