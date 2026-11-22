#!/usr/bin/env bash
# Fairness check: give the released edit the same answer nudge the base got. Every step skips finished work.
#   1. edit at 32k (the saved rerun, every answer kept): nudge the 31% of samples cut at 32k
#   2. edit with 4k and 8k thinking budgets (new thinking, 500 x 4), then the nudge
# Usage: job_edit_nudge.sh <gpu_index> <root>
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
E=(--subset data/math500_sub500.jsonl --n 4 --tag __full500x4)
step E_nudge32k "$PY" "$HERE/budget_nudge.py" "$ROOT" think35 --model R1_astra_p50c --tag __full500x4b
for b in 4k 8k; do
  step E_R1_budget$b "$PY" "$HERE/eval_math.py" "$ROOT" R1_astra_p50c budget$b "${E[@]}" && \
  step E_nudge$b "$PY" "$HERE/budget_nudge.py" "$ROOT" budget$b --model R1_astra_p50c
done
echo "[$(date '+%F %T')] edit-nudge job finished" | tee -a "$LOG/job.log"
