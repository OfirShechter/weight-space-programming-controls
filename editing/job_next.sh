#!/usr/bin/env bash
# Follow-up runs, in priority order. Every step skips finished work, so the job can be re-run after an interruption.
#   1. overlap check: practice problems vs MATH-500 (CPU)
#   2. refit B2 to save every layer's direction (dirs_all.json)
#   3. budget-matched B2 sweep on practice problems (sweep_b2.py) -> models/B2_tuned
#   4. B2_tuned on all 500 MATH-500 problems x 4
#   5. thinking budgets 4k / 8k / 16k on 500 x 4 (accuracy vs tokens)
#   6. audit reruns with every answer saved (*.all.jsonl.gz): base thinking off, then the edit with thinking
# Usage: job_next.sh <gpu_index> <root>
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
    echo "[$(date '+%F %T')] done  $name: $(grep -v -E 'INFO|WARNING|it/s|s/it' "$LOG/$name.log" | tail -1 | cut -c1-300)" | tee -a "$LOG/job.log"
  else
    echo "[$(date '+%F %T')] FAIL  $name (see $LOG/$name.log)" | tee -a "$LOG/job.log"; return 1
  fi
}
E=(--subset data/math500_sub500.jsonl --n 4)
step N_overlap "$PY" "$HERE/check_overlap.py" "$ROOT"
[ -f "$ROOT/auto/dirs_all.json" ] || step N_refit "$PY" "$HERE/auto_direction.py" fit "$ROOT" || exit 1
step N_sweep "$PY" "$HERE/sweep_b2.py" "$ROOT" && \
step N_B2T_think "$PY" "$HERE/eval_math.py" "$ROOT" B2_tuned think35 "${E[@]}" --tag __full500x4
for b in 4k 8k 16k; do
  step N_R0_budget$b "$PY" "$HERE/eval_math.py" "$ROOT" R0_base budget$b "${E[@]}" --tag __full500x4
done
step N_R0_nothink_b "$PY" "$HERE/eval_math.py" "$ROOT" R0_base nothink35 "${E[@]}" --tag __full500x4b
step N_R1_think_b   "$PY" "$HERE/eval_math.py" "$ROOT" R1_astra_p50c think35 "${E[@]}" --tag __full500x4b
SAVE="$HERE/saved"; mkdir -p "$SAVE/results" "$SAVE/auto"
cp "$ROOT"/results/*.json "$ROOT"/results/*.samples.jsonl "$SAVE/results/" 2>/dev/null
cp "$ROOT"/auto/sweep.json "$ROOT"/auto/overlap.json "$ROOT"/auto/dirs_all.json "$SAVE/auto/" 2>/dev/null
echo "[$(date '+%F %T')] next job finished" | tee -a "$LOG/job.log"
