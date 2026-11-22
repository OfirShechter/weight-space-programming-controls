#!/usr/bin/env bash
# Confirmation run: all 500 MATH-500 problems, 4 samples each, for the arms the post compares.
# Cheap thinking-off arms first, then the two thinking arms being compared, then the base (needed only for "% of gain").
# Usage: job_full.sh <gpu_index> <root>     (results: <root>/results/<arm>__<protocol>__full500x4.json)
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
    echo "[$(date '+%F %T')] done  $name: $(tail -1 "$LOG/$name.log" | cut -c1-300)" | tee -a "$LOG/job.log"
  else
    echo "[$(date '+%F %T')] FAIL  $name (see $LOG/$name.log)" | tee -a "$LOG/job.log"
  fi
}
[ -f "$ROOT/data/math500_sub500.jsonl" ] || "$PY" - "$ROOT" <<'PYEOF'
import json, sys
from datasets import load_dataset
rows = [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"]} for r in load_dataset("HuggingFaceH4/MATH-500", split="test")]
with open(f"{sys.argv[1]}/data/math500_sub500.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
PYEOF
E=(--subset data/math500_sub500.jsonl --n 4 --tag __full500x4)
step F_R0_nothink "$PY" "$HERE/eval_math.py" "$ROOT" R0_base nothink35 "${E[@]}"
step F_R1_nothink "$PY" "$HERE/eval_math.py" "$ROOT" R1_astra_p50c nothink35 "${E[@]}"
step F_R1_think   "$PY" "$HERE/eval_math.py" "$ROOT" R1_astra_p50c think35 "${E[@]}"
step F_B2L_think  "$PY" "$HERE/eval_math.py" "$ROOT" B2_L15_19 think35 "${E[@]}"
step F_R0_think   "$PY" "$HERE/eval_math.py" "$ROOT" R0_base think35 "${E[@]}"
SAVE="$HERE/saved/results"; mkdir -p "$SAVE"
cp "$ROOT"/results/*__full500x4.json "$ROOT"/results/*__full500x4.samples.jsonl "$SAVE/" 2>/dev/null
echo "[$(date '+%F %T')] full job finished" | tee -a "$LOG/job.log"
