#!/usr/bin/env bash
# The rest of the budget curve on MATH-500, and GSM8K. Parts can run in parallel on separate GPUs; every step skips
# finished work.
#   p1: MATH-500 edit at 16k + nudge (completes the budget curve), then GSM8K thinking off and 8k budgets (+ nudge)
#   p2: GSM8K base with full thinking (32k), then the nudge on its cut answers
#   p3: GSM8K edit with full thinking (32k), then the nudge on its cut answers
#   p4 / p5: GSM8K base / edit at a 16k budget, then the nudge
# GSM8K follows the authors' evaluate.py: all 1,319 test problems, 2 samples each, same prompt and grader.
# Usage: job_paper.sh <gpu_index> <root> <p1|p2|p3|p4|p5>
set -uo pipefail
GPU="$1"; ROOT="$2"; PART="$3"
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
# GSM8K test set, exactly as the authors' evaluate.py builds it (ids gsm8k_test_<i>, answer after "####", no commas)
G="$ROOT/data/gsm8k_test.jsonl"
if [ ! -f "$G" ]; then
  "$PY" - "$G" <<'PYEOF'
import json, sys
from datasets import load_dataset
rows = load_dataset("openai/gsm8k", "main", split="test")
with open(sys.argv[1] + ".tmp", "w") as f:
    for i, r in enumerate(rows):
        f.write(json.dumps({"id": f"gsm8k_test_{i}", "problem": r["question"],
                            "answer": r["answer"].split("####")[-1].strip().replace(",", "")}) + "\n")
PYEOF
  mv "$G.tmp" "$G"
fi
M=(--subset data/math500_sub500.jsonl --n 4 --tag __full500x4)
S=(--subset data/gsm8k_test.jsonl --n 2 --tag __gsm8k)
case "$PART" in
  p1)
    step P_R1_budget16k "$PY" "$HERE/eval_math.py" "$ROOT" R1_astra_p50c budget16k "${M[@]}" && \
    step P_R1_nudge16k "$PY" "$HERE/budget_nudge.py" "$ROOT" budget16k --model R1_astra_p50c
    step P_G_R0_nothink "$PY" "$HERE/eval_math.py" "$ROOT" R0_base nothink35 "${S[@]}"
    for m in R0_base R1_astra_p50c; do
      step P_G_${m}_budget8k "$PY" "$HERE/eval_math.py" "$ROOT" $m budget8k "${S[@]}" && \
      step P_G_${m}_nudge8k "$PY" "$HERE/budget_nudge.py" "$ROOT" budget8k --model $m --tag __gsm8k --subset data/gsm8k_test.jsonl
    done ;;
  p2|p3)
    m=$([ "$PART" = p2 ] && echo R0_base || echo R1_astra_p50c)
    step P_G_${m}_think "$PY" "$HERE/eval_math.py" "$ROOT" $m think35 "${S[@]}" && \
    step P_G_${m}_nudge32k "$PY" "$HERE/budget_nudge.py" "$ROOT" think35 --model $m --tag __gsm8k --subset data/gsm8k_test.jsonl ;;
  p4|p5)  # GSM8K at a 16k budget (+ nudge), between the 8k and 32k points; p4 base, p5 edit
    m=$([ "$PART" = p4 ] && echo R0_base || echo R1_astra_p50c)
    export EVAL_EAGER=1
    step P_G_${m}_budget16k "$PY" "$HERE/eval_math.py" "$ROOT" $m budget16k "${S[@]}" && \
    step P_G_${m}_nudge16k "$PY" "$HERE/budget_nudge.py" "$ROOT" budget16k --model $m --tag __gsm8k --subset data/gsm8k_test.jsonl ;;
esac
echo "[$(date '+%F %T')] paper-$PART job finished" | tee -a "$LOG/job.log"
