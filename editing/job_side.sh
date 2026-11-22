#!/usr/bin/env bash
# Side effects outside math: does the released edit change general knowledge and reasoning? MMLU-Pro (TIGER-Lab),
# a fixed random 1,000 of its 12,032 test questions (seed 0), 10 options each, graded by the boxed option letter
# (grade.py). Same prompt instruction as the math runs (mathqa.build_prompt: reason step by step, answer in \boxed{}).
#   p6: base   p7: edit      each: thinking off (x2), 8k budget (+ nudge), 32k thinking (+ nudge on cut answers)
# Usage: job_side.sh <gpu_index> <root> <p6|p7>
set -uo pipefail
GPU="$1"; ROOT="$2"; PART="$3"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH="$(dirname "$PY"):$PATH"
# own compile cache per part: parallel jobs sharing one cache crashed vLLM at engine start
export VLLM_CACHE_ROOT="$ROOT/.cache/vllm_$PART${SIDE_SEED:+s$SIDE_SEED}" TORCHINDUCTOR_CACHE_DIR="$ROOT/.cache/inductor_$PART${SIDE_SEED:+s$SIDE_SEED}"
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
D="$ROOT/data/mmlupro_1000.jsonl"
if [ ! -f "$D" ]; then
  "$PY" - "$D" "$PART" <<'PYEOF'
import json, random, sys
from datasets import load_dataset
rows = list(load_dataset("TIGER-Lab/MMLU-Pro", split="test"))
idx = sorted(random.Random(0).sample(range(len(rows)), 1000))
L = "ABCDEFGHIJ"
with open(sys.argv[1] + ".tmp" + sys.argv[2], "w") as f:
    for i in idx:
        r = rows[i]
        opts = "\n".join(f"{L[k]}. {o}" for k, o in enumerate(r["options"]))
        f.write(json.dumps({"id": f"mmlupro_{r['question_id']}", "kind": "mc", "category": r["category"],
                            "problem": f"{r['question']}\n\nOptions:\n{opts}\n\nAnswer with the letter of the correct option.",
                            "answer": r["answer"]}) + "\n")
PYEOF
  mv "$D.tmp$PART" "$D"
fi
m=$([ "$PART" = p6 ] && echo R0_base || echo R1_astra_p50c)
# second sample: SIDE_SEED=1 draws fresh samples (EVAL_SEED) under tag __mmlupro_s1; thinking off already has 2
SFX=""; if [ -n "${SIDE_SEED:-}" ]; then export EVAL_SEED="$SIDE_SEED"; SFX="_s$SIDE_SEED"; fi
T="__mmlupro$SFX"
A=(--subset data/mmlupro_1000.jsonl --tag "$T")
nudge() { step "$1" "$PY" "$HERE/budget_nudge.py" "$ROOT" "$2" --model $m --tag "$T" --subset data/mmlupro_1000.jsonl; }
[ -z "$SFX" ] && step S_${m}_nothink "$PY" "$HERE/eval_math.py" "$ROOT" $m nothink35 "${A[@]}" --n 2
step S_${m}_budget8k$SFX "$PY" "$HERE/eval_math.py" "$ROOT" $m budget8k "${A[@]}" --n 1 && nudge S_${m}_nudge8k$SFX budget8k
step S_${m}_think$SFX "$PY" "$HERE/eval_math.py" "$ROOT" $m think35 "${A[@]}" --n 1 && nudge S_${m}_nudge32k$SFX think35
echo "[$(date '+%F %T')] side-$PART${SIDE_SEED:+-s$SIDE_SEED} job finished" | tee -a "$LOG/job.log"
