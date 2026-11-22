#!/usr/bin/env bash
# First GPU job: the 100-problem MATH-500 subset (base, released edit, decoding baselines, B1 and B2 controls).
# Every step skips work that is already done, so the job can be re-run after an interruption.
# Usage: job.sh <gpu_index> <root>
set -uo pipefail
GPU="$1"; ROOT="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python}"   # override per machine: PY=/path/to/env/bin/python bash job.sh ...
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false
# FlashInfer's top-k/top-p sampler is JIT-compiled on first use, and that build can fail when the system ninja does not
# match the environment; vLLM's PyTorch sampler gives the same sampling distribution. Env tools first on PATH.
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH="$(dirname "$PY"):$PATH"
LOG="$ROOT/logs"; mkdir -p "$LOG"
step() {  # step <name> <cmd...>
  local name="$1"; shift
  echo "[$(date '+%F %T')] start $name" | tee -a "$LOG/job.log"
  if "$@" >"$LOG/$name.log" 2>&1; then
    echo "[$(date '+%F %T')] done  $name: $(tail -1 "$LOG/$name.log" | cut -c1-300)" | tee -a "$LOG/job.log"
  else
    echo "[$(date '+%F %T')] FAIL  $name (see $LOG/$name.log)" | tee -a "$LOG/job.log"; return 1
  fi
}
# 0. does vLLM run on this driver (CUDA 12.6) with the cu129 build? 3 problems, short budget.
step smoke "$PY" "$HERE/eval_math.py" "$ROOT" R0_base nothink35 --max-problems 3 || exit 2
# 1. reproduce the post's setting on the subset, then the cheap decoding-only baselines
step R0_think    "$PY" "$HERE/eval_math.py" "$ROOT" R0_base think35
step R1_think    "$PY" "$HERE/eval_math.py" "$ROOT" R1_astra_p50c think35
step R0_nothink  "$PY" "$HERE/eval_math.py" "$ROOT" R0_base nothink35
step R1_nothink  "$PY" "$HERE/eval_math.py" "$ROOT" R1_astra_p50c nothink35
step R0_budget   "$PY" "$HERE/eval_math.py" "$ROOT" R0_base budget4k
# 2. random direction with Astra's columns and alpha
step B1_think    "$PY" "$HERE/eval_math.py" "$ROOT" B1_random_dir think35
# 3. automated difference-of-means direction, no LLM involved
step B2_gen      "$PY" "$HERE/auto_direction.py" gen "$ROOT" --n 160 && \
step B2_fit      "$PY" "$HERE/auto_direction.py" fit "$ROOT" && {
  step B2_auto_think "$PY" "$HERE/eval_math.py" "$ROOT" B2_auto think35
  step B2_L_think    "$PY" "$HERE/eval_math.py" "$ROOT" B2_L15_19 think35
}
step summary "$PY" "$HERE/summarize.py" "$ROOT"
# save everything needed to inspect or rebuild the results into the repo (not the 1.6 GB model files:
# those are pinned downloads plus the saved edits in astra_edit.json and directions.json)
SAVE="$HERE/saved"; mkdir -p "$SAVE/results" "$SAVE/auto" "$SAVE/logs"
cp "$ROOT"/results/*.json "$ROOT"/results/*.samples.jsonl "$SAVE/results/" 2>/dev/null
cp "$ROOT"/auto/fit.json "$ROOT"/auto/directions.json "$SAVE/auto/" 2>/dev/null
[ -f "$ROOT/auto/gen.jsonl" ] && gzip -c "$ROOT/auto/gen.jsonl" > "$SAVE/auto/gen.jsonl.gz"
cp "$LOG"/*.log "$SAVE/logs/" 2>/dev/null
for m in "$ROOT"/models/*/; do n=$(basename "$m"); [ -f "$m/variant.json" ] && cp "$m/variant.json" "$SAVE/auto/${n}_variant.json"; done
echo "[$(date '+%F %T')] job finished; results saved to $SAVE" | tee -a "$LOG/job.log"
