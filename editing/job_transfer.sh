#!/usr/bin/env bash
# Transfer study: does the automated recipe (B2: difference-of-means loop direction, budget-matched search) beat
# budget forcing on other loop-prone models? No LLM edit exists for them. Every step skips finished work.
#   screen : set up the root, then base with thinking (32k) on 100 MATH-500 problems x 1 -> how often it loops
#   full   : B2 fit + the same 50-try search as on Qwen3.5-0.8B (sweep_b2.py, stages A/B at 16k, C at 32k), then
#            500 MATH-500 problems x 2 samples: base and B2 at 32k (+ nudge on cut answers), base at 8k / 16k
#            budgets (+ nudge), B2 at 8k / 16k budgets (+ nudge), base thinking off
# Parallel stages (each skips finished work; run in this order, jobs within a stage in parallel):
#   sweepA (SHARD=i/n) + base (ARM=think|8k|16k|nothink)  ->  sweepB (SHARD=i/5)  ->  select  ->  sweepC (SHARD=i/5)
#   ->  select32  ->  b2 (ARM=think|8k|16k)
# Usage: job_transfer.sh <gpu_index> <model_short_name> <screen|full|sweepA|sweepB|select|sweepC|select32|base|b2>
set -uo pipefail
GPU="$1"; M="$2"; STAGE="$3"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python}"
MAIN="${WSP:?set WSP to the working root}"
ROOT="${TRANSFER_BASE:-$(dirname "$MAIN")/wsp_transfer}/$M"
case "$M" in   # pinned revisions (checked 2026-11-15)
  qwen35_2b)  REPO=Qwen/Qwen3.5-2B; REV=15852e8c16360a2fea060d615a32b45270f8a8fc; NOTHINK=1 ;;
  qwen35_4b)  REPO=Qwen/Qwen3.5-4B; REV=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a; NOTHINK=1 ;;  # checked 2026-11-16
  qwen35_9b)  REPO=Qwen/Qwen3.5-9B; REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a; NOTHINK=1 ;;
  qwen3_06b)  REPO=Qwen/Qwen3-0.6B; REV=c1899de289a04d12100db370d81485cdf75e47ca; NOTHINK=1 ;;
  qwen3_17b)  REPO=Qwen/Qwen3-1.7B; REV=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e; NOTHINK=1 ;;
  r1d_15b)    REPO=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B; REV=ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562; NOTHINK=0 ;;
  *) echo "unknown model $M"; exit 2 ;;
esac
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH="$(dirname "$PY"):$PATH"
# a compile cache per model root and parallel part: parallel jobs sharing one cache crashed vLLM at engine start
SHARD="${SHARD:-}"; ARM="${ARM:-}"; PART="$STAGE${SHARD:+_${SHARD%/*}}${ARM:+_$ARM}"
export VLLM_CACHE_ROOT="$ROOT/.cache/vllm_$PART" TORCHINDUCTOR_CACHE_DIR="$ROOT/.cache/inductor_$PART"
mkdir -p "$ROOT/logs"; LOG="$ROOT/logs"
step() {
  local name="$1"; shift
  echo "[$(date '+%F %T')] $M start $name" | tee -a "$LOG/job.log" >> "$MAIN/logs/transfer.log"
  if "$@" >"$LOG/$name.log" 2>&1; then
    echo "[$(date '+%F %T')] $M done  $name: $(grep -v -E 'INFO|WARNING|it/s|s/it|warn' "$LOG/$name.log" | tail -1 | cut -c1-300)" | tee -a "$LOG/job.log" >> "$MAIN/logs/transfer.log"
  else
    echo "[$(date '+%F %T')] $M FAIL  $name (see $LOG/$name.log)" | tee -a "$LOG/job.log" >> "$MAIN/logs/transfer.log"; return 1
  fi
}
step setup "$PY" "$HERE/transfer_setup.py" "$MAIN" "$ROOT" "$REPO" "$REV" || exit 1
E="$HERE/eval_math.py"; N="$HERE/budget_nudge.py"
T=(--subset data/math500_sub500.jsonl --n 2 --tag __t500x2)
nudge() { step "$1" "$PY" "$N" "$ROOT" "$2" --model "$3" --tag __t500x2 --subset data/math500_sub500.jsonl; }
case "$STAGE" in
  screen)
    step screen "$PY" "$E" "$ROOT" R0_base think35 --subset data/math500_sub100.jsonl --n 1 --tag __screen ;;
  full)
    step gen "$PY" "$HERE/auto_direction.py" gen "$ROOT" && \
    step fit "$PY" "$HERE/auto_direction.py" fit "$ROOT" && \
    step sweep "$PY" "$HERE/sweep_b2.py" "$ROOT" && \
    step sweep32 "$PY" "$HERE/sweep_b2.py" "$ROOT" --stage32 || exit 1
    step R0_think "$PY" "$E" "$ROOT" R0_base think35 "${T[@]}" && nudge R0_nudge32 think35 R0_base
    step B2_think "$PY" "$E" "$ROOT" B2_tuned32 think35 "${T[@]}" && nudge B2_nudge32 think35 B2_tuned32
    for b in 8k 16k; do
      for m in R0_base B2_tuned32; do
        step ${m}_budget$b "$PY" "$E" "$ROOT" $m budget$b "${T[@]}" && nudge ${m}_nudge$b budget$b $m
      done
    done
    [ "$NOTHINK" = 1 ] && step R0_nothink "$PY" "$E" "$ROOT" R0_base nothink35 "${T[@]}" ;;
  sweepA) step sweepA_${SHARD%/*} "$PY" "$HERE/sweep_b2.py" "$ROOT" --shard "$SHARD" ;;
  sweepB) step sweepB_${SHARD%/*} "$PY" "$HERE/sweep_b2.py" "$ROOT" --shard "$SHARD" --top ;;
  select) step sweep "$PY" "$HERE/sweep_b2.py" "$ROOT" ;;              # summary; runs anything a shard missed
  sweepC) step sweepC_${SHARD%/*} "$PY" "$HERE/sweep_b2.py" "$ROOT" --stage32 --shard "$SHARD" ;;
  select32) step sweep32 "$PY" "$HERE/sweep_b2.py" "$ROOT" --stage32 ;;  # writes models/B2_tuned32
  base|b2)
    m=$([ "$STAGE" = base ] && echo R0_base || echo B2_tuned32); k=$([ "$STAGE" = base ] && echo R0 || echo B2)
    case "$ARM" in
      think)   step ${k}_think "$PY" "$E" "$ROOT" $m think35 "${T[@]}" && nudge ${k}_nudge32 think35 $m ;;
      8k|16k)  step ${m}_budget$ARM "$PY" "$E" "$ROOT" $m budget$ARM "${T[@]}" && nudge ${m}_nudge$ARM budget$ARM $m ;;
      nothink) step ${k}_nothink "$PY" "$E" "$ROOT" $m nothink35 "${T[@]}" ;;
      *) echo "ARM must be think|8k|16k|nothink"; exit 2 ;;
    esac ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
echo "[$(date '+%F %T')] $M transfer-$PART job finished" | tee -a "$LOG/job.log" >> "$MAIN/logs/transfer.log"
