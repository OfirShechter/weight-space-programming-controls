#!/usr/bin/env bash
# Run run_queue.py until every read is done, waiting out usage limits. Each read is saved on its own as it finishes
# and finished reads are skipped, so an interruption (limit, closed terminal, reboot) loses at most the reads in
# flight; just start this again. Any other error stops for good (no retry loop that spends tokens).
# Usage: bash run_until_done.sh <out_dir> [run_queue.py args...]
#   e.g. bash run_until_done.sh results/fable52 --model claude-fable-5-2 --effort high --samples 10 --par 1
set -u
cd "$(dirname "$0")"
OUT="$1"; shift
for try in $(seq 1 120); do
  python -I -X utf8 run_queue.py data/prompts "$OUT" "$@"; rc=$?
  [ $rc -eq 0 ] && { echo "[$(date '+%F %T')] all reads done"; exit 0; }
  msg=$(cat "$OUT/STOPPED" 2>/dev/null)
  if echo "$msg" | grep -qiE "limit|usage|rate|overloaded|529|429|credit|reset"; then
    echo "[$(date '+%F %T')] stopped by a limit, waiting 30 min (try $try): ${msg:0:200}"; sleep 1800
  else
    echo "[$(date '+%F %T')] stopped by another error, not retrying: ${msg:0:300}"; exit $rc
  fi
done
