#!/usr/bin/env bash
# Run one tool-free reader call per prompt file, in parallel, through the Claude Code CLI.
# All built-in tools are disabled (--tools ""), no MCP servers, no settings files, and the same
# neutral system line the authors used ("Answer the user's request.").
#
# Usage: run_reader.sh <model> <effort> <out_dir> <prompt.txt>...
#   e.g. run_reader.sh claude-opus-5-5 high runs/c0 prompts/c0/*.txt
# Each prompt writes <out_dir>/<basename>.json (the CLI's JSON result: text, usage, cost).
set -euo pipefail
CLAUDE_BIN="${CLAUDE_BIN:-claude}"   # the Claude Code CLI; set CLAUDE_BIN if it is not on PATH
model="$1"; effort="$2"; out="$3"; shift 3
mkdir -p "$out"
for p in "$@"; do
  name="$(basename "$p" .txt)"
  (
    cd "$out"
    "$CLAUDE_BIN" -p --model "$model" --effort "$effort" --tools "" --strict-mcp-config \
      --setting-sources "" --no-session-persistence \
      --system-prompt "Answer the user's request." --output-format json \
      < "$p" > "$name.json" 2> "$name.err" || echo "failed: $name" >&2
  ) &
done
wait
