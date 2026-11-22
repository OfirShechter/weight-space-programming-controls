"""Resumable queue of tool-free reader calls.

Backends:
  claude : the Claude Code CLI (the logged-in subscription), all tools off; what the 7-8 Nov reads used.
  openai : any OpenAI-compatible /chat/completions endpoint (most APIs, or a local `vllm serve`). No tools are
           sent, so the call is tool-free by construction. The key is read from --api-key-env (never stored).
Both use the authors' system prompt ("Answer the user's request.") and one fresh call per read.

Each job is (condition, game, sample). Results go to <out>/<cond>/<game>_s<k>.json; finished jobs are skipped.
The queue stops at the first error result (e.g. a usage limit) and writes <out>/STOPPED with the message,
so it can be re-run after the limit resets. Keep one <out_dir> per reader model.

Usage: python -I run_queue.py <prompts_root> <out_dir> [--samples 5] [--par 2]
         [--backend claude --model claude-opus-5-5 --effort high]
         [--backend openai --base-url URL --model NAME --api-key-env OPENAI_API_KEY --max-tokens N --reasoning-effort high]
  <prompts_root>/<cond>/<game>.txt must exist for every condition used below.
  --samples N > 5 adds reads 6..N for the original, footprint-erased and map-only versions (permuted stays at 2).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GAMES = ["breakout", "pong", "pacman", "freeway"]
# priority order: the most decisive conditions first, more samples later
PLAN = [("c5_map_only", 0), ("c1_permuted", 0), ("c2_erased", 1), ("c5_map_only", 1), ("c0_original", 1),
        ("c2_erased", 2), ("c5_map_only", 2), ("c0_original", 2), ("c1_permuted", 1),
        ("c2_erased", 3), ("c5_map_only", 3), ("c0_original", 3), ("c2_erased", 4), ("c5_map_only", 4),
        ("c0_original", 4)]


def claude_bin():
    """The Claude Code CLI: $CLAUDE_BIN, else `claude` on PATH, else the newest copy bundled with the VS Code extension."""
    if os.environ.get("CLAUDE_BIN"):
        return os.environ["CLAUDE_BIN"]
    if shutil.which("claude"):
        return shutil.which("claude")
    c = sorted(glob.glob(os.path.expanduser("~/.vscode-server/extensions/anthropic.claude-code-*/resources/native-binary/claude")) +
               glob.glob(os.path.expanduser("~/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude.exe")),
               key=lambda s: [int(x) for x in s.split("claude-code-")[1].split("-")[0].split(".")])  # newest version
    if not c:
        sys.exit("Claude Code CLI not found: install it or set CLAUDE_BIN")
    return c[-1]


SYSTEM = "Answer the user's request."


def call_openai(args, prompt):
    """One chat completion; returns the same keys the claude backend keeps, or raises on an API error."""
    body = {"model": args.model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
    if args.max_tokens:
        body["max_tokens"] = args.max_tokens
    if args.reasoning_effort:
        body["reasoning_effort"] = args.reasoning_effort
    req = urllib.request.Request(args.base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {os.environ.get(args.api_key_env, 'none')}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read())
    ch = d["choices"][0]
    return {"result": ch["message"].get("content") or "", "usage": d.get("usage"), "total_cost_usd": None,
            "duration_ms": round(1000 * (time.time() - t0)), "is_error": False, "stop_reason": ch.get("finish_reason"),
            "reader": {"backend": "openai", "base_url": args.base_url, "model": d.get("model", args.model)}}


def run(job, args, out):
    cond, game, k = job
    dst = out / cond / f"{game}_s{k}.json"
    if (out / "STOPPED").exists():
        return job, "skipped"
    if args.backend == "openai":
        try:
            keep = call_openai(args, (Path(args.prompts) / cond / f"{game}.txt").read_text())
        except Exception as e:  # HTTP errors, limits, timeouts: stop the queue so it can be resumed
            (out / "STOPPED").write_text(f"{job}: {type(e).__name__}: {str(e)[:500]}")
            return job, "error"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(keep, indent=1))
        return job, "ok"
    cmd = [claude_bin(), "-p", "--model", args.model, "--effort", args.effort, "--tools", "", "--strict-mcp-config",
           "--setting-sources", "", "--no-session-persistence", "--system-prompt", "Answer the user's request.",
           "--output-format", "json"]
    with open(Path(args.prompts) / cond / f"{game}.txt", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdin=f, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=out)
    try:
        d = json.loads(p.stdout)
    except Exception:
        (out / "STOPPED").write_text(f"{job}: unparsable output\n{p.stdout[:500]}\n{p.stderr[:500]}")
        return job, "error"
    if d.get("is_error"):
        (out / "STOPPED").write_text(f"{job}: {d.get('result', '')[:500]}")
        return job, "error"
    keep = {k2: d.get(k2) for k2 in ("result", "usage", "total_cost_usd", "duration_ms", "is_error", "stop_reason")}
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(keep, indent=1))
    return job, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts"); ap.add_argument("out")
    ap.add_argument("--model", default="claude-opus-5-5"); ap.add_argument("--effort", default="high")
    ap.add_argument("--par", type=int, default=2)
    ap.add_argument("--backend", choices=["claude", "openai"], default="claude")
    ap.add_argument("--base-url", default=None); ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--max-tokens", type=int, default=None); ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--conds", default=None, help="comma-separated conditions: run only these, reads 0..samples-1")
    ap.add_argument("--max-jobs", type=int, default=0, help="run only the first N pending jobs (smoke test)")
    args = ap.parse_args()
    if args.backend == "openai" and not args.base_url:
        ap.error("--backend openai needs --base-url")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "STOPPED").unlink(missing_ok=True)
    # every read 0..N-1 of the three main versions (2 of the permuted check), most decisive versions first; the
    # original PLAN is kept as a prefix so the 7-8 Nov Opus run order is unchanged
    plan = PLAN + [(c, k) for k in range(args.samples) for c in ("c2_erased", "c5_map_only", "c0_original")]
    plan += [("c1_permuted", k) for k in range(2)]
    plan = list(dict.fromkeys(plan))
    if args.conds:
        plan = [(c, k) for k in range(args.samples) for c in args.conds.split(",")]
    (out / "reader.json").write_text(json.dumps({k: getattr(args, k) for k in
                                                 ("backend", "model", "effort", "base_url", "max_tokens",
                                                  "reasoning_effort", "samples")}, indent=1))
    jobs = [(c, g, k) for c, k in plan for g in GAMES if not (out / c / f"{g}_s{k}.json").exists()]
    jobs = jobs[:args.max_jobs] if args.max_jobs else jobs
    print(f"{len(jobs)} jobs to run", flush=True)
    with ThreadPoolExecutor(args.par) as ex:
        for job, status in ex.map(lambda j: run(j, args, out), jobs):
            print(job, status, flush=True)
    if (out / "STOPPED").exists():
        print("STOPPED:", (out / "STOPPED").read_text()[:300])
        sys.exit(3)


if __name__ == "__main__":
    main()
