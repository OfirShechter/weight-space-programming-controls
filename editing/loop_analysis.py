"""When do answers start looping, and do they get out? Explains why base Qwen plateaus under budget forcing.

Reads saved full answers (eval_math.py *.all.jsonl.gz, thinking at 32k) and, per sample:
  onset   : approximate token position of the first repeated 20-word span (a loop has started)
  stuck   : the last 1,000 words are mostly (>= 80%) spans already written earlier (still looping at the end)
  escaped : a loop started but the answer did not end inside one
Token positions are estimated from the character position and the sample's saved token count.
Also reports, for budgets 4k/8k/16k/32k, the share of samples whose loop started before the budget: thinking after
that point adds nothing a nudge can use.

Usage: python loop_analysis.py <saved_dir> [--out results/loop_analysis.json]
"""
import argparse
import gzip
import json
from pathlib import Path

N = 20                      # words per span
TAIL = 1000                 # words checked for "stuck at the end"
BUDGETS = [4096, 8192, 16384, 32768]
ARMS = [("R0_base__think35__full500x4b", "Base"),
        ("R1_astra_p50c__think35__full500x4b", "Released edit"),
        ("B2_tuned32__think35__full500x4", "B2 tuned at 32k")]


def analyse(text, n_tokens):
    words = text.split()
    if len(words) < N + 1:
        return None, False
    seen, onset_w, rep = set(), None, []
    # character offset of each word, to map a word index to a token estimate
    offs, pos = [], 0
    for w in words:
        pos = text.find(w, pos); offs.append(pos); pos += len(w)
    for i in range(len(words) - N + 1):
        g = hash(tuple(words[i:i + N]))
        r = g in seen
        rep.append(r)
        if r and onset_w is None:
            onset_w = i + N - 1
        seen.add(g)
    tail = rep[-TAIL:] if len(rep) >= TAIL else rep
    stuck = sum(tail) / len(tail) >= 0.8
    onset_tok = None if onset_w is None else round(offs[onset_w] / max(len(text), 1) * n_tokens)
    return onset_tok, stuck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("saved"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    saved = Path(a.saved)
    out = {"span_words": N, "tail_words": TAIL, "arms": {}}
    print(f"{'arm':18s} {'n':>5s} {'loops':>6s} {'median onset':>13s} {'stuck@end':>10s} {'escaped':>8s}   "
          + "  ".join(f"onset<{b // 1024}k" for b in BUDGETS))
    for key, label in ARMS:
        f = saved / "answers" / f"{key}.all.jsonl.gz"
        meta = saved / "results" / f"{key}.json"
        if not f.exists() or not meta.exists():
            continue
        toks = {p["id"]: p["tokens"] for p in json.loads(meta.read_text())["per_problem"]}
        rows, onsets = [], []
        for r in map(json.loads, gzip.open(f, "rt", encoding="utf-8")):
            for j, (t, c, cut) in enumerate(zip(r["texts"], r["correct"], r["cut"])):
                o, s = analyse(t, toks[r["id"]][j])
                rows.append({"id": r["id"], "j": j, "onset": o, "stuck": s, "cut": cut, "correct": c})
        n = len(rows)
        loop = [x for x in rows if x["onset"] is not None]
        on = sorted(x["onset"] for x in loop)
        stuck = sum(x["stuck"] for x in rows)
        esc = sum(1 for x in loop if not x["stuck"])
        before = {b: sum(1 for x in loop if x["onset"] < b) / n for b in BUDGETS}
        curve = [(t, sum(1 for x in on if x < t) / n) for t in range(0, 32769, 1024)]
        acc_by = {k: (sum(x["correct"] for x in g) / len(g) if g else None, len(g)) for k, g in
                  (("no loop", [x for x in rows if x["onset"] is None]), ("loop, escaped", [x for x in loop if not x["stuck"]]),
                   ("loop, stuck at end", [x for x in rows if x["stuck"]]))}
        med = on[len(on) // 2] if on else None
        out["arms"][key] = {"label": label, "n": n, "loop_share": len(loop) / n, "median_onset_tokens": med,
                            "stuck_share": stuck / n, "escaped_share": esc / n, "onset_before_budget": before,
                            "onset_curve": curve, "acc_by_loop_status": acc_by}
        print(f"{label:18s} {n:5d} {100 * len(loop) / n:5.1f}% {med if med is not None else '-':>13} {100 * stuck / n:9.1f}% "
              f"{100 * esc / n:7.1f}%   " + "  ".join(f"{100 * before[b]:8.1f}%" for b in BUDGETS))
        for k, (acc, cnt) in acc_by.items():
            print(f"{'':18s}   accuracy, {k}: {'-' if acc is None else f'{100 * acc:.1f}%'} (n={cnt})")
    dst = Path(a.out) if a.out else saved / "results" / "loop_analysis.json"
    dst.write_text(json.dumps(out, indent=1))
    print("written", dst)


if __name__ == "__main__":
    main()
