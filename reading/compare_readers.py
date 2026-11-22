"""Reading results for several readers side by side, with Fisher exact tests between versions.

For each reader directory (one run_queue.py output per reader model), scores every read with score_runs.grade, then
prints correct / total per version and two-sided Fisher exact tests: map only vs footprint erased, and original vs
footprint erased. The non-LLM footprint baseline (footprint_baseline.py) is added as a reference row if given.

Usage: python -I compare_readers.py <label>=<runs_dir> [<label>=<runs_dir> ...] [--baseline results/footprint_baseline.json]
                                    [--out results/readers.json]
  e.g. python -I compare_readers.py opus-5.5=results/full fable-5.2=results/fable52 --baseline results/footprint_baseline.json
"""
import argparse
import json
from math import comb
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling modules, also under python -I

from score_runs import COND, grade

GAMES = ["breakout", "pong", "pacman", "freeway"]


def fisher_two_sided(a, b, c, d):
    """Exact test for the 2x2 table [[a, b], [c, d]] (rows: versions; columns: correct, wrong)."""
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    p = lambda x: comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)
    p0 = p(a)
    return min(1.0, sum(p(x) for x in range(max(0, c1 - r2), min(r1, c1) + 1) if p(x) <= p0 * (1 + 1e-9)))


def score_dir(d: Path):
    out = {}
    for cond in COND:
        res = []
        for f in sorted((d / cond).glob("*_s*.json")):
            game = f.stem.rsplit("_s", 1)[0]
            res.append(grade(json.loads(f.read_text()).get("result") or "", game))
        out[cond] = (sum(r["correct"] for r in res), len(res))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("readers", nargs="+"); ap.add_argument("--baseline", default=None); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = {}
    for spec in a.readers:
        label, d = spec.split("=", 1)
        rows[label] = score_dir(Path(d))
    if a.baseline:
        b = json.loads(Path(a.baseline).read_text())
        rows["footprint rules (no LLM)"] = {c: (sum(v["correct"] for v in b[c].values()), len(b[c])) for c in COND if c in b}
    print(f"{'reader':26s} " + " ".join(f"{c:>13s}" for c in COND) + f" {'p(map vs erased)':>17s} {'p(orig vs erased)':>18s}")
    out = {}
    for label, s in rows.items():
        cells = " ".join(f"{s[c][0]:>6d}/{s[c][1]:<6d}" if c in s and s[c][1] else f"{'-':>13s}" for c in COND)
        def fp(x, y):
            if x not in s or y not in s or not s[x][1] or not s[y][1]:
                return None
            (k1, n1), (k2, n2) = s[x], s[y]
            return fisher_two_sided(k1, n1 - k1, k2, n2 - k2)
        pm, po = fp("c5_map_only", "c2_erased"), fp("c0_original", "c2_erased")
        print(f"{label:26s} {cells} {('-' if pm is None else f'{pm:.4f}'):>17s} {('-' if po is None else f'{po:.4f}'):>18s}")
        out[label] = {"counts": s, "p_map_vs_erased": pm, "p_orig_vs_erased": po}
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
