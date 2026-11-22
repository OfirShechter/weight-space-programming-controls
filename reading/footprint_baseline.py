"""A non-LLM reader: name the game from the training-data footprint alone, with four hand-written rules.

It reads the same prompt files the LLM reader gets (weights as text), recomputes the per-cell map of trained W1
rows (row norm above 2x the median of the interior obstacle rows, as in exp1_footprint.py), and applies rules about
where the agent and the reward targets ever appeared, using only the playable 8 x 8 interior:
  agent confined to one column                                 -> Pong (a side paddle)
  agent confined to one row, reward targets in the top rows    -> Breakout (paddle below bricks)
  agent covers many rows, reward targets only in the top row   -> Freeway (cross the road to the goal)
  agent region has holes (cells it never visits inside its
  bounding box: maze walls)                                    -> Pac-Man
  anything else                                                -> unknown
The rules encode common knowledge about the four games, the same knowledge an LLM brings. The point is not that
this is clever: it shows how much of "reading" the footprint alone explains, and that it fails exactly where the
footprint is erased.

Usage: python -I footprint_baseline.py <prompts_root> [--out results/footprint_baseline.json]
  <prompts_root>/<condition>/<game>.txt as written by make_control_prompts.py / make_footprint_erased_prompts.py
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling modules, also under python -I

import numpy as np

from make_control_prompts import parse

CONDS = ["c0_original", "c1_permuted", "c2_erased", "c2b_erased_mix", "c5_map_only"]
GAMES = ["breakout", "pong", "pacman", "freeway"]


def trained_map(W1):
    m = np.linalg.norm(W1, axis=1).reshape(10, 10, 5)
    thr = 2 * np.median(m[1:9, 1:9, 3])
    return (m > thr)[1:9, 1:9, :]  # interior only, (8, 8, 5)


def name_game(t):
    agent, targets = t[:, :, 0], t[:, :, 2]
    ar, ac = np.nonzero(agent)
    tr, _ = np.nonzero(targets)
    if len(ar) == 0:
        return "unknown", "agent never appears"
    if len(set(ac)) == 1:
        return "pong", f"agent only in column {ac[0] + 1}"
    if len(set(ar)) == 1:
        if len(tr) and tr.max() <= 2:
            return "breakout", f"agent only in row {ar[0] + 1}, targets only in rows 1-{tr.max() + 1}"
        return "unknown", f"agent only in row {ar[0] + 1}, targets elsewhere"
    if len(tr) and set(tr) == {0} and len(set(ar)) >= 4:
        return "freeway", f"agent in {len(set(ar))} rows, targets only in row 1"
    box = agent[ar.min():ar.max() + 1, ac.min():ac.max() + 1]
    if box.mean() < 0.9:
        return "pacman", f"agent visits {box.mean():.0%} of its bounding box (walls)"
    return "unknown", f"agent fills {box.mean():.0%} of rows {ar.min() + 1}-{ar.max() + 1}; targets in {len(set(tr))} rows"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.prompts)
    res = {}
    for cond in CONDS:
        res[cond] = {}
        for g in GAMES:
            f = root / cond / f"{g}.txt"
            if not f.exists():
                continue
            P, _ = parse(f.read_text().split("\n"))
            guess, why = name_game(trained_map(P["W1"]))
            res[cond][g] = {"guess": guess, "correct": guess == g, "evidence": why}
        ok = sum(v["correct"] for v in res[cond].values())
        print(f"{cond:13s} {ok}/{len(res[cond])}   " + "; ".join(f"{g}->{v['guess']}" for g, v in res[cond].items()))
        for g, v in res[cond].items():
            print(f"{'':16s}{g:9s} {v['evidence']}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
