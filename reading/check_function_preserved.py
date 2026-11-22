"""Does each control version behave like the original network? Measured on the exact weights in the prompt files.

For every game, parse the weights back out of each condition's prompt text and compare its greedy action (argmax of
the 5 logits) with the original's on three sets of inputs:
  live    : game-like states that use only inputs the original network trained on (its footprint), with the wall
            border on: 1 agent cell, up to 3 moving objects, up to 3 reward targets, previous-frame copies.
            Real game states only ever activate trained inputs, so this is the set that matters.
  random  : random sparse binary inputs (3% density, border on), which also switch on untrained inputs.
  one-dead: live states with one untrained input switched on: the worst case for the footprint-erased version.
Expected: permuted 100% everywhere; footprint erased ~100% on live and lower on one-dead; map only near chance (20%).

Usage: python -I check_function_preserved.py <prompts_root> [--out results/function_check.json]
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling modules, also under python -I

import numpy as np

from make_control_prompts import forward, parse

CONDS = ["c1_permuted", "c2_erased", "c2b_erased_mix", "c5_map_only"]
GAMES = ["breakout", "pong", "pacman", "freeway"]
BORDER = [(r * 10 + c) * 5 + 3 for r in range(10) for c in range(10) if r in (0, 9) or c in (0, 9)]


def live_mask(W1):
    m = np.linalg.norm(W1, axis=1)
    thr = 2 * np.median(m.reshape(10, 10, 5)[1:9, 1:9, 3])
    return m > thr


def live_states(live, rng, n):
    X = np.zeros((n, 500))
    X[:, BORDER] = 1
    by_ch = {ch: [i for i in range(500) if live[i] and i % 5 == ch and i not in BORDER] for ch in range(5)}
    for k in range(n):
        on = []
        if by_ch[0]:
            on += list(rng.choice(by_ch[0], 1))
        for ch, hi in ((1, 3), (2, 3)):
            if by_ch[ch]:
                on += list(rng.choice(by_ch[ch], min(len(by_ch[ch]), rng.integers(0, hi + 1)), replace=False))
        prev = [i - (i % 5) + 4 for i in on if i % 5 in (0, 1)]   # previous-frame channel at the same cells
        on += [i for i in prev if live[i]]
        X[k, on] = 1
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.prompts)
    rng = np.random.default_rng(0)
    res = {}
    print(f"{'game':9s} {'version':12s} {'live':>7s} {'random':>7s} {'one-dead':>9s}   (greedy-action agreement with the original)")
    for g in GAMES:
        P, _ = parse((root / "c0_original" / f"{g}.txt").read_text().split("\n"))
        live = live_mask(P["W1"])
        dead = [i for i in range(500) if not live[i] and i not in BORDER]
        XL = live_states(live, rng, 4000)
        XR = (rng.random((4000, 500)) < 0.03).astype(float); XR[:, BORDER] = 1
        XD = XL.copy(); XD[np.arange(len(XD)), rng.choice(dead, len(XD))] = 1
        res[g] = {"n_live_inputs": int(live.sum()), "n_dead_inputs": len(dead)}
        for cond in CONDS:
            f = root / cond / f"{g}.txt"
            if not f.exists():
                continue
            Q, _ = parse(f.read_text().split("\n"))
            agree = {k: float(np.mean(forward(P, X).argmax(1) == forward(Q, X).argmax(1))) for k, X in
                     (("live", XL), ("random", XR), ("one_dead", XD))}
            res[g][cond] = agree
            print(f"{g:9s} {cond:12s} {100 * agree['live']:6.1f}% {100 * agree['random']:6.1f}% {100 * agree['one_dead']:8.1f}%")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
