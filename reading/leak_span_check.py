"""Leak check for C2b: can the dead (mixed) rows be told apart from trained rows by linear structure alone?

Each C2b dead row is a mix of 3 trained rows of the same channel plus 5% noise. A reader who suspects this could fit
every interior row as a combination of 3 other interior rows of its channel (no labels used) and flag rows that fit
well. If dead rows fit much better than trained rows, the occupancy map (the footprint) is recoverable.

Per channel: greedy 3-atom fit (orthogonal matching pursuit) over the other 63 interior rows; score = relative
residual. Reports AUC (dead vs trained) and the best-threshold accuracy (optimistic: the threshold uses the labels).
Usage: python -I leak_span_check.py [condition, default c2b_erased_mix]  ->  results/leak_span_<condition>.json
Also reports AUCs for simple per-row statistics (norm, max |w|, kurtosis); AUC far from 0.5 either way = leak.
"""
import json
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from make_footprint_erased_prompts import HEADER, dead_mask

ROOT = Path(__file__).parent
COND = next((a for a in sys.argv[1:] if not a.startswith("-")), "c2b_erased_mix")
NPZ, PROMPTS = ROOT / "data/exp1_networks", ROOT / "data/prompts" / COND


def omp_resid(y, D, k=3):
    r, idx = y.copy(), []
    for _ in range(k):
        idx.append(int(np.argmax(np.abs(D @ r) / np.linalg.norm(D, axis=1))))
        A = D[idx].T
        r = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return np.linalg.norm(r) / np.linalg.norm(y)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] < neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def read_rows(path):
    lines = path.read_text().split("\n")
    s = next(i for i, l in enumerate(lines) if l.startswith(HEADER)) + 1
    return np.array([[float(v) for v in l.split()] for l in lines[s:s + 500]])


out = {}
for meta in sorted(NPZ.glob("*.json")):
    game = json.loads(meta.read_text())["game"]
    dead = dead_mask(np.load(NPZ / f"{meta.stem}.npz")["actor/dense1/kernel"])
    W = read_rows(PROMPTS / f"{game}.txt")
    sd, st, per_ch = [], [], {}
    for ch in range(5):
        cells = [(r, c) for r in range(1, 9) for c in range(1, 9)]
        idx = [(r * 10 + c) * 5 + ch for r, c in cells]
        lab = np.array([dead[r, c, ch] for r, c in cells])
        if lab.all() or not lab.any():
            continue
        X = W[idx]
        res = np.array([omp_resid(X[i], np.delete(X, i, 0)) for i in range(len(X))])
        z = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)
        stats = {"norm": np.linalg.norm(X, axis=1), "maxabs": np.abs(X).max(1), "kurt": (z ** 4).mean(1)}
        per_ch[ch] = {"n_dead": int(lab.sum()), "auc": auc(res[lab], res[~lab]),
                      **{f"auc_{k}": auc(v[lab], v[~lab]) for k, v in stats.items()},
                      "median_resid_dead": float(np.median(res[lab])), "median_resid_trained": float(np.median(res[~lab]))}
        sd += list(res[lab]); st += list(res[~lab])
    allr = np.array(sd + st); lab = np.array([1] * len(sd) + [0] * len(st))
    acc = max(((allr <= t) == lab).mean() for t in allr)
    out[game] = {"auc": auc(sd, st), "best_threshold_acc": float(acc), "base_rate": float(max(lab.mean(), 1 - lab.mean())),
                 "per_channel": per_ch}
    print(f"{game:9s} AUC {out[game]['auc']:.2f}  best acc {acc:.2f} (base {out[game]['base_rate']:.2f})  "
          + "  ".join(f"ch{c}:{v['auc']:.2f}({v['median_resid_dead']:.2f}/{v['median_resid_trained']:.2f})"
                      for c, v in per_ch.items()))
(ROOT / f"results/leak_span_{COND}.json").write_text(json.dumps(out, indent=1))
