"""Does the C2b linear leak give the game away without an LLM? Recover the map from C2b text alone (a row is "dead"
if 3 other rows of its channel rebuild it to within residual THR; no labels), then apply the footprint rules.
Usage: python -I leak_span_rules.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from footprint_baseline import GAMES, name_game  # noqa: E402
from leak_span_check import omp_resid, read_rows  # noqa: E402  (runs the check once on import)

for thr in (0.1, 0.15, 0.2):
    out = []
    for game in GAMES:
        W = read_rows(Path(__file__).parent / f"data/prompts/c2b_erased_mix/{game}.txt").reshape(10, 10, 5, 32)
        t = np.zeros((8, 8, 5), bool)
        for ch in range(5):
            X = W[1:9, 1:9, ch].reshape(64, 32)
            res = np.array([omp_resid(X[i], np.delete(X, i, 0)) for i in range(64)])
            t[:, :, ch] = (res > thr).reshape(8, 8)
        guess, why = name_game(t)
        out.append(f"{game}->{guess} ({why})")
    print(f"thr {thr}: " + "; ".join(out))
