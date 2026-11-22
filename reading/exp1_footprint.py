"""How much of the original Experiment 1 ("which game?") is explained by shallow weight statistics?

Loads the four released actor networks from RikiyaT/lm_understand-logs (exp1_which_game), downloading any that are
missing from <networks_dir> (data/exp1_networks already holds them), and prints, for each, without running the network on any game:
  1. the output-layer bias and per-action column norms (165 numbers),
  2. the mean action distribution on random sparse binary observations,
  3. a per-cell "footprint" map of first-layer row norms for channels 0-2.

Inputs that are never active during training receive no gradient, so their W1 rows stay at
initialisation scale. The footprint therefore draws where each channel was ever non-zero
in the training data, which is a picture of the game layout.

Usage:  python -I exp1_footprint.py <networks_dir>
"""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

BASE = "https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp1_which_game/networks"
IDS = ["0bbcc3ea", "a9f535cc", "bdf99279", "dfc6cb65"]
ACTS = ["noop", "up", "down", "left", "right"]


def fetch(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for i in IDS:
        for name in (f"{i}.npz", f"meta/{i}.json"):
            out = d / Path(name).name
            if not out.exists():
                urllib.request.urlretrieve(f"{BASE}/{name}", out)


def main(d: Path) -> None:
    fetch(d)
    rng = np.random.default_rng(0)
    for i in IDS:
        z = np.load(d / f"{i}.npz", allow_pickle=False)
        game = json.loads((d / f"{i}.json").read_text())["game"]
        W1, b1 = z["actor/dense1/kernel"], z["actor/dense1/bias"]
        W2, b2 = z["actor/dense2/kernel"], z["actor/dense2/bias"]
        W3, b3 = z["actor/logits/kernel"], z["actor/logits/bias"]

        print(f"\n=== {game} ({i})")
        print("  b3        :", " ".join(f"{a}={v:+.2f}" for a, v in zip(ACTS, b3)))
        print("  |W3 col|  :", " ".join(f"{a}={v:.2f}" for a, v in zip(ACTS, np.linalg.norm(W3, axis=0))))

        X = (rng.random((5000, 500)) < 0.03).astype(np.float32)
        for r in range(10):
            for c in range(10):
                if r in (0, 9) or c in (0, 9):
                    X[:, (r * 10 + c) * 5 + 3] = 1  # wall border is always in channel 3
        logits = np.tanh(np.tanh(X @ W1 + b1) @ W2 + b2) @ W3 + b3
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        print("  mean pi(a):", " ".join(f"{a}={v:.2f}" for a, v in zip(ACTS, p.mean(0))))

        m = np.linalg.norm(W1, axis=1).reshape(10, 10, 5)
        thr = 2 * np.median(m[1:9, 1:9, 3])  # interior obstacle rows are ~untouched init
        for ch, label in [(0, "agent"), (1, "moving objects"), (2, "reward targets")]:
            print(f"  ch{ch} {label}  ('#' = trained, '.' = ~init)")
            for r in range(1, 9):
                print("     " + "".join("#" if m[r, c, ch] > thr else "." for c in range(1, 9)))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
