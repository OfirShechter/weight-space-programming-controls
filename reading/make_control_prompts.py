"""Build control prompts for the exp 1 reading test from the authors' original prompts.

  c1_permuted : hidden units of both layers permuted, with random tanh sign flips. The network computes
                exactly the same function. A reader that truly reads the weights must not change its answer.
  c5_map_only : keep the occupancy map, destroy the function. Each trained W1 row has its 32 values shuffled,
                so every row norm and the per-cell map are unchanged. Dead (init-scale) rows are untouched.
                b1, W2, b2, W3 and b3 are shuffled within each tensor, which also removes the action-usage cue.
                If the reader still names the game, the map alone is enough.

Usage: python -I make_control_prompts.py <prompt_dir> <npz_dir> <out_dir>
Writes <out_dir>/c1_permuted/<game>.txt and <out_dir>/c5_map_only/<game>.txt.
"""
import json
import sys
from pathlib import Path

import numpy as np

BLOCKS = [("W1", "W1 (layer 1 weights)", (500, 32)), ("b1", "b1 (layer 1 biases)", (32,)),
          ("W2", "W2 (layer 2 weights)", (32, 32)), ("b2", "b2 (layer 2 biases)", (32,)),
          ("W3", "W3 (output layer weights)", (32, 5)), ("b3", "b3 (output layer biases)", (5,))]


def parse(lines):
    out, where = {}, {}
    for key, header, shape in BLOCKS:
        i = next(k for k, l in enumerate(lines) if l.startswith(header)) + 1
        n = shape[0] if len(shape) == 2 else 1
        vals = np.array([[float(v) for v in l.split()] for l in lines[i:i + n]])
        out[key] = vals.reshape(shape)
        where[key] = (i, n)
    return out, where


def fmt_row(r):
    return " ".join(f"{float(v) + 0.0:.3f}" for v in r)


def render(lines, where, P):
    lines = list(lines)
    for key, _, shape in BLOCKS:
        i, n = where[key]
        rows = P[key] if len(shape) == 2 else P[key][None, :]
        lines[i:i + n] = [fmt_row(r) for r in rows]
    return lines


def forward(P, X):
    return np.tanh(np.tanh(X @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"]) @ P["W3"] + P["b3"]


def permuted(P, rng):
    p1, p2 = rng.permutation(32), rng.permutation(32)
    s1, s2 = rng.choice([-1.0, 1.0], 32), rng.choice([-1.0, 1.0], 32)
    Q = {}
    Q["W1"] = P["W1"][:, p1] * s1
    Q["b1"] = P["b1"][p1] * s1
    W2 = P["W2"][p1, :] * s1[:, None]
    Q["W2"] = W2[:, p2] * s2
    Q["b2"] = P["b2"][p2] * s2
    Q["W3"] = P["W3"][p2, :] * s2[:, None]
    Q["b3"] = P["b3"].copy()
    return Q


def map_only(P, rng):
    m = np.linalg.norm(P["W1"], axis=1).reshape(10, 10, 5)
    thr = 2 * np.median(m[1:9, 1:9, 3])
    trained = (m > thr).reshape(-1)
    Q = {k: v.copy() for k, v in P.items()}
    for i in np.where(trained)[0]:
        Q["W1"][i] = rng.permutation(P["W1"][i])
    for k in ("b1", "W2", "b2", "W3", "b3"):
        Q[k] = rng.permutation(P[k].reshape(-1)).reshape(P[k].shape)
    return Q


def main(prompt_dir: Path, npz_dir: Path, out_dir: Path) -> None:
    rng = np.random.default_rng(1)
    X = (rng.random((4000, 500)) < 0.03).astype(float)
    for meta in sorted(npz_dir.glob("*.json")):
        game = json.loads(meta.read_text())["game"]
        lines = (prompt_dir / f"{game}.txt").read_text().split("\n")
        P, where = parse(lines)
        for name, fn in (("c1_permuted", permuted), ("c5_map_only", map_only)):
            Q = fn(P, rng)
            d = out_dir / name
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{game}.txt").write_text("\n".join(render(lines, where, Q)))
            a, b = forward(P, X), forward(Q, X)
            same = np.mean(a.argmax(1) == b.argmax(1))
            mp = lambda W: np.linalg.norm(W, axis=1)
            print(f"{game:8s} {name:12s} greedy-action agreement with original {same:.2f}; "
                  f"W1 row norms identical: {np.allclose(mp(P['W1']), mp(Q['W1']), atol=1e-9)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
