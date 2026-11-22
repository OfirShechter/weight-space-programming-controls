"""Build 'footprint-erased' (condition C2) versions of the released exp 1 prompts.

For every interior cell and channel whose W1 row is still at init scale ("dead": the input was
never, or almost never, active during training), replace that row's text with a copy of a randomly
chosen *trained* W1 row of the same channel. Dead inputs are (almost) always 0 on reachable game
states, so the policy's behaviour on those states is unchanged. But every interior cell now looks
"trained", so the occupancy map that draws the game is gone.

Border cells (row/col 0 or 9) are left untouched; they are the wall and look alike in every game.
A channel with no trained interior rows is left as is (it is uniform already).
The mask is a threshold on row norms, so the function is only approximately preserved; exact
preservation needs the environments (log which inputs are ever non-zero).

--mix (condition C2b, 2026-11-17): the copies above leak. Each dead row equals some trained row, so the number of
distinct rows per channel and where they repeat still draw the map (Fable 5.2 used this in 15 of 16 reads). With
--mix every dead row is instead a fresh random mix of 3 trained rows of the same channel (Dirichlet weights), rescaled
to the norm of another random trained row, plus Gaussian noise (5% of that norm per entry). No two rows are equal,
and the function is preserved exactly as before (dead inputs are 0 on reachable states).

--gauss (condition C2c, 2026-11-18): C2b still leaks through linear structure (leak_span_check.py: 3 other rows
rebuild a mixed row to the 5% noise level, AUC ~0.85 for dead vs trained). With --gauss every dead row is an
independent draw from a Gaussian fitted to the trained rows of its channel (their mean and covariance, plus 5% of the
average variance on the diagonal so a draw is not confined to the trained rows' span).
REJECTED, not used for reads: it leaks the other way. The trained rows are heavy-tailed and low-rank in some channels,
the Gaussian draws are not (single-channel AUCs 0.0-0.2 or 0.8-0.9 for residual, norm or kurtosis, at any shrinkage
0-0.05; results/leak_span_c2c_erased_gauss.json). Any synthetic fill differs from trained rows in some statistic, so
The post rests on the end-to-end test instead (leak_span_rules.py: the map recovered from C2b names no game).

Usage: python -I make_footprint_erased_prompts.py <prompt_dir> <npz_dir> <out_dir> [--mix | --gauss]
  prompt_dir: original <game>.txt prompts; npz_dir: <id>.npz + <id>.json from exp1_which_game.
"""
import json
import sys
from pathlib import Path

import numpy as np

SHRINK = float(__import__("os").environ.get("C2C_SHRINK", "0.05"))
HEADER = "W1 (layer 1 weights), shape 500 x 32"


def dead_mask(W1: np.ndarray) -> np.ndarray:
    m = np.linalg.norm(W1, axis=1).reshape(10, 10, 5)
    thr = 2 * np.median(m[1:9, 1:9, 3])
    return m <= thr  # (10, 10, 5) True = still at init scale


def mix_row(T: np.ndarray, rng) -> str:
    """A new row from the trained rows T of one channel: random convex mix, a trained row's norm, plus noise."""
    k = min(3, len(T))
    v = rng.dirichlet(np.ones(k)) @ T[rng.choice(len(T), k, replace=False)]
    n = np.linalg.norm(T[rng.integers(len(T))])
    v = v / (np.linalg.norm(v) + 1e-12) * n
    v = v + rng.normal(0, 0.05 * n / np.sqrt(len(v)), len(v))
    return " ".join(f"{x:.3f}" for x in v)


def gauss_row(T: np.ndarray, rng) -> str:
    """A new row drawn from a Gaussian fitted to the trained rows T of one channel (shrunk covariance)."""
    mu = T.mean(0)
    S = np.cov(T, rowvar=False) if len(T) > 1 else np.zeros((T.shape[1],) * 2)
    S = S + SHRINK * np.trace(S) / len(mu) * np.eye(len(mu)) if np.trace(S) > 0 else np.eye(len(mu)) * 1e-4
    return " ".join(f"{x:.3f}" for x in rng.multivariate_normal(mu, S))


def main(prompt_dir: Path, npz_dir: Path, out_dir: Path, mix: bool = False, gauss: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for meta_path in sorted(npz_dir.glob("*.json")):
        game = json.loads(meta_path.read_text())["game"]
        W1 = np.load(npz_dir / f"{meta_path.stem}.npz", allow_pickle=False)["actor/dense1/kernel"]
        dead = dead_mask(W1)

        lines = (prompt_dir / f"{game}.txt").read_text().split("\n")
        start = next(i for i, l in enumerate(lines) if l.startswith(HEADER)) + 1
        rows = lines[start:start + 500]
        assert len(rows) == 500 and all(len(r.split()) == 32 for r in rows)
        # sanity: text rows match the npz to 3 decimals
        assert np.allclose(np.array([[float(v) for v in r.split()] for r in rows]), W1, atol=6e-4)

        new_rows, n_filled = list(rows), 0
        for ch in range(5):
            trained = [(r * 10 + c) * 5 + ch for r in range(1, 9) for c in range(1, 9) if not dead[r, c, ch]]
            if not trained:
                continue
            for r in range(1, 9):
                for c in range(1, 9):
                    if dead[r, c, ch]:
                        new_rows[(r * 10 + c) * 5 + ch] = (gauss_row(W1[trained], rng) if gauss
                                                           else mix_row(W1[trained], rng) if mix
                                                           else rows[rng.choice(trained)])
                        n_filled += 1
        lines[start:start + 500] = new_rows
        (out_dir / f"{game}.txt").write_text("\n".join(lines))
        print(f"{game}: filled {n_filled} dead interior rows")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), "--mix" in sys.argv, "--gauss" in sys.argv)
