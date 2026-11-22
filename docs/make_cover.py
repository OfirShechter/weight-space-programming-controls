"""The post's link-preview image (assets/cover.png, 1200 x 630): the title next to the four games and the
first-layer footprints of the four released networks.

Reads the prompt files in ../reading/data/prompts. Usage: python -I docs/make_cover.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_fig_footprint_games import DRAW, GAMES, HERE, panel, trained  # noqa: E402

W, H = 1200, 630                      # pixels; figure fractions below are relative to these
SIDE, GAP = 140, 12                   # panel size and gap between panels, pixels
INK, MUTED = "#1f2328", "#5b636d"


def main():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="white")

    fig.text(64 / W, 1 - 196 / H, "Footprints\nand loops", fontsize=46, fontweight="bold", color=INK,
             va="top", linespacing=1.05)
    fig.text(64 / W, 1 - 376 / H, "What is left of weight-space\nprogramming after the controls",
             fontsize=19, color=MUTED, va="top", linespacing=1.35)

    x0 = W - 56 - 4 * SIDE - 3 * GAP
    y_top = (H - 2 * SIDE - GAP - 34) / 2          # 34 px of game names above the top row
    for j, (game, name) in enumerate(GAMES):
        x = x0 + j * (SIDE + GAP)
        top = fig.add_axes([x / W, 1 - (y_top + 34 + SIDE) / H, SIDE / W, SIDE / H])
        DRAW[game](top, trained(game))
        top.set_title(name, fontsize=13, color=INK, pad=6)
        bottom = fig.add_axes([x / W, 1 - (y_top + 34 + 2 * SIDE + GAP) / H, SIDE / W, SIDE / H])
        bottom.imshow(panel("c0_original", game), interpolation="nearest", extent=(0, 10, 10, 0))
        for ax in (top, bottom):
            ax.set_xlim(0, 10); ax.set_ylim(10, 0); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
    fig.text(x0 / W, 1 - (y_top + 34 + 2 * SIDE + GAP + 26) / H,
             "Top: the game. Bottom: the trained network's first-layer row norms.", fontsize=11, color=MUTED, va="top")

    out = HERE / "assets" / "cover.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=100, facecolor="white")
    print("written", out)


if __name__ == "__main__":
    main()
