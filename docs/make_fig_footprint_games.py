"""The footprint figure in the post (footprint_games.png).

Top row: each game drawn on the same 10x10 grid as the network's input. Below it, the per-cell first-layer row norms
of the original networks, the map-only versions and the footprint-erased versions (agent, moving objects and reward
targets as red, green and blue). The drawings are placed using the original networks' footprints, so drawing and map
line up: Pac-Man's maze is where the agent rows were trained, and so on.

Reads the prompt files in ../reading/data/prompts. Usage: python -I docs/make_fig_footprint_games.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle, Wedge

HERE = Path(__file__).resolve().parent
READING = HERE.parent / "reading"
sys.path.insert(0, str(READING))
from make_footprint_erased_prompts import HEADER  # noqa: E402

GAMES = [("breakout", "Breakout"), ("pong", "Pong"), ("pacman", "Pac-Man"), ("freeway", "Freeway")]
ROWS = [("c0_original", "Original"), ("c5_map_only", "Map only"), ("c2b_erased_mix", "Footprint\nerased")]


def norms(cond, game):
    lines = (READING / "data" / "prompts" / cond / f"{game}.txt").read_text().splitlines()
    s = next(i for i, l in enumerate(lines) if l.startswith(HEADER)) + 1
    W = np.array([[float(v) for v in l.split()] for l in lines[s:s + 500]])
    return np.linalg.norm(W, axis=1).reshape(10, 10, 5)


def panel(cond, game):
    m = norms(cond, game)[:, :, :3]                       # agent, moving objects, reward targets
    return m / m.max(axis=(0, 1), keepdims=True)


def trained(game):
    m = norms("c0_original", game)
    return m > 2 * np.median(m[1:9, 1:9, 3])              # (10, 10, 5) bool: the map


def cell(ax, r, c, color, pad=0.0, **kw):
    ax.add_patch(Rectangle((c + pad, r + pad), 1 - 2 * pad, 1 - 2 * pad, color=color, lw=0, **kw))


def breakout(ax, t):
    ax.set_facecolor("#000000")
    for c in range(10):                                   # grey frame (top and sides)
        cell(ax, 0, c, "#8c8c8c")
    for r in range(10):
        cell(ax, r, 0, "#8c8c8c"); cell(ax, r, 9, "#8c8c8c")
    for r, col in zip((1, 2, 3), ("#c84848", "#c66c3a", "#b47a30")):
        for c in range(10):
            if t[r, c, 2]:                                # bricks = reward-target cells
                cell(ax, r, c, col, pad=0.06)
    ax.add_patch(Rectangle((3.0, 8.3), 3.0, 0.45, color="#c84848", lw=0))   # paddle on the agent row
    ax.add_patch(Rectangle((5.3, 5.3), 0.4, 0.4, color="#c84848", lw=0))    # ball


def pong(ax, t):
    ax.set_facecolor("#000000")
    for c in range(10):
        cell(ax, 0, c, "#d0d0d0", pad=0.0); cell(ax, 9, c, "#d0d0d0")
    for r in np.arange(1.15, 9, 0.7):                     # centre net
        ax.add_patch(Rectangle((4.93, r), 0.14, 0.4, color="#d0d0d0", lw=0))
    ax.add_patch(Rectangle((1.3, 3.0), 0.4, 3.0, color="#ececec", lw=0))    # our paddle (agent column)
    ax.add_patch(Rectangle((8.3, 5.0), 0.4, 3.0, color="#ececec", lw=0))    # opponent
    ax.add_patch(Rectangle((6.3, 3.3), 0.4, 0.4, color="#ececec", lw=0))    # ball
    ax.text(3.0, 2.2, "3", color="#d0d0d0", fontsize=10, ha="center", va="center", family="monospace")
    ax.text(7.0, 2.2, "1", color="#d0d0d0", fontsize=10, ha="center", va="center", family="monospace")


def pacman(ax, t):
    ax.set_facecolor("#000000")
    maze = t[:, :, 0]
    for r in range(10):
        for c in range(10):
            if not maze[r, c]:                            # wall = the agent was never there
                cell(ax, r, c, "#1f2fd6", pad=0.12)
                ax.add_patch(Rectangle((c + 0.12, r + 0.12), 0.76, 0.76, fill=False, ec="#4a5cff", lw=0.8))
    for r in range(10):
        for c in range(10):
            if maze[r, c] and t[r, c, 2]:
                ax.add_patch(Circle((c + 0.5, r + 0.5), 0.09, color="#ffcfa8", lw=0))   # pellets
    for (r, c) in ((1, 1), (1, 8), (8, 1), (8, 8)):
        if maze[r, c]:
            ax.add_patch(Circle((c + 0.5, r + 0.5), 0.22, color="#ffcfa8", lw=0))       # power pellets
    ax.add_patch(Wedge((4.5, 8.5), 0.42, 30, 330, color="#ffe600", lw=0))             # Pac-Man, facing right
    for (r, c, col) in ((4, 2, "#ff2a2a"), (4, 7, "#ffa6e0")):                        # two ghosts
        x, y = c + 0.5, r + 0.5
        body = [(x - 0.4, y + 0.4), (x - 0.4, y), (x + 0.4, y), (x + 0.4, y + 0.4),
                (x + 0.27, y + 0.27), (x + 0.13, y + 0.4), (x, y + 0.27), (x - 0.13, y + 0.4), (x - 0.27, y + 0.27)]
        ax.add_patch(Polygon(body, color=col, lw=0))
        ax.add_patch(Circle((x, y), 0.4, color=col, lw=0, clip_on=True))
        ax.add_patch(Rectangle((x - 0.4, y), 0.8, 0.27, color=col, lw=0))
        for dx in (-0.15, 0.15):
            ax.add_patch(Circle((x + dx, y - 0.05), 0.11, color="white", lw=0))
            ax.add_patch(Circle((x + dx + 0.04, y - 0.03), 0.05, color="#1f2fd6", lw=0))


def freeway(ax, t):
    ax.set_facecolor("#5a5a5a")
    for c in range(10):
        cell(ax, 0, c, "#3f8f3f"); cell(ax, 9, c, "#3f8f3f")
    for r in range(10):
        cell(ax, r, 0, "#3f8f3f"); cell(ax, r, 9, "#3f8f3f")
    for c in range(1, 9):
        cell(ax, 1, c, "#a8a8a8")                         # goal sidewalk (reward-target row)
        cell(ax, 8, c, "#a8a8a8")                         # start sidewalk
    lanes = [r for r in range(10) if t[r, 1:9, 1].any()]  # car rows
    for r in lanes[:-1]:
        for x in np.arange(1.1, 9, 0.8):                  # dashed lane lines
            ax.add_patch(Rectangle((x, r + 0.96), 0.45, 0.08, color="#f2d64b", lw=0))
    cars = {2: (2, "#e04040"), 3: (6, "#3a7be0"), 4: (3, "#f0f0f0"), 5: (7, "#e0a030"), 6: (1, "#a050d0"), 7: (5, "#40c0c0")}
    for r in lanes:
        c, col = cars.get(r, (4, "#e04040"))
        ax.add_patch(Rectangle((c + 0.05, r + 0.2), 1.4, 0.6, color=col, lw=0))
        ax.add_patch(Rectangle((c + 0.95, r + 0.28), 0.3, 0.44, color="#1a1a1a", lw=0, alpha=0.6))  # windscreen
    ax.add_patch(Circle((4.5, 8.55), 0.28, color="#fff35c", lw=0))                    # chicken
    ax.add_patch(Polygon([(4.4, 8.25), (4.6, 8.25), (4.5, 8.08)], color="#ff8c1a", lw=0))


DRAW = {"breakout": breakout, "pong": pong, "pacman": pacman, "freeway": freeway}


def main():
    fig, axes = plt.subplots(4, 4, figsize=(6.6, 6.9))
    for j, (game, gname) in enumerate(GAMES):                 # top row: the game, drawn on the grid
        ax = axes[0, j]
        DRAW[game](ax, trained(game))
        ax.set_title(gname, fontsize=9)
        if j == 0:
            ax.set_ylabel("Game\n(illustration)", fontsize=9)
    for i, (cond, rname) in enumerate(ROWS, start=1):
        for j, (game, gname) in enumerate(GAMES):
            ax = axes[i, j]
            ax.imshow(panel(cond, game), interpolation="nearest", extent=(0, 10, 10, 0))
            if j == 0:
                ax.set_ylabel(rname, fontsize=9)
    for ax in axes.flat:
        ax.set_xlim(0, 10); ax.set_ylim(10, 0); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[0]:                                        # faint 10x10 grid so the cells line up
        for k in range(1, 10):
            ax.axhline(k, color="white", lw=0.25, alpha=0.12); ax.axvline(k, color="white", lw=0.25, alpha=0.12)
    fig.tight_layout(pad=0.4)
    fig.savefig(HERE / "footprint_games.png", dpi=160)
    print("written", HERE / "footprint_games.png")


if __name__ == "__main__":
    main()
