"""Score reader runs with the authors' rule: the first clause of the 'Game:' line is mapped to a game by keyword
(Frogger / road-crossing counts as Freeway, as in exp1_which_game/eval). Prints a per-condition table and writes JSON.

Usage: python -I score_runs.py <runs_dir> <out.json>     (<runs_dir>/<condition>/<game>_s<k>.json)
"""
import json
import re
import sys
from pathlib import Path

KEYS = {"breakout": ["breakout", "arkanoid", "brick"], "pong": ["pong", "squash"],
        "pacman": ["pac-man", "pacman", "pac man"], "freeway": ["freeway", "frogger", "road-crossing", "road crossing"]}
COND = ["c0_original", "c1_permuted", "c2_erased", "c2b_erased_mix", "c5_map_only"]


def grade(text, game):
    g = re.search(r"Game:\**\s*(.+)", text)
    c = re.search(r"Confidence:\**\s*([0-9]+)", text)
    name = re.sub(r"[*_`]", "", g.group(1)).strip() if g else ""
    first = name.replace("‑", "-").split(",")[0].split(";")[0].lower()
    hits = [k for k, kws in KEYS.items() if any(w in first for w in kws)]
    return {"answer": name[:160], "correct": hits[:1] == [game], "conf": int(c.group(1)) if c else None,
            "no_maze": "maze" in name.lower() and ("no " in name.lower() or "without" in name.lower())}


def main(runs: Path, out: Path):
    res = {}
    for cond in COND:
        for f in sorted((runs / cond).glob("*_s*.json")):
            game, k = f.stem.rsplit("_s", 1)
            d = json.loads(f.read_text())
            r = grade(d["result"], game)
            r["sec"] = (d.get("duration_ms") or 0) // 1000
            res.setdefault(cond, {}).setdefault(game, []).append(r)
    print(f"{'condition':14s} {'correct':>9s} {'conf':>5s}  per game")
    summary = {}
    for cond, games in res.items():
        allr = [r for rs in games.values() for r in rs]
        n, ok = len(allr), sum(r["correct"] for r in allr)
        conf = sum(r["conf"] or 0 for r in allr) / max(1, n)
        per = {g: f"{sum(r['correct'] for r in rs)}/{len(rs)}" for g, rs in games.items()}
        summary[cond] = {"n": n, "correct": ok, "mean_conf": round(conf, 1), "per_game": per}
        print(f"{cond:14s} {ok:>4d}/{n:<4d} {conf:5.0f}  {per}")
    out.write_text(json.dumps({"summary": summary, "runs": res}, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
