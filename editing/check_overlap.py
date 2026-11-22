"""Check that the practice problems (MATH train: B2's fit set and the sweep's dev sets) do not overlap MATH-500.

Text is lower-cased with whitespace removed; a pair is flagged if >= 50% of the shorter problem's 40-character
shingles also occur in the other. Prints the flagged pairs and writes <root>/auto/overlap.json.

Usage: python check_overlap.py <root>
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

K = 40


def norm(t):
    return "".join(t.lower().split())


def shingles(t):
    t = norm(t)
    return {t[i:i + K] for i in range(max(1, len(t) - K + 1))}


def main(root: Path):
    test = [json.loads(l) for l in open(root / "data" / "math500_sub500.jsonl")]
    train = [json.loads(l) for l in open(root / "data" / "math_train.jsonl")]
    idx, tsh = defaultdict(set), {}
    for r in test:
        tsh[r["id"]] = shingles(r["problem"])
        for s in tsh[r["id"]]:
            idx[s].add(r["id"])
    exact = {norm(r["problem"]) for r in test}
    flagged = []
    for r in train:
        sh = shingles(r["problem"])
        hits = Counter(t for s in sh for t in idx.get(s, ()))
        for t, c in hits.items():
            frac = c / min(len(sh), len(tsh[t]))
            if frac >= 0.5:
                flagged.append({"train": r["id"], "test": t, "shared": round(frac, 3)})
    n_exact = sum(norm(r["problem"]) in exact for r in train)
    out = {"n_train": len(train), "n_test": len(test), "exact_duplicates": n_exact, "flagged": flagged}
    (root / "auto" / "overlap.json").write_text(json.dumps(out, indent=1))
    print(f"train {len(train)} vs MATH-500 {len(test)}: exact duplicates {n_exact}, near duplicates {len(flagged)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
