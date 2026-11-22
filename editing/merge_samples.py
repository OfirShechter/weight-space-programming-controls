"""Merge two independent runs of the same arm (different EVAL_SEED) into one result with both samples per problem.

  results/<arm><tag>.json + results/<arm><tag>_s1.json -> results/<arm><tag>2.json
An arm with no _s1 run is copied as is when it already has 2 samples (thinking off was run with --n 2).

Usage: python merge_samples.py <results_dir> [--tag __mmlupro]
"""
import json
import sys
from pathlib import Path

ARMS = [f"{m}__{p}" for m in ("R0_base", "R1_astra_p50c")
        for p in ("nothink35", "budget8k", "budget8knudge", "think35", "think35nudge")]


def main(rdir: Path, tag: str):
    for arm in ARMS:
        a, b, out = rdir / f"{arm}{tag}.json", rdir / f"{arm}{tag}_s1.json", rdir / f"{arm}{tag}2.json"
        if not a.exists():
            print("missing", a.name); continue
        da = json.loads(a.read_text())
        if not b.exists():
            if da["samples"] >= 2:
                out.write_text(json.dumps(da)); print(f"{arm}: copied ({da['samples']} samples)")
            else:
                print(f"{arm}: no second run yet")
            continue
        db = json.loads(b.read_text())
        pb = {p["id"]: p for p in db["per_problem"]}
        assert set(pb) == {p["id"] for p in da["per_problem"]}, arm
        per = [{k: (p[k] + pb[p["id"]][k] if k in ("correct", "cut", "tokens") else p[k]) for k in p}
               for p in da["per_problem"]]
        flat = lambda k: [x for p in per for x in p[k]]
        n = len(flat("correct"))
        m = dict(da, samples=da["samples"] + db["samples"], acc=sum(flat("correct")) / n,
                 cut_share=sum(flat("cut")) / n, mean_tokens=sum(flat("tokens")) / n,
                 seconds=da.get("seconds", 0) + db.get("seconds", 0), per_problem=per, merged=[a.name, b.name])
        out.write_text(json.dumps(m))
        print(f"{arm}: {100*da['acc']:.1f} + {100*db['acc']:.1f} -> {100*m['acc']:.1f} ({m['samples']} samples)")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "__mmlupro")
