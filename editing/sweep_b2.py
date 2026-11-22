"""Budget-matched tuning of B2 (the LLM-free loop direction), so it gets about as many tries as the LLM did (~50).

Search space (all choices automated; directions and column rankings from auto_direction.py fit -> dirs_all.json):
  layer sets: 5 single layers (9, 12, 15, 18, 21) and 5 pairs 4 apart ((9,13) ... (17,21)), chosen as a regular grid
             (other depths: the same grid scaled by n_layers / 24, see grid())
  alpha in {3, 6, 12} with the top-64 columns, plus alpha 6 with the top-256 columns  -> 40 configs
Scoring uses practice problems only (MATH train, disjoint from the 160 used to fit and from MATH-500):
  stage A: every config on 64 problems x 1 sample, thinking on, 16k tokens       (data/dev_a.jsonl)
  stage B: the top 5 of stage A on 128 new problems x 2 samples, same protocol    (data/dev_b.jsonl)
  winner = best stage-B accuracy -> models/B2_tuned (then evaluated on MATH-500 by job_next.sh)
Base and the released edit are scored on both dev sets for reference (not counted as tries).
Every run is saved as results/<name>__think16k__dev{A,B}.json, so the sweep resumes after an interruption.

Stage C (--stage32, added after stage B's winner failed on MATH-500 at 32k): the same 5 configs on the stage-B
problems at the test's 32k tokens; winner -> models/B2_tuned32, report in auto/sweep32.json. 50 tries in total.

Usage: python sweep_b2.py <root> [--stage32] [--shard i/n [--top]]
  --shard i/n: score only every n-th config starting at i, write no summary (parallel jobs; a run without --shard
  then reads the saved results, picks the winners and writes the summary). Stage A by default; --top: stage B
  (needs all of stage A); with --stage32: stage C (needs auto/sweep.json). Shard 0 also scores the references.
"""
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def grid(nl=24):
    """The search grid, scaled to the model's depth (for 24 layers exactly the original grid)."""
    f = lambda x: round(x * nl / 24)
    sets = [[f(x)] for x in (9, 12, 15, 18, 21)] + [[f(x), f(x) + f(4)] for x in (9, 11, 13, 15, 17)]
    return [(s, a, 64) for s in sets for a in (3.0, 6.0, 12.0)] + [(s, 6.0, 256) for s in sets]


SETS = [[9], [12], [15], [18], [21], [9, 13], [11, 15], [13, 17], [15, 19], [17, 21]]
CONFIGS = grid(24)
assert [c[0] for c in CONFIGS[::3][:10]] == SETS


def name(s, a, k):
    return f"SW_L{'-'.join(map(str, s))}_a{a:g}_k{k}"


def dev_sets(root):
    rows = [json.loads(l) for l in open(root / "data" / "math_train.jsonl")]
    random.Random(0).shuffle(rows)  # same order as auto_direction.gen, which used rows[:160]
    for fn, part in (("dev_a.jsonl", rows[160:224]), ("dev_b.jsonl", rows[224:352])):
        f = root / "data" / fn
        if not f.exists():
            with open(f, "w") as o:
                for r in part:
                    o.write(json.dumps({"id": r["id"], "problem": r["problem"], "answer": r["answer"]}) + "\n")


def run(root, model, stage):
    proto = "think35" if stage == "C" else "think16k"  # stage C: the stage-B problems again, at the test's 32k
    out = root / "results" / f"{model}__{proto}__dev{stage}.json"
    if not out.exists():
        subset, n = ("data/dev_a.jsonl", 1) if stage == "A" else ("data/dev_b.jsonl", 2)
        subprocess.run([sys.executable, str(HERE / "eval_math.py"), str(root), model, proto, "--subset", subset,
                        "--n", str(n), "--tag", f"__dev{stage}"], check=True)
    d = json.loads(out.read_text())
    return {"acc": d["acc"], "cut": d["cut_share"], "tok": d["mean_tokens"], "n": d["n_problems"] * d["samples"]}


def main(root: Path, stage32: bool = False, shard=None, top_stage=False):
    sys.path.insert(0, str(HERE))
    from prepare import apply_projection, n_layers, weights_file, write_variant
    from safetensors.torch import load_file
    dev_sets(root)
    dirs = {int(k): v for k, v in json.loads((root / "auto" / "dirs_all.json").read_text()).items()}
    base_dir = (root / "models" / "R0_base").resolve()
    base_sd = load_file(str(weights_file(base_dir)))
    configs = grid(n_layers(base_sd))
    refs = [m for m in ("R0_base", "R1_astra_p50c") if (root / "models" / m).exists()]  # no LLM edit on other models

    def build(s, a, k, out):
        if not (out / "variant.json").exists():  # written last, so a write cut by a time limit is redone
            shutil.rmtree(out, ignore_errors=True)
            edits = {L: {"u": dirs[L]["u"], "J": sorted(dirs[L]["rank"][:k]), "alpha": a} for L in s}
            write_variant(base_dir, out, apply_projection(base_sd, edits), {"kind": "B2 sweep", "layers": s, "alpha": a, "k": k})

    def score(cfg, stage):
        nm = name(*cfg)
        if not (root / "results" / f"{nm}__{'think35' if stage == 'C' else 'think16k'}__dev{stage}.json").exists():
            build(*cfg, root / "models" / nm)
        r = run(root, nm, stage)
        shutil.rmtree(root / "models" / nm, ignore_errors=True)  # 1.6 GB each; rebuilt from dirs_all.json if needed
        print(f"[{stage}] {nm}: acc {r['acc']:.3f} cut {r['cut']:.3f}", flush=True)
        return r

    if shard:  # one share of one stage; the summary is left to a later run without --shard
        i, n = shard
        if stage32:
            stage, todo = "C", [tuple(v["cfg"]) for v in json.loads((root / "auto" / "sweep.json").read_text())["stage_B"].values()]
        elif top_stage:
            A = {name(*c): dict(score(c, "A"), cfg=c) for c in configs}  # all saved already: reads only
            stage, todo = "B", [A[k]["cfg"] for k in sorted(A, key=lambda k: (-A[k]["acc"], A[k]["cut"]))[:5]]
        else:
            stage, todo = "A", configs
        if i == 0:
            for m in refs:
                run(root, m, stage)
        for c in todo[i::n]:
            score(tuple(c), stage)
        return
    if stage32:  # stage C: rescore stage B's 5 configs at 32k (the test budget), pick the winner from those scores
        sw = json.loads((root / "auto" / "sweep.json").read_text())
        ref = {m: run(root, m, "C") for m in refs}
        C = {n: dict(score(tuple(v["cfg"]), "C"), cfg=v["cfg"]) for n, v in sw["stage_B"].items()}
        best = max(C, key=lambda n: (C[n]["acc"], sw["stage_B"][n]["acc"]))
        s, a, k = C[best]["cfg"]
        build(s, a, k, root / "models" / "B2_tuned32")
        (root / "auto" / "sweep32.json").write_text(json.dumps(
            {"tries": sw["tries"] + len(C), "reference": ref, "stage_C": C, "best": best,
             "best_cfg": {"layers": s, "alpha": a, "k": k}}, indent=1))
        print("best", best, json.dumps(C[best])); return
    ref = {m: {st: run(root, m, st) for st in "AB"} for m in refs}
    A = {name(*c): dict(score(c, "A"), cfg=c) for c in configs}
    top = sorted(A, key=lambda n: (-A[n]["acc"], A[n]["cut"]))[:5]
    B = {n: dict(score(A[n]["cfg"], "B"), cfg=A[n]["cfg"]) for n in top}
    best = max(B, key=lambda n: (B[n]["acc"], A[n]["acc"]))
    s, a, k = B[best]["cfg"]
    build(s, a, k, root / "models" / "B2_tuned")
    out = {"tries": len(configs) + len(top), "reference": ref, "stage_A": A, "stage_B": B, "best": best,
           "best_cfg": {"layers": s, "alpha": a, "k": k}}
    (root / "auto" / "sweep.json").write_text(json.dumps(out, indent=1))
    print("best", best, json.dumps(B[best]))


if __name__ == "__main__":
    sh = tuple(map(int, sys.argv[sys.argv.index("--shard") + 1].split("/"))) if "--shard" in sys.argv else None
    main(Path(sys.argv[1]), stage32="--stage32" in sys.argv, shard=sh, top_stage="--top" in sys.argv)
