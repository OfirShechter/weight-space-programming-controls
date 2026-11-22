"""Paired comparison of the 500-problem x 4-sample runs (*__full500x4.json).

Per problem, accuracy = mean of its 4 samples. Arms are compared on the same problems: mean difference, paired
standard error, z, two-sided p, and a 95% bootstrap interval over problems.

Usage: python compare_full.py <results_dir> [--suite math|gsm8k|mmlupro]
  math  (default): MATH-500, 500 x 4 (*__full500x4.json)   -> full500x4_compare.json
  gsm8k          : GSM8K test, 1,319 x 2 (*__gsm8k.json)     -> gsm8k_compare.json
  mmlupro        : MMLU-Pro, fixed 1,000 x 1 (*__mmlupro.json) -> mmlupro_compare.json
  transfer       : a transfer model, 500 x 2 (*__t500x2.json)   -> transfer_compare.json
"""
import json
import math
import random
import sys
from pathlib import Path

ARMS = [("R0_base__think35", "Base, thinking"),
        ("R1_astra_p50c__think35", "Released edit, thinking"),
        ("B2_L15_19__think35", "B2 auto direction, layers 15/19"),
        ("B2_tuned__think35", "B2 tuned (sweep winner: layers 9/13, alpha 12)"),
        ("B2_tuned32__think35", "B2 tuned at 32k (50 tries; layers 15/19, alpha 6, 256 cols)"),
        ("B2_tuned32__think35nudge", "B2 tuned at 32k + answer nudge on cut answers"),
        ("R0_base__budget4k", "Base, 4k thinking budget"),
        ("R0_base__budget8k", "Base, 8k thinking budget"),
        ("R0_base__budget16k", "Base, 16k thinking budget"),
        ("R0_base__budget4knudge", "Base, 4k budget + answer nudge"),
        ("R0_base__budget8knudge", "Base, 8k budget + answer nudge"),
        ("R0_base__budget16knudge", "Base, 16k budget + answer nudge"),
        ("R0_base__think35nudge@b", "Base, 32k + answer nudge on cut answers"),
        ("R1_astra_p50c__budget4knudge", "Released edit, 4k budget + answer nudge"),
        ("R1_astra_p50c__budget8knudge", "Released edit, 8k budget + answer nudge"),
        ("R1_astra_p50c__budget16k", "Released edit, 16k thinking budget"),
        ("R1_astra_p50c__budget16knudge", "Released edit, 16k budget + answer nudge"),
        ("R1_astra_p50c__think35nudge@b", "Released edit, 32k + answer nudge on cut answers"),
        ("R0_base__nothink35", "Base, thinking off"),
        ("R1_astra_p50c__nothink35", "Released edit, thinking off"),
        # loop-triggered (triggered.py; base answers cut at a sustained loop, then:)
        ("TRIG_nudge_w300d0.6__think35@b", "Triggered: answer nudge at the loop"),
        ("TRIG_R0_base_w300d0.6__think35@b", "Triggered: base continues (control)"),
        ("TRIG_B2_tuned32_w300d0.6__think35@b", "Triggered: switch to B2 (no LLM)"),
        ("TRIG_R1_astra_p50c_w300d0.6__think35@b", "Triggered: switch to the released edit")]
PAIRS = [("TRIG_nudge_w300d0.6__think35@b", "R0_base__think35nudge@b"),
         ("TRIG_R0_base_w300d0.6__think35@b", "R0_base__think35nudge@b"),
         ("TRIG_B2_tuned32_w300d0.6__think35@b", "R0_base__think35nudge@b"),
         ("TRIG_B2_tuned32_w300d0.6__think35@b", "TRIG_R0_base_w300d0.6__think35@b"),
         ("TRIG_B2_tuned32_w300d0.6__think35@b", "B2_tuned32__think35nudge"),
         ("TRIG_R1_astra_p50c_w300d0.6__think35@b", "R0_base__think35nudge@b"),
         ("TRIG_R1_astra_p50c_w300d0.6__think35@b", "R1_astra_p50c__think35nudge@b"),
         ("R1_astra_p50c__think35nudge@b", "TRIG_B2_tuned32_w300d0.6__think35@b"),
("R1_astra_p50c__think35", "B2_L15_19__think35"),
         ("R1_astra_p50c__think35", "B2_tuned__think35"),
         ("R1_astra_p50c__think35", "R0_base__budget16k"),
         ("R1_astra_p50c__think35", "R0_base__budget4knudge"),
         ("R1_astra_p50c__think35", "R0_base__budget8knudge"),
         ("R1_astra_p50c__think35", "R0_base__budget16knudge"),
         ("R0_base__budget4knudge", "R0_base__nothink35"),
         ("R1_astra_p50c__budget4knudge", "R0_base__budget4knudge"),
         ("R1_astra_p50c__budget8knudge", "R0_base__budget8knudge"),
         ("R1_astra_p50c__budget16k", "R0_base__budget16k"),
         ("R1_astra_p50c__budget16knudge", "R0_base__budget16knudge"),
         ("R1_astra_p50c__think35nudge@b", "R0_base__think35nudge@b"),
         ("R1_astra_p50c__think35nudge@b", "R0_base__budget16knudge"),
         ("R1_astra_p50c__think35", "B2_tuned32__think35"),
         ("R1_astra_p50c__think35nudge@b", "B2_tuned32__think35nudge"),
         ("R1_astra_p50c__think35", "R0_base__nothink35"),
         ("R1_astra_p50c__think35", "R1_astra_p50c__nothink35"),
         ("B2_L15_19__think35", "R0_base__nothink35")]
G_ARMS = [("R0_base__think35", "Base, thinking (32k)"),
          ("R1_astra_p50c__think35", "Released edit, thinking (32k)"),
          ("R0_base__think35nudge", "Base, 32k + answer nudge on cut answers"),
          ("R1_astra_p50c__think35nudge", "Released edit, 32k + answer nudge on cut answers"),
          ("R0_base__budget8k", "Base, 8k thinking budget"),
          ("R1_astra_p50c__budget8k", "Released edit, 8k thinking budget"),
          ("R0_base__budget8knudge", "Base, 8k budget + answer nudge"),
          ("R1_astra_p50c__budget8knudge", "Released edit, 8k budget + answer nudge"),
          ("R0_base__budget16k", "Base, 16k thinking budget"),
          ("R1_astra_p50c__budget16k", "Released edit, 16k thinking budget"),
          ("R0_base__budget16knudge", "Base, 16k budget + answer nudge"),
          ("R1_astra_p50c__budget16knudge", "Released edit, 16k budget + answer nudge"),
          ("R0_base__nothink35", "Base, thinking off")]
G_PAIRS = [("R1_astra_p50c__think35", "R0_base__think35"),
           ("R1_astra_p50c__think35nudge", "R0_base__think35nudge"),
           ("R1_astra_p50c__budget8k", "R0_base__budget8k"),
           ("R1_astra_p50c__budget8knudge", "R0_base__budget8knudge"),
           ("R1_astra_p50c__budget16k", "R0_base__budget16k"),
           ("R1_astra_p50c__budget16knudge", "R0_base__budget16knudge"),
           ("R1_astra_p50c__think35", "R0_base__budget8knudge"),
           ("R1_astra_p50c__think35", "R0_base__nothink35"),
           ("R0_base__budget8knudge", "R0_base__nothink35")]
# MMLU-Pro side effects: 1,000 fixed questions x 1 sample (thinking off x 2), graded by option letter (grade.py)
M_ARMS = [("R0_base__nothink35", "Base, thinking off"),
          ("R1_astra_p50c__nothink35", "Released edit, thinking off"),
          ("R0_base__budget8k", "Base, 8k thinking budget"),
          ("R1_astra_p50c__budget8k", "Released edit, 8k thinking budget"),
          ("R0_base__budget8knudge", "Base, 8k budget + answer nudge"),
          ("R1_astra_p50c__budget8knudge", "Released edit, 8k budget + answer nudge"),
          ("R0_base__think35", "Base, thinking (32k)"),
          ("R1_astra_p50c__think35", "Released edit, thinking (32k)"),
          ("R0_base__think35nudge", "Base, 32k + answer nudge on cut answers"),
          ("R1_astra_p50c__think35nudge", "Released edit, 32k + answer nudge on cut answers")]
M_PAIRS = [("R1_astra_p50c__nothink35", "R0_base__nothink35"),
           ("R1_astra_p50c__budget8k", "R0_base__budget8k"),
           ("R1_astra_p50c__budget8knudge", "R0_base__budget8knudge"),
           ("R1_astra_p50c__think35", "R0_base__think35"),
           ("R1_astra_p50c__think35nudge", "R0_base__think35nudge"),
           ("R1_astra_p50c__think35nudge", "R0_base__nothink35"),
           ("R0_base__think35nudge", "R0_base__nothink35")]
# transfer study (job_transfer.sh): a model with no LLM edit, 500 x 2, B2 = the automated recipe (no LLM)
T_ARMS = [("R0_base__nothink35", "Base, thinking off"),
          ("R0_base__think35", "Base, thinking (32k)"), ("B2_tuned32__think35", "B2, thinking (32k)"),
          ("R0_base__think35nudge", "Base, 32k + answer nudge"), ("B2_tuned32__think35nudge", "B2, 32k + answer nudge"),
          ("R0_base__budget8k", "Base, 8k budget"), ("B2_tuned32__budget8k", "B2, 8k budget"),
          ("R0_base__budget8knudge", "Base, 8k + nudge"), ("B2_tuned32__budget8knudge", "B2, 8k + nudge"),
          ("R0_base__budget16k", "Base, 16k budget"), ("B2_tuned32__budget16k", "B2, 16k budget"),
          ("R0_base__budget16knudge", "Base, 16k + nudge"), ("B2_tuned32__budget16knudge", "B2, 16k + nudge")]
T_PAIRS = [(f"B2_tuned32__{x}", f"R0_base__{x}") for x in
           ["think35", "think35nudge", "budget8k", "budget8knudge", "budget16k", "budget16knudge"]] +           [("R0_base__think35nudge", "R0_base__nothink35"), ("B2_tuned32__think35nudge", "R0_base__nothink35")]
SUITES = {"math": (ARMS, PAIRS, "__full500x4", "full500x4_compare.json"),
          "gsm8k": (G_ARMS, G_PAIRS, "__gsm8k", "gsm8k_compare.json"),
          "mmlupro": (M_ARMS, M_PAIRS, "__mmlupro", "mmlupro_compare.json"),
          # both samples per question, after merge_samples.py (-> *__mmlupro2.json)
          "mmlupro2": (M_ARMS, M_PAIRS, "__mmlupro2", "mmlupro2_compare.json"),
          "transfer": (T_ARMS, T_PAIRS, "__t500x2", "transfer_compare.json")}


def per_problem(d):
    return {p["id"]: sum(p["correct"]) / len(p["correct"]) for p in d["per_problem"]}


def main(rdir: Path, suite="math"):
    ARMS, PAIRS, tag, out_name = SUITES[suite]
    data = {}
    for k, _ in ARMS:
        f = rdir / (f"{k[:-2]}{tag}b.json" if k.endswith("@b") else f"{k}{tag}.json")  # @b: the rerun
        if f.exists():
            data[k] = json.loads(f.read_text())
    out = {"arms": {}, "pairs": []}
    base, edit = data.get("R0_base__think35"), data.get("R1_astra_p50c__think35")
    gain = edit["acc"] - base["acc"] if base and edit else None
    print(f"{'arm':36s} {'acc':>6s} {'cut':>6s} {'tokens':>7s} {'% of gain':>9s}")
    for k, label in ARMS:
        if k not in data:
            continue
        d = data[k]
        frac = (d["acc"] - base["acc"]) / gain if gain else None
        out["arms"][k] = {"label": label, "acc": d["acc"], "cut": d["cut_share"], "tok": d["mean_tokens"],
                          "n": d["n_problems"] * d["samples"], "frac_of_edit_gain": frac}
        fs = f"{100*frac:8.0f}%" if frac is not None else "        -"
        print(f"{label:36s} {100*d['acc']:6.1f} {100*d['cut_share']:5.1f}% {d['mean_tokens']:7.0f} {fs}")
    rng = random.Random(0)
    print(f"\n{'pair (A - B)':62s} {'diff':>6s} {'se':>5s} {'z':>5s} {'p':>7s}  95% CI")
    for a, b in PAIRS:
        if a not in data or b not in data:
            continue
        pa, pb = per_problem(data[a]), per_problem(data[b])
        ids = sorted(set(pa) & set(pb))
        diffs = [pa[i] - pb[i] for i in ids]
        n = len(diffs); m = sum(diffs) / n
        se = math.sqrt(sum((x - m) ** 2 for x in diffs) / (n - 1) / n)
        z = m / se if se else float("inf")
        p = math.erfc(abs(z) / math.sqrt(2))
        boots = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(10000))
        lo, hi = boots[249], boots[9749]
        la, lb = dict(ARMS)[a], dict(ARMS)[b]
        out["pairs"].append({"a": a, "b": b, "diff": m, "se": se, "z": z, "p": p, "ci95": [lo, hi], "n_problems": n})
        print(f"{la + ' - ' + lb:62s} {100*m:6.1f} {100*se:5.1f} {z:5.2f} {p:7.4f}  [{100*lo:.1f}, {100*hi:.1f}]")
    (rdir / out_name).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[sys.argv.index("--suite") + 1] if "--suite" in sys.argv else "math")
