"""Collect the editing results into one table: accuracy, share cut at the token limit, mean tokens, fraction of Astra's gain.

Usage: python summarize.py <root>   -> prints a table and writes <root>/results/summary.json
(run compare_full.py <root>/results first to include the 500 x 4 confirmation run)
"""
import json
import math
import sys
from pathlib import Path

ORDER = [("R0_base", "think35", "Base, thinking (post's setting)"),
         ("R1_astra_p50c", "think35", "Astra's edit, thinking"),
         ("B1_random_dir", "think35", "Random direction, Astra's columns"),
         ("B2_auto", "think35", "Automated loop direction, auto layers"),
         ("B2_L15_19", "think35", "Automated loop direction, layers 15+19"),
         ("R0_base", "budget4k", "Base, 4k thinking budget then answer"),
         ("R0_base", "nothink35", "Base, thinking off (model card setting)"),
         ("R1_astra_p50c", "nothink35", "Astra's edit, thinking off")]


def main(root: Path):
    res = {}
    for m, p, label in ORDER:
        f = root / "results" / f"{m}__{p}.json"
        if f.exists():
            d = json.loads(f.read_text())
            res[(m, p)] = {"label": label, "acc": d["acc"], "cut": d["cut_share"], "tok": d["mean_tokens"],
                           "n": d["n_problems"] * d["samples"], "s": d["seconds"]}
    r0, r1 = res.get(("R0_base", "think35")), res.get(("R1_astra_p50c", "think35"))
    gain = (r1["acc"] - r0["acc"]) if r0 and r1 else None
    print(f"{'arm':44s} {'acc':>6s} {'±se':>5s} {'cut':>6s} {'tokens':>7s} {'% of gain':>9s}")
    for k, v in res.items():
        se = math.sqrt(v["acc"] * (1 - v["acc"]) / v["n"])
        frac = (v["acc"] - r0["acc"]) / gain if gain else float("nan")
        v["frac_of_astra_gain"] = frac
        print(f"{v['label']:44s} {100*v['acc']:6.1f} {100*se:5.1f} {100*v['cut']:5.1f}% {v['tok']:7.0f} {100*frac:8.0f}%")
    fit = root / "auto" / "fit.json"
    extra = json.loads(fit.read_text()) if fit.exists() else {}
    out = {"arms": {f"{m}__{p}": v for (m, p), v in res.items()}, "auto_fit": {k: extra.get(k) for k in ("top2_layers", "per_layer", "n_with_loops")}}
    # confirmation run (job_full.sh): all 500 problems x 4 samples, paired tests from compare_full.py
    cmp = root / "results" / "full500x4_compare.json"
    if cmp.exists():
        out["full500x4"] = json.loads(cmp.read_text())
    cmp = root / "results" / "gsm8k_compare.json"  # GSM8K, 1,319 x 2 (compare_full.py --suite gsm8k)
    if cmp.exists():
        out["gsm8k"] = json.loads(cmp.read_text())
    cmp = root / "results" / "mmlupro2_compare.json"  # MMLU-Pro side effects, 1,000 x 2 (--suite mmlupro2)
    if cmp.exists():
        out["mmlupro"] = json.loads(cmp.read_text())
    cmp = root / "transfer" / "qwen35_2b" / "results" / "transfer_compare.json"  # Qwen3.5-2B, 500 x 2 (--suite transfer)
    if cmp.exists():
        out["transfer_2b"] = json.loads(cmp.read_text())
    (root / "results" / "summary.json").write_text(json.dumps(out, indent=1))
    print("summary written")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
