"""Loop-triggered intervention: act only once a loop has started, reusing saved base answers.

For every saved base answer (eval_math.py think35, *.all.jsonl.gz) find the loop onset: the start of the first
20-word span that repeats an earlier one (the rule of loop_analysis.py), if it lies inside the thinking. Answers with
no onset keep the base answer (+ the answer nudge if cut at 32k, as in think35nudge). Answers with an onset are cut
there and then, by --mode:
  nudge    : force the answer at once ("</think> Final Answer: \\boxed{", 64 tokens, base model)
  continue : continue thinking with --model (B2_tuned32: no LLM; R1_astra_p50c: the released edit, upper reference;
             R0_base: control, a fresh continuation without any switch) up to the same 32,768-token total, then the
             answer nudge if still cut
Samples are paired with the base run, so compare_full.py can test against base + nudge directly.

Usage: python triggered.py <root> --mode continue --model B2_tuned32 [--src R0_base__think35__full500x4b]
         [--subset data/math500_sub500.jsonl] [--max-problems 20]
Writes results/TRIG_<mode or model>__think35<tag>.json (+ .all.jsonl.gz), where <tag> is the source's tag.
"""
import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

N = 20  # words per span, as loop_analysis.py
TOTAL = 32768
NUDGE = "\n</think>\n\nFinal Answer: \\boxed{"
SEED = int(os.environ.get("EVAL_SEED", 0))
P = dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5)


W = int(os.environ.get("TRIG_WINDOW", 300))      # sustained loop: of the last W spans ...
DENS = float(os.environ.get("TRIG_DENSITY", 0.6))  # ... at least this share repeat an earlier span (W=1: first repeat)


def onset_char(text):
    """Character offset where a sustained loop starts, or None.

    A span = N consecutive words; it repeats if the same N words appeared earlier. The loop is detected at the first
    span i where at least DENS of the last W spans repeat; it is cut at the first repeating span inside that window.
    (One repeated 20-word span, W=1, fired on 97.5% of base answers at a median of 3.5k tokens: too eager.)"""
    words, offs, pos = text.split(), [], 0
    for w in words:
        pos = text.find(w, pos); offs.append(pos); pos += len(w)
    seen, rep, run = set(), [], 0
    for i in range(len(words) - N + 1):
        g = hash(tuple(words[i:i + N]))
        rep.append(g in seen); seen.add(g)
        run += rep[-1] - (rep[-1 - W] if len(rep) > W else 0)
        if len(rep) >= W and run >= DENS * W:
            return offs[next(k for k in range(i - W + 1, i + 1) if rep[k])]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--mode", choices=["nudge", "continue"], required=True)
    ap.add_argument("--model", default="R0_base"); ap.add_argument("--src", default="R0_base__think35__full500x4b")
    ap.add_argument("--subset", default="data/math500_sub500.jsonl"); ap.add_argument("--max-problems", type=int, default=0)
    a = ap.parse_args()
    root = Path(a.root); res = root / "results"
    tag = a.src.split("think35", 1)[1]
    arm = "nudge" if a.mode == "nudge" else a.model
    out = res / f"TRIG_{arm}_w{W}d{DENS:g}__think35{tag}{'_smoke' if a.max_problems else ''}.json"
    if out.exists():
        print("exists:", out); return
    sys.path.insert(0, str(root)); import mathqa as Q
    sys.path.insert(0, str(Path(__file__).parent)); from grade import grade
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    meta = {p["id"]: p for p in json.loads((res / f"{a.src}.json").read_text())["per_problem"]}
    texts = {r["id"]: r["texts"] for r in map(json.loads, gzip.open(str(res / a.src) + ".all.jsonl.gz", "rt"))}
    rows = [json.loads(l) for l in open(root / a.subset)]
    rows = rows[:a.max_problems] if a.max_problems else rows
    mname = "R0_base" if a.mode == "nudge" else a.model
    mdir = str((root / "models" / mname).resolve())
    tok = AutoTokenizer.from_pretrained(mdir)
    prompts = {r["id"]: tok.apply_chat_template([{"role": "user", "content": Q.build_prompt(r["problem"])}], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=True) for r in rows}
    final, ntok, info = {}, {}, {}  # (id, j) -> text / tokens / onset
    jobs = []  # (key, prefix text after the prompt, tokens used, kind)
    for r in rows:
        for j, (t, n, c) in enumerate(zip(texts[r["id"]], meta[r["id"]]["tokens"], meta[r["id"]]["cut"])):
            k = (r["id"], j)
            th = t.split("</think>")[0] if "</think>" in t else t
            oc = onset_char(th)
            if oc is None:  # no loop in the thinking: base answer, nudged if cut (= base + nudge)
                final[k], ntok[k], info[k] = t, n, None
                if c:
                    jobs.append((k, t, n, "nudge"))
                continue
            pre = t[:oc]
            used = len(tok(pre, add_special_tokens=False)["input_ids"])
            info[k] = used
            jobs.append((k, pre, used, "nudge" if a.mode == "nudge" else "continue"))
    llm = LLM(model=mdir, dtype="bfloat16", gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM", 0.85)),
              max_model_len=TOTAL + 8192, seed=SEED, enable_prefix_caching=True)
    t0 = time.time()

    def nudge_all(items):  # items: (key, text, tokens) -> answer forced with 64 tokens
        if not items:
            return
        ps = [prompts[k[0]] + t + (NUDGE if "</think>" not in t else "\n\nFinal Answer: \\boxed{") for k, t, _ in items]
        outs = llm.generate(ps, [SamplingParams(n=1, max_tokens=64, seed=SEED + i, **P) for i in range(len(ps))])
        for (k, t, n), p, o in zip(items, ps, outs):
            final[k] = p[len(prompts[k[0]]):] + o.outputs[0].text; ntok[k] = n + len(o.outputs[0].token_ids)

    cont = [(k, t, n) for k, t, n, kind in jobs if kind == "continue"]
    still = []
    if cont:
        outs = llm.generate([prompts[k[0]] + t for k, t, _ in cont],
                            [SamplingParams(n=1, max_tokens=TOTAL - n, seed=SEED + i, **P) for i, (_, _, n) in enumerate(cont)])
        for (k, t, n), o in zip(cont, outs):
            s = o.outputs[0]; final[k] = t + s.text; ntok[k] = n + len(s.token_ids)
            if s.finish_reason == "length":
                still.append((k, final[k], ntok[k]))
    nudge_all([(k, t, n) for k, t, n, kind in jobs if kind == "nudge"] + still)

    per, ok, tot = [], 0, 0
    for r in rows:
        js = range(len(texts[r["id"]]))
        c = [grade(Q, final[(r["id"], j)], r) for j in js]
        per.append({"id": r["id"], "correct": c, "cut": [False] * len(c), "tokens": [ntok[(r["id"], j)] for j in js],
                    "onset": [info[(r["id"], j)] for j in js]})
        ok += sum(c); tot += len(c)
    on = [x for p in per for x in p["onset"] if x is not None]
    d = {"model": arm, "protocol": "think35trig", "window": W, "density": DENS, "src": a.src, "mode": a.mode, "n_problems": len(rows),
         "samples": len(per[0]["correct"]), "acc": ok / tot, "cut_share": 0.0,
         "mean_tokens": sum(x for p in per for x in p["tokens"]) / tot, "seconds": round(time.time() - t0),
         "triggered": len(on) / tot, "median_onset": sorted(on)[len(on) // 2] if on else None,
         "n_still_cut": len(still), "per_problem": per}
    out.write_text(json.dumps(d))
    with gzip.open(str(out.with_suffix("")) + ".all.jsonl.gz", "wt") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "texts": [final[(r["id"], j)] for j in range(len(texts[r["id"]]))]}) + "\n")
    print(json.dumps({k: v for k, v in d.items() if k != "per_problem"}), flush=True)


if __name__ == "__main__":
    main()
