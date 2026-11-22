"""Thinking budget with the standard answer nudge ("budget forcing"), reusing the thinking already generated.

eval_math.py's budget protocols stop thinking at N tokens by appending a bare "</think>"; the model then usually
restarts the solution and runs out of its 2,048 answer tokens. Here every sample that hit the budget is continued
from the same thinking with "\\n</think>\\n\\nFinal Answer: \\boxed{" and 64 tokens. Samples that finished within the
budget keep their original answer. No new thinking is sampled, so the comparison with the bare version is paired.

Protocol think35 (no budget): the samples cut at 32k get the same nudge (after "</think>" only if thinking had not ended).

Usage: python budget_nudge.py <root> budget8k [--model R0_base] [--tag __full500x4] [--subset data/math500_sub500.jsonl]
Reads  results/<model>__<protocol><tag>.json and .all.jsonl.gz; writes results/<model>__<protocol>nudge<tag>.json
"""
import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

NUDGE = "\n</think>\n\nFinal Answer: \\boxed{"
MARK = "\n</think>\n\n"  # what eval_math.py appended at the budget
SEED = int(os.environ.get("EVAL_SEED", 0))  # as in eval_math.py
P = dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, seed=SEED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("protocol"); ap.add_argument("--model", default="R0_base")
    ap.add_argument("--tag", default="__full500x4"); ap.add_argument("--subset", default="data/math500_sub500.jsonl")
    a = ap.parse_args()
    root = Path(a.root); res = root / "results"
    src = res / f"{a.model}__{a.protocol}{a.tag}.json"
    out = res / f"{a.model}__{a.protocol}nudge{a.tag}.json"
    if out.exists():
        print("exists:", out); return
    full = a.protocol == "think35"
    budget = 32768 if full else int(a.protocol[6:-1]) * 1024
    sys.path.insert(0, str(root))
    import mathqa as Q
    sys.path.insert(0, str(Path(__file__).parent))
    from grade import grade
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    meta = {p["id"]: p for p in json.loads(src.read_text())["per_problem"]}
    texts = {r["id"]: r["texts"] for r in map(json.loads, gzip.open(str(src.with_suffix("")) + ".all.jsonl.gz", "rt"))}
    rows = [json.loads(l) for l in open(root / a.subset)]
    mdir = str((root / "models" / a.model).resolve())
    tok = AutoTokenizer.from_pretrained(mdir)
    prompts = {r["id"]: tok.apply_chat_template([{"role": "user", "content": Q.build_prompt(r["problem"])}], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=True) for r in rows}
    jobs, where = [], []
    for r in rows:
        for j, (t, n, c) in enumerate(zip(texts[r["id"]], meta[r["id"]]["tokens"], meta[r["id"]]["cut"])):
            if full and c:  # cut at 32k: nudge from where it stopped
                think, nud = t, (NUDGE if "</think>" not in t else "\n\nFinal Answer: \\boxed{")
            elif not full and n > budget:  # hit the budget: eval_math appended MARK after exactly `budget` tokens
                think, nud = t[:t.index(MARK)], NUDGE
            else:
                continue
            jobs.append(prompts[r["id"]] + think + nud); where.append((r["id"], j, think + nud, n))
    # eager (no torch.compile / CUDA graphs): only 64 tokens are generated, and a compiled graph cached by an earlier
    # eval_math run at the same length crashed engine start-up for the edit (dynamo: NoneType has no .size)
    cfg = json.loads((Path(mdir) / "config.json").read_text())
    mm = {"limit_mm_per_prompt": {"image": 0, "video": 0}} if "vision_config" in cfg else {}
    llm = LLM(model=mdir, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=budget + 8192, seed=SEED,  # 4096 was 1 token short once
              enforce_eager=True, **mm)
    t0 = time.time()
    outs = llm.generate(jobs, SamplingParams(n=1, max_tokens=64, **P))
    new = {}
    for (pid, j, pre, n), o in zip(where, outs):
        new[(pid, j)] = (pre + o.outputs[0].text, (n if full else budget) + len(o.outputs[0].token_ids))
    per, ok, tot = [], 0, 0
    for r in rows:
        ts, ls = list(texts[r["id"]]), list(meta[r["id"]]["tokens"])
        for j in range(len(ts)):
            if (r["id"], j) in new:
                ts[j], ls[j] = new[(r["id"], j)]
        c = [grade(Q, t, r) for t in ts]  # grade.py: math-verify, or the option letter for "kind": "mc" rows
        per.append({"id": r["id"], "correct": c, "cut": [False] * len(c), "tokens": ls})
        ok += sum(c); tot += len(c)
    flat = [x for p in per for x in p["tokens"]]
    d = {"model": a.model, "protocol": a.protocol + "nudge", "n_problems": len(rows), "samples": len(per[0]["correct"]),
         "acc": ok / tot, "cut_share": 0.0, "mean_tokens": sum(flat) / tot, "seconds": round(time.time() - t0),
         "n_nudged": len(jobs), "per_problem": per}
    out.write_text(json.dumps(d))
    print(json.dumps({k: v for k, v in d.items() if k != "per_problem"}), flush=True)


if __name__ == "__main__":
    main()
