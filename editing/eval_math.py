"""MATH-500 subset evaluation with vLLM, following the authors' evaluate.py protocols.

  think35   : thinking on, T=0.6, top_p 0.95, top_k 20, presence_penalty 1.5, 32,768 new tokens (the post's numbers)
  nothink35 : thinking off, the model card's non-thinking setting: T=0.7, top_p 0.8, top_k 20, presence 1.5, 8,192
  budget4k  : think35 for at most 4,096 tokens; unfinished answers get "</think>" appended and 2,048 more tokens
              (also budget8k, budget16k)
  think16k  : think35 with 16,384 new tokens (cheaper; scores the B2 sweep on practice problems)

Prompt and grading: the authors' mathqa.build_prompt / extract / is_correct (last \\boxed{}, math-verify).

Usage: python eval_math.py <root> <model_name> <protocol> [--subset data/math500_sub100.jsonl] [--n 1] [--tag T]
Writes <root>/results/<model_name>__<protocol><T>.json
"""
import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

SEED = int(os.environ.get("EVAL_SEED", 0))  # 0 = every earlier run; another value draws a fresh sample
P_THINK = dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, seed=SEED)
P_NOTHINK = dict(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, seed=SEED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("model"); ap.add_argument("protocol")
    ap.add_argument("--subset", default="data/math500_sub100.jsonl"); ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--max-problems", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix for the output name, e.g. __full500x4")
    a = ap.parse_args()
    root = Path(a.root)
    out = root / "results" / f"{a.model}__{a.protocol}{a.tag}.json"
    if out.exists() and not a.max_problems:
        print("exists:", out); return
    sys.path.insert(0, str(root))
    import mathqa as Q
    sys.path.insert(0, str(Path(__file__).parent))
    from grade import grade
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in open(root / a.subset)]
    if a.max_problems:
        rows = rows[:a.max_problems]
    mdir = str((root / "models" / a.model).resolve())
    tok = AutoTokenizer.from_pretrained(mdir)
    budget = int(a.protocol[6:-1]) * 1024 if a.protocol.startswith("budget") else 0
    think = a.protocol.startswith(("think", "budget"))
    max_tok = budget + 2048 if budget else {"think35": 32768, "think16k": 16384, "nothink35": 8192}[a.protocol]
    cfg = json.loads((Path(mdir) / "config.json").read_text())
    mm = {"limit_mm_per_prompt": {"image": 0, "video": 0}} if "vision_config" in cfg else {}
    llm = LLM(model=mdir, dtype="bfloat16", gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM", 0.85)),
              max_model_len=max_tok + 2048, seed=SEED, enforce_eager=os.environ.get("EVAL_EAGER") == "1", **mm)
    # EVAL_EAGER=1 skips torch.compile: concurrent jobs sharing the compile cache can crash at engine start
    prompts = [tok.apply_chat_template([{"role": "user", "content": Q.build_prompt(r["problem"])}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=think) for r in rows]
    t0 = time.time()
    if budget:
        o1 = llm.generate(prompts, SamplingParams(n=a.n, max_tokens=budget, **P_THINK))
        texts, cuts, lens = [], [], []
        follow, where = [], []
        for i, o in enumerate(o1):
            texts.append([]); cuts.append([]); lens.append([])
            for s in o.outputs:
                texts[i].append(s.text); cuts[i].append(False); lens[i].append(len(s.token_ids))
                if s.finish_reason == "length":
                    follow.append(prompts[i] + s.text + "\n</think>\n\n"); where.append((i, len(texts[i]) - 1))
        if follow:
            o2 = llm.generate(follow, SamplingParams(n=1, max_tokens=2048, **P_THINK))
            for (i, j), o in zip(where, o2):
                s = o.outputs[0]
                texts[i][j] += "\n</think>\n\n" + s.text
                cuts[i][j] = s.finish_reason == "length"
                lens[i][j] += len(s.token_ids)
    else:
        sp = SamplingParams(n=a.n, max_tokens=max_tok, **(P_THINK if think else P_NOTHINK))
        outs = llm.generate(prompts, sp)
        texts = [[s.text for s in o.outputs] for o in outs]
        cuts = [[s.finish_reason == "length" for s in o.outputs] for o in outs]
        lens = [[len(s.token_ids) for s in o.outputs] for o in outs]
    per, ok, tot = [], 0, 0
    for r, ts, cs, ls in zip(rows, texts, cuts, lens):
        c = [grade(Q, t, r) for t in ts]  # grade.py: math-verify, or the option letter for "kind": "mc" rows
        per.append({"id": r["id"], "correct": c, "cut": cs, "tokens": ls})
        ok += sum(c); tot += len(c)
    flat = lambda k: [x for p in per for x in p[k]]
    res = {"model": a.model, "protocol": a.protocol, "n_problems": len(rows), "samples": a.n, "acc": ok / tot,
           "cut_share": sum(flat("cut")) / tot, "mean_tokens": sum(flat("tokens")) / tot, "seconds": round(time.time() - t0),
           "per_problem": per}
    if a.max_problems:
        print(json.dumps({k: v for k, v in res.items() if k != "per_problem"})); return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    with open(out.with_suffix(".samples.jsonl"), "w") as f:
        for r, ts in list(zip(rows, texts))[:10]:
            f.write(json.dumps({"id": r["id"], "answer": r["answer"], "text": ts[0][-6000:]}) + "\n")
    # every answer in full, for auditing the grading (not copied into the repo: ~30-60 MB per 500 x 4 arm)
    with gzip.open(str(out.with_suffix("")) + ".all.jsonl.gz", "wt") as f:
        for r, ts, p in zip(rows, texts, per):
            f.write(json.dumps({"id": r["id"], "answer": r["answer"], "correct": p["correct"], "cut": p["cut"],
                                "texts": ts}) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "per_problem"}), flush=True)


if __name__ == "__main__":
    main()
