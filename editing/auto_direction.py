"""B2: an automated, LLM-free version of Astra's edit (difference-of-means "loop direction" ablation).

  gen : the base model writes thinking-mode answers to practice problems (MATH train, never evaluated), 8,192 tokens max.
  fit : teacher-forced pass over those texts. Each generated token is labelled "in a loop" if the 32-token n-gram
        ending at it already occurred earlier in the same answer, otherwise "clean". For every layer's MLP
        (down_proj), compute
          u_L     = mean output on loop tokens - mean output on clean tokens (normalised)
          score_L = |difference| / mean output norm
          col_j   = (mean a_j on loop - mean a_j on clean) * (u_L . W[:, j])   (how much column j writes the difference)
        Build two models, both with Astra's alpha and 64 columns per layer, so only the choice of
        direction, layers and columns differs:
          B2_auto     : the 2 layers with the largest score_L, top-64 columns by col_j   (no LLM choices at all)
          B2_L15_19   : Astra's layers 15 and 19, automated u and columns
        Also reports the cosine between the automated u and Astra's u in layers 15 and 19.

Other models (transfer study, transfer_setup.py): sizes and layer count come from the weights. Without
astra_edit.json in <root> there is no LLM edit to compare with: alpha defaults to 6 (the LLM's value on Qwen3.5-0.8B)
and only B2_auto and dirs_all.json are written; sweep_b2.py does the tuning.

Usage: python auto_direction.py gen <root> [--n 160]      (vLLM, GPU)
       python auto_direction.py fit <root>                (transformers, GPU)
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch

NG = 32
P_THINK = dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, seed=0)


def gen(root: Path, n: int):
    sys.path.insert(0, str(root))
    import mathqa as Q
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    out = root / "auto" / "gen.jsonl"
    if out.exists():
        print("exists:", out); return
    rows = [json.loads(l) for l in open(root / "data" / "math_train.jsonl")]
    random.Random(0).shuffle(rows)
    rows = rows[:n]
    mdir = str((root / "models" / "R0_base").resolve())
    tok = AutoTokenizer.from_pretrained(mdir)
    llm = LLM(model=mdir, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=8192 + 2048, seed=0,
              **mm_kwargs(mdir))
    prompts = [tok.apply_chat_template([{"role": "user", "content": Q.build_prompt(r["problem"])}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True) for r in rows]
    outs = llm.generate(prompts, SamplingParams(n=1, max_tokens=8192, **P_THINK))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r, p, o in zip(rows, prompts, outs):
            s = o.outputs[0]
            f.write(json.dumps({"id": r["id"], "prompt_ids": list(o.prompt_token_ids), "gen_ids": list(s.token_ids),
                                "finish": s.finish_reason}) + "\n")
    print("generated", len(rows))


def mm_kwargs(mdir):
    cfg = json.loads((Path(mdir) / "config.json").read_text())
    return {"limit_mm_per_prompt": {"image": 0, "video": 0}} if "vision_config" in cfg else {}


def loop_mask(ids):
    seen, m = set(), []
    for t in range(len(ids)):
        if t + 1 < NG:
            m.append(False); continue
        g = tuple(ids[t + 1 - NG:t + 1])
        m.append(g in seen)
        seen.add(g)
    return m


def fit(root: Path):
    sys.path.insert(0, str(Path(__file__).parent))
    from prepare import apply_projection, down_key, n_layers, weights_file, write_variant
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    mdir = (root / "models" / "R0_base").resolve()
    try:
        model = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.bfloat16)
    except Exception:
        model = AutoModelForImageTextToText.from_pretrained(mdir, dtype=torch.bfloat16)
    model = model.cuda().eval()
    mlps = {}
    for name, mod in model.named_modules():
        if name.endswith("mlp.down_proj") and ".layers." in name and "visual" not in name:
            mlps[int(name.split(".layers.")[1].split(".")[0])] = mod
    base_sd = load_file(str(weights_file(mdir)))
    NL = n_layers(base_sd)
    assert sorted(mlps) == list(range(NL)), sorted(mlps)
    H, I = base_sd[down_key(base_sd, 0)].shape
    acc = {L: {"o_loop": torch.zeros(H, dtype=torch.float64, device="cuda"), "o_clean": torch.zeros(H, dtype=torch.float64, device="cuda"),
               "a_loop": torch.zeros(I, dtype=torch.float64, device="cuda"), "a_clean": torch.zeros(I, dtype=torch.float64, device="cuda"),
               "onorm": 0.0} for L in mlps}
    cnt = {"loop": 0, "clean": 0}
    cur = {}

    def hook(L):
        def f(mod, inp, out):
            a, o = inp[0][0].double(), out[0].double()
            ml, mc = cur["loop"], cur["clean"]
            acc[L]["o_loop"] += o[ml].sum(0); acc[L]["o_clean"] += o[mc].sum(0)
            acc[L]["a_loop"] += a[ml].sum(0); acc[L]["a_clean"] += a[mc].sum(0)
            acc[L]["onorm"] += float(o[ml | mc].norm(dim=1).sum())
        return f

    hs = [mod.register_forward_hook(hook(L)) for L, mod in mlps.items()]
    rows = [json.loads(l) for l in open(root / "auto" / "gen.jsonl")]
    n_loopers = 0
    with torch.no_grad():
        for r in rows:
            ids = r["prompt_ids"] + r["gen_ids"]
            lm = loop_mask(r["gen_ids"])
            if not any(lm) and r["finish"] == "length":
                continue  # cut without an exact repeat: ambiguous, skip
            n_loopers += any(lm)
            P = len(r["prompt_ids"])
            loop = torch.zeros(len(ids), dtype=torch.bool); clean = torch.zeros(len(ids), dtype=torch.bool)
            loop[P:] = torch.tensor(lm)
            first_loop = lm.index(True) if any(lm) else len(lm)
            clean[P:P + first_loop] = True  # tokens before the first repeat
            cur["loop"], cur["clean"] = loop.cuda(), clean.cuda()
            cnt["loop"] += int(loop.sum()); cnt["clean"] += int(clean.sum())
            model(input_ids=torch.tensor([ids], device="cuda"), logits_to_keep=1)  # skip the 248k-vocab logits
    for h in hs:
        h.remove()
    print(f"texts {len(rows)}, with loops {n_loopers}, loop tokens {cnt['loop']}, clean tokens {cnt['clean']}")

    af = root / "astra_edit.json"
    astra = json.loads(af.read_text())["edits"] if af.exists() else {}
    alpha = sum(e["alpha"] for e in astra.values()) / len(astra) if astra else 6.0
    stats, dirs, da_col = {}, {}, {}
    for L in mlps:
        d = acc[L]["o_loop"] / cnt["loop"] - acc[L]["o_clean"] / cnt["clean"]
        mean_norm = acc[L]["onorm"] / (cnt["loop"] + cnt["clean"])
        u = (d / d.norm()).cpu()
        W = base_sd[down_key(base_sd, L)].double()
        da = (acc[L]["a_loop"] / cnt["loop"] - acc[L]["a_clean"] / cnt["clean"]).cpu()
        col = da * (u @ W)
        da_col[L] = col
        J = torch.topk(col, 64).indices.sort().values
        stats[L] = {"score": float(d.norm() / mean_norm)}
        dirs[L] = {"u": u.tolist(), "J": J.tolist(), "alpha": alpha}
        if str(L) in astra:
            ua = torch.tensor(astra[str(L)]["u"], dtype=torch.float64)
            stats[L]["cos_with_astra_u"] = float(abs(u @ ua))
            stats[L]["col_overlap_with_astra"] = len(set(J.tolist()) & set(astra[str(L)]["J"]))
    # every layer's direction and column ranking (best first), for the budget-matched sweep in sweep_b2.py
    (root / "auto" / "dirs_all.json").write_text(json.dumps(
        {L: {"u": dirs[L]["u"], "rank": torch.argsort(da_col[L], descending=True)[:1024].tolist()} for L in mlps}))
    top2 = sorted(stats, key=lambda L: -stats[L]["score"])[:2]
    report = {"alpha_used": alpha, "top2_layers": top2, "per_layer": stats, "counts": cnt, "n_texts": len(rows),
              "n_with_loops": n_loopers}
    (root / "auto" / "fit.json").write_text(json.dumps(report, indent=1))
    # the exact edits applied in B2_auto and B2_L15_19 (direction u, columns J, alpha), so the models can be rebuilt
    # with prepare.apply_projection without re-generating or re-fitting
    variants = [("B2_auto", top2)] + ([("B2_L15_19", [15, 19])] if astra else [])
    (root / "auto" / "directions.json").write_text(json.dumps({n: {L: dirs[L] for L in ls} for n, ls in variants}))
    print(json.dumps({"top2": top2, "alpha": alpha, "L15": stats.get(15), "L19": stats.get(19)}))
    del model
    torch.cuda.empty_cache()
    for name, layers in variants:
        out = root / "models" / name
        if not out.exists():
            write_variant(mdir, out, apply_projection(base_sd, {L: dirs[L] for L in layers}),
                          {"kind": name, "layers": layers, "alpha": alpha})
    print("wrote B2 variants")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("root"); ap.add_argument("--n", type=int, default=160)
    a = ap.parse_args()
    (gen if a.cmd == "gen" else lambda r, n: fit(r))(Path(a.root), a.n)
