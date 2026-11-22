"""CPU-only preparation for the editing experiments (what did the LLM add to the Qwen3.5-0.8B math edit?).

Downloads (pinned revisions):
  base   Qwen/Qwen3.5-0.8B @ 2fc06364715b967f1860aea9cf38778875588b17   (the edit was made on this one)
  edited RikiyaT/Qwen3.5-0.8B-weight-space-programming @ d1671a17...      (Astra's p50c edit, hash-verified upstream)
  MATH-500 (HuggingFaceH4/MATH-500) -> a fixed random subset; MATH train practice problems (from the release)
  mathqa.py from the release (the authors' prompt builder and grader)

Then recovers Astra's edit from the weight difference. In layers 15 and 19 it is
W[:, J] <- W[:, J] - alpha * u (u^T W[:, J]) on 64 down_proj columns J. The script builds the random-direction control (B1).

Usage: python prepare.py <root> [--n 100]
"""
import argparse
import json
import os
import re
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

BASE = ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17")
EDIT = ("RikiyaT/Qwen3.5-0.8B-weight-space-programming", "d1671a178705bcab7644806d23c535517f35723b")
REL = "https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp8_math"
NAME = "model.language_model.layers.{}.mlp.down_proj.weight"


def snapshot(repo, rev):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo, revision=rev))


def weights_file(d: Path) -> Path:
    return next(p for p in d.iterdir() if p.name.endswith(".safetensors"))


def recover_edit(base_sd, edit_sd):
    """Return {layer: dict(J, u, alpha, resid)} for every down_proj tensor that differs."""
    out = {}
    for L in range(24):
        n = NAME.format(L)
        if n not in base_sd or torch.equal(base_sd[n], edit_sd[n]):
            continue
        W, We = base_sd[n].double(), edit_sd[n].double()
        D = We - W
        J = torch.nonzero(D.abs().sum(0) > 0).flatten()
        U, S, _ = torch.linalg.svd(D[:, J], full_matrices=False)
        u = U[:, 0]
        P = torch.outer(u, u @ W[:, J])
        alpha = float(-(D[:, J] * P).sum() / (P * P).sum())
        if alpha < 0:
            u, alpha = -u, -alpha  # sign of u is arbitrary; keep alpha > 0 (the projection is invariant)
            P = torch.outer(u, u @ W[:, J]); alpha = float(-(D[:, J] * P).sum() / (P * P).sum())
        resid = float(torch.linalg.norm(D[:, J] + alpha * P) / torch.linalg.norm(D[:, J]))
        out[L] = {"J": J.tolist(), "u": u.float().tolist(), "alpha": alpha, "rank1_resid": resid,
                  "sv_ratio": float(S[1] / S[0]) if len(S) > 1 else 0.0}
    return out


def down_key(sd, L):
    """The layer-L MLP down_proj weight name in any decoder-only checkpoint (Qwen3.5 nests it under language_model;
    Qwen3 and DeepSeek-R1-Distill do not). Vision and multi-token-prediction blocks never match."""
    pat = re.compile(rf"^model\.(language_model\.)?layers\.{L}\.mlp\.down_proj\.weight$")
    ks = [k for k in sd if pat.match(k)]
    assert len(ks) == 1, (L, ks)
    return ks[0]


def n_layers(sd):
    pat = re.compile(r"^model\.(language_model\.)?layers\.(\d+)\.mlp\.down_proj\.weight$")
    return 1 + max(int(m.group(2)) for k in sd if (m := pat.match(k)))


def apply_projection(sd, edits):
    sd = dict(sd)
    for L, e in edits.items():
        n = down_key(sd, L)
        W = sd[n].double()
        J = torch.tensor(e["J"])
        u = torch.tensor(e["u"], dtype=torch.float64)
        u = u / u.norm()
        W[:, J] = W[:, J] - e["alpha"] * torch.outer(u, u @ W[:, J])
        sd[n] = W.to(torch.bfloat16)
    return sd


def write_variant(base_dir: Path, out_dir: Path, sd, info):
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in base_dir.iterdir():
        if not p.name.endswith(".safetensors") and p.is_file():
            shutil.copy(p, out_dir / p.name)
    save_file(sd, str(out_dir / weights_file(base_dir).name), metadata={"format": "pt"})
    (out_dir / "variant.json").write_text(json.dumps(info, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    root = Path(a.root); (root / "data").mkdir(parents=True, exist_ok=True); (root / "models").mkdir(exist_ok=True)

    # authors' grader
    urllib.request.urlretrieve(f"{REL}/code/mathqa.py", root / "mathqa.py")
    # mathqa imports mcqa (not in the release); install the stand-in next to it
    shutil.copy(Path(__file__).with_name("mcqa_shim.py"), root / "mcqa.py")
    # MATH-500 subset
    from datasets import load_dataset
    rows = [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"]}
            for r in load_dataset("HuggingFaceH4/MATH-500", split="test")]
    idx = np.random.default_rng(0).permutation(len(rows))[:a.n]
    with open(root / "data" / f"math500_sub{a.n}.jsonl", "w") as f:
        for i in sorted(idx):
            f.write(json.dumps(rows[i]) + "\n")
    # practice problems for the automated direction (never evaluated)
    urllib.request.urlretrieve(f"{REL}/practice/math_train.jsonl", root / "data" / "math_train.jsonl")

    base_dir, edit_dir = snapshot(*BASE), snapshot(*EDIT)
    for name, d in (("R0_base", base_dir), ("R1_astra_p50c", edit_dir)):
        link = root / "models" / name
        if not link.exists():
            link.symlink_to(d)
    base_sd = load_file(str(weights_file(base_dir)))
    edit_sd = load_file(str(weights_file(edit_dir)))
    diff = [k for k in base_sd if not torch.equal(base_sd[k], edit_sd[k])]
    print("tensors that differ:", diff)
    edits = recover_edit(base_sd, edit_sd)
    us = {L: torch.tensor(e["u"]) for L, e in edits.items()}
    Ls = sorted(us)
    cos = float(us[Ls[0]] @ us[Ls[-1]]) if len(Ls) > 1 else 1.0
    summary = {L: {k: e[k] for k in ("alpha", "rank1_resid", "sv_ratio")} | {"n_cols": len(e["J"])} for L, e in edits.items()}
    print("recovered edit:", json.dumps(summary), "cos(u15,u19) =", round(cos, 4))
    (root / "astra_edit.json").write_text(json.dumps({"edits": edits, "cos_between_layers": cos}))

    # check: re-applying the recovered projection reproduces the edited tensors (up to bf16 rounding)
    re_sd = apply_projection(base_sd, edits)
    for L in edits:
        n = NAME.format(L)
        err = (re_sd[n].float() - edit_sd[n].float()).abs().max().item()
        print(f"layer {L}: max |rebuilt - edited| = {err:.2e}")

    # B1: random unit direction, same columns and alpha
    g = torch.Generator().manual_seed(0)
    b1 = {}
    for L, e in edits.items():
        r = torch.randn(len(e["u"]), generator=g, dtype=torch.float64)
        b1[L] = {"J": e["J"], "u": (r / r.norm()).tolist(), "alpha": e["alpha"]}
    out = root / "models" / "B1_random_dir"
    if not out.exists():
        write_variant(base_dir, out, apply_projection(base_sd, b1), {"kind": "random direction, Astra's columns and alpha"})
    print("prepared", root)


if __name__ == "__main__":
    main()
