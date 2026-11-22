"""Set up a root for the transfer study: the automated recipe (B2) on another loop-prone model, with no LLM edit.

Creates <new_root> with the main root's evaluation data and grader (links, so the problems are identical), downloads
the model at a pinned revision, and writes it to <new_root>/models/R0_base as a single model.safetensors (shards
merged, so prepare.weights_file / write_variant work unchanged). The revision is recorded in <new_root>/model.json.

Usage: python transfer_setup.py <main_root> <new_root> <hf_repo> <revision>
"""
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

DATA = ["math500_sub500.jsonl", "math500_sub100.jsonl", "math_train.jsonl", "gsm8k_test.jsonl"]


def main(main_root: Path, root: Path, repo: str, rev: str):
    for d in ("data", "models", "results", "auto", "logs"):
        (root / d).mkdir(parents=True, exist_ok=True)
    for f in DATA:
        if (main_root / "data" / f).exists() and not (root / "data" / f).exists():
            (root / "data" / f).symlink_to((main_root / "data" / f).resolve())
    for f in ("mathqa.py", "mcqa.py"):
        shutil.copy(main_root / f, root / f)
    out = root / "models" / "R0_base"
    if (out / "model.json").exists():
        print("exists:", out); return
    src = Path(snapshot_download(repo, revision=rev))
    out.mkdir(exist_ok=True)
    sd = {}
    for p in sorted(src.iterdir()):
        if p.name.endswith(".safetensors"):
            sd.update(load_file(str(p)))
        elif p.is_file() and not p.name.endswith(".index.json"):
            shutil.copy(p, out / p.name)
    save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    info = {"repo": repo, "revision": rev, "n_tensors": len(sd)}
    (out / "model.json").write_text(json.dumps(info, indent=1))  # written last: marks a complete copy
    (root / "model.json").write_text(json.dumps(info, indent=1))
    print("set up", root, json.dumps(info))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4])
