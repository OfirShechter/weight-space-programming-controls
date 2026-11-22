"""Install Barry's rebuilt LP edits (weight-space-lp) as model dirs for eval_math.py / budget_nudge.py / triggered.py.

  <wslp>/edits/shift_k8.safetensors      -> <root>/models/LP_astra   (LP columns, Astra's direction)
  <wslp>/edits/realshift_k8.safetensors  -> <root>/models/LP_heur    (LP columns, heuristic direction)

Each edit file holds the full bf16 down_proj tensors of layers 15 and 19; they replace the base tensors, everything
else is copied from <root>/models/R0_base. Prints the rebuilt edit's size next to Barry's committed metadata
(<wslp>/edits_ref/*.json) as a check that the rebuild matches his.
Usage: python lp_install.py <root> <wslp>
"""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent))
from prepare import down_key, weights_file, write_variant  # noqa: E402

root, wslp = Path(sys.argv[1]), Path(sys.argv[2])
base_dir = (root / "models" / "R0_base").resolve()
base_sd = load_file(str(weights_file(base_dir)))
for edit, name in (("shift_k8", "LP_astra"), ("realshift_k8", "LP_heur")):
    new = load_file(str(wslp / "edits" / f"{edit}.safetensors"))
    sd, rep = dict(base_sd), {}
    for k, v in new.items():
        L = int(k.split(".layers.")[1].split(".")[0])
        n = down_key(sd, L)
        assert sd[n].shape == v.shape, (k, n)
        w0, w = sd[n], v.to(torch.bfloat16).contiguous()
        ch = (w != w0)
        rep[L] = {"changed_weights_bf16": int(ch.sum()), "changed_columns": int(ch.any(0).sum()),
                  "rel_frobenius": float((w.float() - w0.float()).norm() / w0.float().norm())}
        sd[n] = w
    write_variant(base_dir, root / "models" / name, sd, {"source": f"weight-space-lp {edit} (rebuilt)", "layers": rep})
    ref = json.load(open(wslp / "edits_ref" / f"{edit}.json"))["layers"]
    for L in sorted(rep):
        r = ref[str(L)]
        print(f"{name} L{L}: rebuilt {rep[L]['changed_weights_bf16']} weights, {rep[L]['changed_columns']} cols, "
              f"rel {rep[L]['rel_frobenius']:.4f} | Barry {r['changed_weights_bf16']} weights, "
              f"{r['changed_columns']} cols, rel {r['rel_frobenius']:.4f}")
