"""Sanity check a model copy: greedy answer to one easy question from transformers and from vLLM, plus tensor checks.

Gibberish from both -> broken weights; gibberish from vLLM only -> a vLLM loading problem.
Usage: python diag_model.py <model_dir>
"""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

mdir = str(Path(sys.argv[1]).resolve())
Q = "What is 17 + 25? Answer in one short sentence."
with safe_open(f"{mdir}/model.safetensors", "pt") as f:
    keys = list(f.keys())
    print("tensors", len(keys), "| lm_head:", [k for k in keys if "lm_head" in k], "| embed:", [k for k in keys if "embed_tokens" in k])
    bad = [k for k in keys if not torch.isfinite(f.get_tensor(k).float()).all()]
    print("non-finite tensors:", bad[:5])

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(mdir)
prompt = tok.apply_chat_template([{"role": "user", "content": Q}], tokenize=False, add_generation_prompt=True,
                                 enable_thinking=False)
print("prompt tail:", repr(prompt[-120:]))

try:
    from transformers import AutoModelForImageTextToText as AM
except ImportError:
    from transformers import AutoModelForCausalLM as AM
m = AM.from_pretrained(mdir, dtype=torch.bfloat16).cuda().eval()
ids = tok(prompt, return_tensors="pt").to("cuda")
out = m.generate(**ids, max_new_tokens=40, do_sample=False)
print("HF  :", repr(tok.decode(out[0, ids["input_ids"].shape[1]:])))
del m; torch.cuda.empty_cache()

from vllm import LLM, SamplingParams
cfg = json.loads((Path(mdir) / "config.json").read_text())
mm = {"limit_mm_per_prompt": {"image": 0, "video": 0}} if "vision_config" in cfg else {}
for eager in (True, False):
    llm = LLM(model=mdir, dtype="bfloat16", gpu_memory_utilization=0.6, max_model_len=4096, enforce_eager=eager, **mm)
    o = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=40))[0].outputs[0].text
    print(f"vLLM (eager={eager}):", repr(o))
    del llm; torch.cuda.empty_cache()
