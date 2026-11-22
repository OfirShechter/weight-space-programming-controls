# Footprints and loops: re-testing weight-space programming

**Blog post: [ofirshechter.github.io/weight-space-programming-controls](https://ofirshechter.github.io/weight-space-programming-controls/)**

Code, prompts, every model answer and the results behind the post. It re-tests the two headline claims of
[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi et al., 2026):

1. **Reading:** an LLM names the game a small policy network was trained on from its raw weights.
   We show it reads a *footprint* of the training data (untouched first-layer rows that draw the game's layout),
   using two controls: one keeps the footprint but breaks the policy, the other keeps the policy but erases the
   footprint.
2. **Editing:** an LLM-written weight edit raises Qwen3.5-0.8B on MATH-500 from 7.9% to 48.1%.
   We show budget forcing recovers almost all of that gain without touching the weights, and that what the edit
   adds is getting out of repetition loops.

Every table in the post can be rebuilt on a CPU from the saved results, in about 20 minutes, with no GPU and no
LLM calls.

## Layout

| Folder | Contents |
|---|---|
| `reading/` | Control prompts, the four released networks, every LLM reader answer, the rule-based reader, leak checks |
| `reading/data/exp1_networks/` | The four policy networks from the original post, unchanged (CC BY 4.0, see `SOURCE.md`) |
| `reading/data/prompts/` | The prompt for each network and version (see [names](#names-in-file-names)), with `SHA256SUMS` |
| `reading/results/` | Every reader answer as JSON (Claude Opus 5.5, Claude Fable 5.2), scores, function and leak checks |
| `editing/` | MATH-500, GSM8K and MMLU-Pro evaluation (vLLM), budget forcing, the automated edit (B2), loop analysis |
| `editing/saved/` | Result JSONs and per-sample outputs behind every editing table |
| `docs/` | The blog post (`post.md`), its figures and the scripts that draw them; served by GitHub Pages |

## Quick start: rebuild the post's numbers (CPU)

Tested with Python 3.12.

```bash
pip install -r requirements.txt
```

**Part 1, reading** (run from `reading/`, a few seconds each):

```bash
cd reading
python -I exp1_footprint.py data/exp1_networks       # the footprint maps of the four released networks
python -I check_function_preserved.py data/prompts   # map only: ~chance agreement; footprint erased: 100%
python -I footprint_baseline.py data/prompts         # four-rule reader: 4/4 with the map, 0/4 erased
python -I compare_readers.py opus-5.5=results/full opus-5.5-c2b=results/opus_c2b fable-5.2=results/fable52 \
       --baseline results/footprint_baseline.json   # the reader table, from the saved answers
python -I score_runs.py results/fable52 results/fable52_scores.json   # per-game counts (Fable: Pong 9/10 erased)
python -I leak_span_check.py c2b_erased_mix          # attacking our own control: AUC 0.83-0.89
python -I leak_span_rules.py                         # ...but the recovered map names no game
```

**Part 2, editing** (run from `editing/`):

```bash
cd editing
python -I compare_full.py saved/results --suite math       # MATH-500 tables, paired 95% CIs (~8 min: bootstrap)
python -I compare_full.py saved/results --suite mmlupro2   # MMLU-Pro table (~3 min)
python -I compare_full.py saved/results --suite gsm8k      # GSM8K (not in the post; same pattern, ~5 min)
python -I loop_analysis.py saved                           # loop onset and escape rates (~5 min)
```

**Figures** (from the repository root):

```bash
python -I docs/make_fig_footprint_games.py   # docs/footprint_games.png
python -I docs/make_cover.py                 # docs/assets/cover.png (the link-preview image)
```

The scripts that save a file (`score_runs.py`, the leak checks, `compare_full.py`, `loop_analysis.py`, the
figures) overwrite the committed copy, and what they write is byte-identical to it.

### Where the numbers come from

| Post section | Output that has the numbers |
|---|---|
| The map / two controls: 19%, 18%, 22%, 12% and 100% agreement | `check_function_preserved.py` |
| Reader table (Opus, Fable, four rules) | `compare_readers.py`: columns `c0_original`, `c5_map_only`, `c2b_erased_mix` |
| Fable found the leak: Opus 10/20, Fable 15/16 | `compare_readers.py`: column `c2_erased` |
| Attacking our own control | `leak_span_check.py`, `leak_span_rules.py` |
| Missing baseline, what the edit adds, which part needed the LLM | `compare_full.py --suite math` |
| 90% vs 11% end in a loop | `loop_analysis.py` (`stuck@end`) |
| MMLU-Pro table | `compare_full.py --suite mmlupro2` |

The two "linear program" rows in the post's last editing table are the numbers reported by
[Barry](https://nathan.rs/posts/weight-space-lp/), not our runs (`lp_install.py` installs his edits for evaluation).

## Names in file names

**Reading versions** (`reading/data/prompts/<version>/<game>.txt`, the same names in `reading/results/`):

| Name | In the post | What it is |
|---|---|---|
| `c0_original` | Original | The authors' prompts, unchanged |
| `c1_permuted` | (sanity check) | Hidden units permuted with sign flips: the same function, different weights |
| `c5_map_only` | Map only | Row norms kept, everything else shuffled: the footprint stays, the policy is gone |
| `c2_erased` | First erased version | Untrained rows replaced by copies of trained rows (leaks through duplicates) |
| `c2b_erased_mix` | Footprint erased | Untrained rows replaced by mixes of three trained rows plus 5% noise |
| `c2c_erased_gauss` | (rejected) | Gaussian fill; only its leak check is kept (`results/leak_span_c2c_erased_gauss.json`) |

**Editing runs** (`editing/saved/results/<model>__<protocol><tag>.json`):

| Part | Name | Meaning |
|---|---|---|
| model | `R0_base` | Qwen3.5-0.8B, unedited |
| model | `R1_astra_p50c` | The released LLM edit (written by GPT-6 Astra; `p50c` is its run in the authors' logs) |
| model | `B1_random_dir` | Control: a random direction with the edit's columns and strength |
| model | `B2_*` | Our automated edit, no LLM. `B2_tuned32` is the one in the post (50 tries, scored at 32k) |
| model | `TRIG_<model>` | Base thinks until a loop starts, then `<model>` continues (`triggered.py`) |
| protocol | `think35` | The post's settings: thinking on, 32,768 tokens, cut answers graded wrong |
| protocol | `nothink35` | Thinking off |
| protocol | `budget4k`, `8k`, `16k` | Thinking stopped at the budget with a bare `</think>` |
| protocol | `...nudge` | Budget forcing: stopped with `</think>` + `Final Answer: \boxed{` |
| tag | (none) | 100-problem MATH-500 subset, 1 sample (first runs) |
| tag | `__full500x4`, `__full500x4b` | All 500 MATH-500 problems, 4 samples (`b`: rerun with every full answer saved) |
| tag | `__gsm8k`, `__mmlupro2` | GSM8K (1,319 × 2), MMLU-Pro (fixed 1,000 × 2) |

`editing/saved/answers/` has the full 32k-token answers behind the loop analysis (base, released edit, B2).

## Re-running the experiments

### LLM readers

The reads used the Claude Code CLI with all tools off and the authors' system prompt
(`reading/results/run_manifest.json` has the exact flags). Each read is saved as it finishes and finished reads are
skipped, so a run can be stopped and restarted:

```bash
cd reading
bash run_until_done.sh results/<new_dir> --model claude-opus-5-5 --effort high --samples 10
python -I run_queue.py data/prompts results/<new_dir> --backend openai --base-url <url> --model <name>  # other APIs
```

`make_control_prompts.py` and `make_footprint_erased_prompts.py --mix` rebuild the control prompts from the
original prompts and networks. `SHA256SUMS` checks the committed ones.

### Models (one GPU)

We used one 48 GB GPU (L40S) with bf16. The environment is pinned in `editing/requirements-lock.txt`, and the
models are downloaded from Hugging Face at pinned revisions (`prepare.py`). `<root>` is any working directory with
room for a few model copies (about 2 GB each):

```bash
cd editing
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-lock.txt
.venv/bin/python prepare.py <root>       # models, data and the authors' grader; recovers the edit; builds B1 (CPU)
PY=$PWD/.venv/bin/python bash job.sh <gpu> <root>
```

Every step of every job skips finished work, so a job can be re-run after an interruption. In the order we ran them:

| Job | Runs |
|---|---|
| `job.sh` | 100-problem subset: base and edit with thinking on and off, B1, fitting B2 |
| `job_full.sh` | All 500 MATH-500 problems × 4 for the main arms |
| `job_next.sh` | Overlap check, B2 sweep (`sweep_b2.py`), thinking budgets, reruns with every answer saved |
| `job_more.sh` | Budget forcing from the saved thinking (`budget_nudge.py`), B2 tuned at 32k |
| `job_edit_nudge.sh`, `job_base32_nudge.sh` | Budget forcing for the edit, and for base at 32k |
| `job_paper.sh <gpu> <root> p1`…`p5` | The edit at 16k, and GSM8K |
| `job_side.sh <gpu> <root> p6` / `p7` | MMLU-Pro, base / edit (two seeds merged with `merge_samples.py`) |
| `triggered.py <root> --mode continue --model <m>` | Switching models at the loop onset |
| `job_transfer.sh` | The automated recipe on other models (not in the post) |

Afterwards, `compare_full.py <root>/results` builds the tables as in the quick start.

## The blog post

`docs/` is a small [Jekyll](https://jekyllrb.com/) site that GitHub Pages builds on every push (Settings → Pages →
Deploy from a branch → `main`, `/docs`). `docs/post.md` is the post in plain Markdown, `docs/_layouts/` and
`docs/assets/css/` are its page template, and `docs/_config.yml` holds the site title and the optional byline.
To preview it locally with Ruby installed: `cd docs && bundle install && bundle exec jekyll serve`.

## Licences

Code: MIT ([`LICENSE`](LICENSE)). The policy networks and the original prompts come from
[RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs) under CC BY 4.0, and the control
prompts are derived from them. Models are downloaded from Hugging Face at pinned revisions, not redistributed.

## Citation

A paper with the full details is coming to arXiv. A citation entry will be added here when it is posted.
