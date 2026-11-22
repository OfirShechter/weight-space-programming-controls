"""Stand-in for the authors' weightenv/lmo/tasks/mcqa.py, which the release does not include.

The release's mathqa.py imports `mcqa.loop_signals` at module load, and uses it only in mathqa.loop_signals (the
'repetitive_end' diagnostic). Our evaluation uses mathqa.build_prompt / extract / is_correct, which do not touch it.
This file is installed as <root>/mcqa.py by prepare.py so that mathqa imports.

repetitive_end follows the rule stated in mathqa.loop_signals' docstring: in the last 200 words, the 4-word positions
whose phrase occurs 3+ times there are at least half of all positions.
"""
from collections import Counter

WINDOW, NGRAM, MIN_COUNT, MIN_FRAC = 200, 4, 3, 0.5


def loop_signals(text):
    w = text.split()[-WINDOW:]
    grams = [tuple(w[i:i + NGRAM]) for i in range(len(w) - NGRAM + 1)]
    if not grams:
        return {"repetitive_end": False}
    c = Counter(grams)
    rep = sum(1 for g in grams if c[g] >= MIN_COUNT)
    return {"repetitive_end": rep >= MIN_FRAC * len(grams)}
