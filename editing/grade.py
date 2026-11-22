"""One grader for every benchmark row.

  math rows (default): the authors' mathqa.is_correct on the last \\boxed{} (math-verify), as before.
  rows with "kind": "mc" (multiple choice, e.g. MMLU-Pro): the last \\boxed{} must name the gold option letter.
     The box content is cleaned of \\text{}, \\textbf{}, \\mathrm{}, braces, parentheses and a trailing option text,
     and its first standalone letter A-J is the prediction ("\\boxed{C}", "\\boxed{(C)}", "\\boxed{\\text{C. 42}}").
"""
import re

_WRAP = re.compile(r"\\(?:text|textbf|mathrm|mathbf)\s*")


def letter(box):
    if not box:
        return None
    s = _WRAP.sub("", box).replace("{", " ").replace("}", " ").replace("(", " ").replace(")", " ")
    m = re.search(r"(?<![A-Za-z])([A-J])(?![A-Za-z])", s)
    return m.group(1) if m else None


def grade(Q, text, row):
    box = Q.extract(text)[0]
    if row.get("kind") == "mc":
        return int(letter(box) == row["answer"])
    return int(Q.is_correct(box, row["answer"]))


if __name__ == "__main__":
    for b, want in (("C", "C"), ("(C)", "C"), ("\\text{C}", "C"), ("\\textbf{C. 42}", "C"), ("C) 42", "C"),
                    ("42", None), ("", None), (None, None), ("\\text{Answer: B}", "B")):
        assert letter(b) == want, (b, letter(b), want)
    print("ok")
