"""E0 — grader validation.

The plan requires hand-checking ~30 items per benchmark before any number is
trusted. This script (a) smoke-tests loading, (b) runs the grader against
adversarial answer formats that real agents actually produce, and (c) dumps a
sample for manual review.

Run:  python experiments/e0_validate_graders.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from puppeteer_stop.tasks import (  # noqa: E402
    extract_answer,
    grade,
    load_gsm_hard,
    load_mmlu_pro,
    render_question,
)

# Formats observed from chain-of-thought agents. Each is (text, kind, expected).
# These encode the failure modes worth guarding: thousands separators, incidental
# numbers mid-reasoning, floats vs ints, scientific notation, and no answer at all
# (which must stay None, not become a wrong answer).
NUMERIC_CASES = [
    ("The answer is 42", "42.0"),
    ("#### 1234567", "1234567.0"),
    ("Final answer: 1,234,567", "1234567.0"),
    ("\\boxed{-17.5}", "-17.5"),
    ("First 12 apples, then 30 more, so the total is 42.", "42.0"),
    ("We compute 3 * 4 = 12 and then subtract 2, giving 10", "10.0"),
    ("The result is $1,200.50", "1200.5"),
    ("Answer = 6.02e23", "6.02e+23"),
    ("I need to reconsider this problem entirely.", None),  # critique: no answer
    ("", None),
]

CHOICE_CASES = [
    ("The answer is C", "C"),
    ("Final answer: (B)", "B"),
    ("\\boxed{J}", "J"),
    ("Answer: D. Because the reaction is exothermic.", "D"),
    ("Let me think about options A and B... I choose B.", "B"),
    ("This requires more analysis before answering.", None),
]


def check_extraction() -> int:
    failures = 0
    print("== numeric extraction ==")
    for text, expected in NUMERIC_CASES:
        got = extract_answer(text, "numeric")
        ok = (got == expected) if expected is None or got is None else (
            abs(float(got) - float(expected)) <= 1e-9 * max(abs(float(expected)), 1.0)
        )
        if not ok:
            failures += 1
        print(f"  [{'ok ' if ok else 'FAIL'}] {text[:46]!r:50} -> {got!r} (want {expected!r})")

    print("== choice extraction ==")
    for text, expected in CHOICE_CASES:
        got = extract_answer(text, "multiple_choice")
        ok = got == expected
        if not ok:
            failures += 1
        print(f"  [{'ok ' if ok else 'FAIL'}] {text[:46]!r:50} -> {got!r} (want {expected!r})")
    return failures


def check_grading() -> int:
    failures = 0
    print("== grading tolerance ==")
    cases = [
        ("42.0", "42.0", "numeric", True),
        ("42.00001", "42.0", "numeric", True),      # rounding restatement
        ("43.0", "42.0", "numeric", False),
        ("1234567.0", "1234567.0", "numeric", True),
        (None, "42.0", "numeric", False),           # no answer is not correct
        ("C", "C", "multiple_choice", True),
        ("c", "C", "multiple_choice", True),
        ("D", "C", "multiple_choice", False),
    ]
    for pred, gold, kind, expected in cases:
        got = grade(pred, gold, kind)  # type: ignore[arg-type]
        ok = got == expected
        if not ok:
            failures += 1
        print(f"  [{'ok ' if ok else 'FAIL'}] grade({pred!r}, {gold!r}) = {got} (want {expected})")
    return failures


def check_loading(n: int = 5) -> int:
    failures = 0
    print("== loading ==")
    try:
        gsm = load_gsm_hard(limit=n)
        print(f"  GSM-Hard: {len(gsm)} tasks")
        for t in gsm[:2]:
            print(f"    gold={t.gold:>20}  Q={t.question[:70]!r}")
        if not gsm:
            failures += 1
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] GSM-Hard load: {exc}")
        failures += 1

    try:
        mmlu = load_mmlu_pro(limit=n)
        print(f"  MMLU-Pro: {len(mmlu)} tasks")
        for t in mmlu[:1]:
            print(f"    gold={t.gold}  options={len(t.options or [])}")
            print(f"    {render_question(t)[:180]!r}")
        if not mmlu:
            failures += 1
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] MMLU-Pro load: {exc}")
        failures += 1
    return failures


if __name__ == "__main__":
    total = check_extraction() + check_grading() + check_loading()
    print()
    if total:
        print(f"{total} check(s) FAILED — fix before generating any corpus.")
        sys.exit(1)
    print("All grader checks passed.")
