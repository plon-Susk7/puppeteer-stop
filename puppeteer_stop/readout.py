"""`answer@t` — what the system would have returned had it halted at step t.

This is the single definition the whole diagnostic rests on, so it lives alone
and is tested directly.

Not every agent emits an answer: a `critique` activation may only criticize.
So `answer@t` is a *readout over accumulated outputs*, not the output of step t:

    answer@t = F_agg({o_0 ... o_t})

Two aggregators, because they disagree informatively:

  `last` — most recent parseable answer up to t.  Measures **drift**: an answer
           that was correct and got overwritten.
  `vote` — majority over all parseable answers up to t (the paper's aggregation).
           Measures **outvoting**: a correct answer that survived but was
           outnumbered.

Divergence between the two is itself a finding, which is why both are recorded
at generation time rather than reconstructed later.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Literal, Sequence

Readout = Literal["last", "vote"]


def readout_last(answers: Sequence[str | None]) -> str | None:
    """Most recent non-None answer."""
    for answer in reversed(answers):
        if answer is not None:
            return answer
    return None


def readout_vote(answers: Sequence[str | None]) -> str | None:
    """Majority vote over non-None answers; ties broken toward the most recent.

    Recency tie-breaking matters more than it looks: at t=1 there is exactly one
    answer, and every subsequent tie would otherwise resolve arbitrarily by dict
    ordering, injecting noise into the abandonment measurement.
    """
    present = [a for a in answers if a is not None]
    if not present:
        return None
    counts = Counter(present)
    top = max(counts.values())
    tied = {a for a, c in counts.items() if c == top}
    if len(tied) == 1:
        return next(iter(tied))
    for answer in reversed(present):
        if answer in tied:
            return answer
    return None


def apply(answers: Sequence[str | None], mode: Readout) -> str | None:
    return readout_last(answers) if mode == "last" else readout_vote(answers)


def prefix_readouts(
    answers: Sequence[str | None], mode: Readout
) -> list[str | None]:
    """`answer@t` for every t — the trajectory of what would have been returned.

    Computed incrementally so a T-step episode costs O(T) aggregations rather
    than O(T^2); at corpus scale this is the difference between seconds and
    minutes of replay.
    """
    out: list[str | None] = []
    if mode == "last":
        current: str | None = None
        for answer in answers:
            if answer is not None:
                current = answer
            out.append(current)
        return out

    counts: Counter[str] = Counter()
    present: list[str] = []
    for answer in answers:
        if answer is not None:
            counts[answer] += 1
            present.append(answer)
        if not present:
            out.append(None)
            continue
        top = max(counts.values())
        tied = {a for a, c in counts.items() if c == top}
        if len(tied) == 1:
            out.append(next(iter(tied)))
        else:
            out.append(next(a for a in reversed(present) if a in tied))
    return out


def correctness_curve(
    answers: Sequence[str | None],
    gold: str,
    kind: str,
    mode: Readout,
) -> list[bool]:
    """Whether halting at each step t would have been correct."""
    from .tasks import grade

    return [
        grade(a, gold, kind)  # type: ignore[arg-type]
        for a in prefix_readouts(answers, mode)
    ]
