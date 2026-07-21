"""E0 — readout and oracle-gap validation on synthetic trajectories.

The E1 number is only as trustworthy as `answer@t`. This exercises the readout
and the gap metrics against hand-constructed episodes where the right answer is
known by inspection, so the logic is verified before any API budget is spent.

Run:  python experiments/e0_validate_readout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.oracle_gap import episode_metrics, report  # noqa: E402
from puppeteer_stop.readout import prefix_readouts, readout_last, readout_vote  # noqa: E402
from puppeteer_stop.trace import EpisodeTrace, StepRecord  # noqa: E402


def make_trace(answers, gold="42.0", episode_id="ep") -> EpisodeTrace:
    """Synthetic episode: `answers` is the per-step parsed answer (None allowed)."""
    steps = [
        StepRecord(
            step=i,
            agent="reasoning",
            model="synthetic",
            model_version="synthetic",
            output="",
            parsed_answer=a,
            answer_last=None,
            answer_vote=None,
            correct_last=False,
            correct_vote=False,
            prompt_tokens=100,
            completion_tokens=100,
        )
        for i, a in enumerate(answers)
    ]
    return EpisodeTrace(
        episode_id=episode_id,
        task_id="t",
        dataset="synthetic",
        kind="numeric",
        question="q",
        gold=gold,
        pool="strong",
        rollout_policy="random",
        seed=0,
        steps=steps,
    )


def check(label, got, want) -> int:
    ok = got == want
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}: {got!r} (want {want!r})")
    return 0 if ok else 1


def main() -> int:
    failures = 0

    print("== readout: last ==")
    failures += check("no answers", readout_last([None, None]), None)
    failures += check("skips None", readout_last(["1.0", None]), "1.0")
    failures += check("takes latest", readout_last(["1.0", "2.0"]), "2.0")

    print("== readout: vote ==")
    failures += check("majority wins", readout_vote(["1.0", "2.0", "1.0"]), "1.0")
    failures += check("tie -> most recent", readout_vote(["1.0", "2.0"]), "2.0")
    failures += check("all None", readout_vote([None]), None)

    print("== prefix readouts ==")
    failures += check(
        "last, with gap",
        prefix_readouts(["1.0", None, "2.0"], "last"),
        ["1.0", "1.0", "2.0"],
    )
    failures += check(
        "vote, minority then majority",
        prefix_readouts(["1.0", "2.0", "2.0"], "vote"),
        ["1.0", "2.0", "2.0"],
    )
    failures += check(
        "leading None",
        prefix_readouts([None, "3.0"], "vote"),
        [None, "3.0"],
    )

    print("== episode metrics ==")
    # Correct at t=1, then talked out of it: the abandonment case.
    m = episode_metrics(make_trace(["7.0", "42.0", "9.0"]), "last")
    failures += check("abandoned", m.abandoned, True)
    failures += check("correct_at_T", m.correct_at_T, False)
    failures += check("correct_at_any", m.correct_at_any, True)
    failures += check("first_correct_step", m.first_correct_step, 1)

    # Correct and stays correct: no abandonment, and the tail is wasted.
    m = episode_metrics(make_trace(["42.0", "42.0", "42.0"]), "last")
    failures += check("stable: abandoned", m.abandoned, False)
    failures += check("stable: correct_at_T", m.correct_at_T, True)
    failures += check("stable: wasted_steps", m.wasted_steps, 2)

    # Never correct.
    m = episode_metrics(make_trace(["1.0", "2.0"]), "last")
    failures += check("never: correct_at_any", m.correct_at_any, False)
    failures += check("never: first_correct_step", m.first_correct_step, None)

    print("== aggregate report ==")
    traces = [
        make_trace(["7.0", "42.0", "9.0"], episode_id="a"),   # abandoned
        make_trace(["42.0", "42.0", "42.0"], episode_id="b"), # correct throughout
        make_trace(["1.0", "2.0", "3.0"], episode_id="c"),    # never correct
        make_trace(["1.0", "2.0", "42.0"], episode_id="d"),   # arrives late
    ]
    rep = report(traces, "last")
    failures += check("acc@T", round(rep.acc_at_T, 4), 0.5)       # b, d
    failures += check("acc@oracle", round(rep.acc_at_oracle, 4), 0.75)  # a, b, d
    failures += check("gap", round(rep.gap, 4), 0.25)
    failures += check("abandonment", round(rep.abandonment_rate, 4), 0.25)
    print()
    print(rep.summary())

    print()
    if failures:
        print(f"{failures} check(s) FAILED — E1 logic is not trustworthy yet.")
        return 1
    print("All readout / oracle-gap checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
