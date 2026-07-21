"""E1 — the oracle stopping gap.

The headline diagnostic. Given trajectories run to fixed depth T, ask how much
accuracy a perfect halting decision would have bought, holding the agent pool,
the prompts and the selection policy fixed.

    acc@T       accuracy of the answer at the final step
    acc@oracle  fraction of episodes correct at *any* step
    gap         acc@oracle - acc@T          <- the number the proposal is built on

Supporting counts:
    abandonment    correct at some t < T, wrong at T   (the vivid one)
    wasted suffix  steps executed after answer@t stopped changing
    first-correct  where correctness first appears

Everything here is pure replay over cached traces: no API calls, no GPU.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from puppeteer_stop.readout import Readout, prefix_readouts
from puppeteer_stop.tasks import grade
from puppeteer_stop.trace import EpisodeTrace


@dataclass
class EpisodeMetrics:
    episode_id: str
    dataset: str
    correct_at_T: bool
    correct_at_any: bool
    abandoned: bool            # was correct earlier, wrong at the end
    first_correct_step: int | None
    last_change_step: int      # last step at which answer@t changed
    n_steps: int
    total_tokens: int

    @property
    def wasted_steps(self) -> int:
        """Steps executed after the returned answer stopped changing."""
        return max(0, self.n_steps - 1 - self.last_change_step)


def episode_metrics(trace: EpisodeTrace, mode: Readout = "last") -> EpisodeMetrics:
    answers = [s.parsed_answer for s in trace.steps]
    curve = prefix_readouts(answers, mode)
    correct = [grade(a, trace.gold, trace.kind) for a in curve]  # type: ignore[arg-type]

    first_correct = next((i for i, c in enumerate(correct) if c), None)
    last_change = 0
    for i in range(1, len(curve)):
        if curve[i] != curve[i - 1]:
            last_change = i

    correct_at_T = bool(correct and correct[-1])
    return EpisodeMetrics(
        episode_id=trace.episode_id,
        dataset=trace.dataset,
        correct_at_T=correct_at_T,
        correct_at_any=any(correct),
        abandoned=any(correct[:-1]) and not correct_at_T if len(correct) > 1 else False,
        first_correct_step=first_correct,
        last_change_step=last_change,
        n_steps=len(trace.steps),
        total_tokens=trace.total_tokens,
    )


@dataclass
class GapReport:
    n: int
    readout: str
    acc_at_T: float
    acc_at_oracle: float
    gap: float
    gap_ci: tuple[float, float]
    abandonment_rate: float
    abandonment_ci: tuple[float, float]
    mean_wasted_steps: float
    wasted_step_fraction: float
    first_correct_hist: dict[int | None, int]
    mean_tokens: float

    def summary(self) -> str:
        lo, hi = self.gap_ci
        alo, ahi = self.abandonment_ci
        lines = [
            f"E1 oracle stopping gap  (readout={self.readout}, n={self.n})",
            f"  acc@T            {self.acc_at_T:.4f}",
            f"  acc@oracle       {self.acc_at_oracle:.4f}",
            f"  GAP              {self.gap:.4f}   95% CI [{lo:.4f}, {hi:.4f}]",
            f"  abandonment      {self.abandonment_rate:.4f}   95% CI [{alo:.4f}, {ahi:.4f}]",
            f"  wasted steps     {self.mean_wasted_steps:.2f} per episode "
            f"({self.wasted_step_fraction:.1%} of all steps)",
            f"  mean tokens      {self.mean_tokens:,.0f}",
            "  first correct at step:",
        ]
        for step in sorted(self.first_correct_hist, key=lambda k: (k is None, k)):
            label = "never" if step is None else f"t={step}"
            count = self.first_correct_hist[step]
            bar = "#" * max(1, round(40 * count / max(1, self.n)))
            lines.append(f"    {label:>6}  {count:4d}  {bar}")
        return "\n".join(lines)

    def verdict(self) -> str:
        """The plan's decision gate: gap >= 3pts OR abandonment >= 5%."""
        if self.gap >= 0.03 or self.abandonment_rate >= 0.05:
            return (
                "PROCEED — headroom is real; the stopping framing is supported."
            )
        return (
            "PIVOT — gap and abandonment are both small. The accuracy story is not "
            "there; fall back to the efficiency framing using wasted steps "
            f"({self.wasted_step_fraction:.1%} of steps were spent after the answer "
            "stopped changing), and reconsider before building the method."
        )


def _bootstrap_ci(
    values: Sequence[float], n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1])


def report(traces: Sequence[EpisodeTrace], mode: Readout = "last") -> GapReport:
    metrics = [episode_metrics(t, mode) for t in traces]
    if not metrics:
        raise ValueError("no traces to analyze")

    n = len(metrics)
    at_T = [float(m.correct_at_T) for m in metrics]
    at_oracle = [float(m.correct_at_any) for m in metrics]
    # Paired per-episode differences — the gap is a paired quantity, so its CI
    # must be built from paired deltas, not from two independent accuracies.
    deltas = [o - t for o, t in zip(at_oracle, at_T)]
    abandoned = [float(m.abandoned) for m in metrics]

    hist: dict[int | None, int] = {}
    for m in metrics:
        hist[m.first_correct_step] = hist.get(m.first_correct_step, 0) + 1

    total_steps = sum(m.n_steps for m in metrics)
    total_wasted = sum(m.wasted_steps for m in metrics)

    return GapReport(
        n=n,
        readout=mode,
        acc_at_T=sum(at_T) / n,
        acc_at_oracle=sum(at_oracle) / n,
        gap=sum(deltas) / n,
        gap_ci=_bootstrap_ci(deltas),
        abandonment_rate=sum(abandoned) / n,
        abandonment_ci=_bootstrap_ci(abandoned),
        mean_wasted_steps=total_wasted / n,
        wasted_step_fraction=total_wasted / max(1, total_steps),
        first_correct_hist=hist,
        mean_tokens=sum(m.total_tokens for m in metrics) / n,
    )
