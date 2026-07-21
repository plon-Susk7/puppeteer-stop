"""E0 — end-to-end pipeline smoke test with a scripted mock provider.

Verifies the whole chain offline and for free:

    tasks -> rollout -> trace JSONL -> read back -> E1 oracle gap

The mock is scripted to contain a *known* abandonment pattern, so the E1 report
it produces can be checked against a hand-computed answer. If this passes, the
only thing standing between here and a real corpus is an API key.

Run:  python experiments/e0_smoke_pipeline.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.oracle_gap import report  # noqa: E402
from puppeteer_stop.llm import Completion  # noqa: E402
from puppeteer_stop.rollout import generate_corpus  # noqa: E402
from puppeteer_stop.tasks import Task  # noqa: E402
from puppeteer_stop.trace import TraceWriter, load_traces  # noqa: E402


class MockClient:
    """Stands in for LLMClient. Emits a scripted answer per (task, step).

    Deliberately encodes the phenomenon under study: task A finds the correct
    answer at step 1 and is then talked out of it.
    """

    def __init__(self, script: dict[str, list[str]]) -> None:
        self.script = script
        self.calls = 0
        # Step is tracked per task rather than inferred from the prompt: the
        # agent action templates legitimately contain bracketed placeholders
        # like [YOUR REASONING RESULT], so counting markup is unreliable.
        self._step: dict[str, int] = {}

    def complete(self, messages, *, seed=None, **kw) -> Completion:
        # Recover which task this is from the rendered question in the payload.
        user = messages[-1]["content"]
        key = next(k for k in self.script if k in user)
        step = self._step.get(key, 0)
        self._step[key] = step + 1

        answers = self.script[key]
        text = answers[min(step, len(answers) - 1)]
        self.calls += 1
        return Completion(
            text=text,
            prompt_tokens=120,
            completion_tokens=60,
            model="mock",
            model_version="mock-1",
        )


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="pstop-smoke-"))
    try:
        tasks = [
            Task("A", "numeric", "TASKA what is it", "42.0", "synthetic"),
            Task("B", "numeric", "TASKB what is it", "7.0", "synthetic"),
            Task("C", "numeric", "TASKC what is it", "5.0", "synthetic"),
        ]
        script = {
            # correct at step 1, abandoned by the end
            "TASKA": ["FINAL ANSWER: 1", "FINAL ANSWER: 42", "FINAL ANSWER: 99"],
            # correct throughout -> tail steps are wasted
            "TASKB": ["FINAL ANSWER: 7", "FINAL ANSWER: 7", "FINAL ANSWER: 7"],
            # never correct
            "TASKC": ["FINAL ANSWER: 1", "FINAL ANSWER: 2", "FINAL ANSWER: 3"],
        }
        client = MockClient(script)
        path = tmp / "smoke.jsonl"
        writer = TraceWriter(path)

        n = generate_corpus(
            tasks, client, writer, policy="roundrobin", depth=3, progress_every=0
        )
        print(f"generated {n} episodes, {client.calls} model calls")

        # Resumability: a second pass must do no work.
        writer2 = TraceWriter(path)
        again = generate_corpus(
            tasks, client, writer2, policy="roundrobin", depth=3, progress_every=0
        )
        print(f"resume pass wrote {again} episodes (expected 0)")

        traces = load_traces(path)
        print(f"read back {len(traces)} traces")

        rep = report(traces, "last")
        print()
        print(rep.summary())
        print()
        print(rep.verdict())

        failures = 0
        checks = [
            ("episodes generated", n, 3),
            ("resume wrote nothing", again, 0),
            ("traces round-tripped", len(traces), 3),
            ("acc@T", round(rep.acc_at_T, 4), 0.3333),      # B only
            ("acc@oracle", round(rep.acc_at_oracle, 4), 0.6667),  # A and B
            ("gap", round(rep.gap, 4), 0.3333),
            ("abandonment", round(rep.abandonment_rate, 4), 0.3333),  # A
        ]
        print()
        for label, got, want in checks:
            ok = got == want
            failures += 0 if ok else 1
            print(f"  [{'ok ' if ok else 'FAIL'}] {label}: {got} (want {want})")

        # Concurrent generation must not interleave JSONL lines. This is the
        # failure mode the writer lock exists to prevent, and a corrupted
        # corpus would only surface much later, during analysis.
        print()
        print("== concurrent generation ==")
        many = [
            Task(f"T{i}", "numeric", f"TASK{i:03d} what is it", "42.0", "synthetic")
            for i in range(60)
        ]
        script_many = {f"TASK{i:03d}": ["FINAL ANSWER: 42"] * 3 for i in range(60)}
        cpath = tmp / "concurrent.jsonl"
        cwriter = TraceWriter(cpath)
        n_conc = generate_corpus(
            many,
            MockClient(script_many),
            cwriter,
            policy="roundrobin",
            depth=3,
            concurrency=8,
            progress_every=0,
        )
        raw_lines = [ln for ln in cpath.read_text(encoding="utf-8").splitlines() if ln.strip()]
        parsed = load_traces(cpath)
        ids = {t.episode_id for t in parsed}

        conc_checks = [
            ("episodes written", n_conc, 60),
            ("raw lines on disk", len(raw_lines), 60),
            ("lines parse cleanly", len(parsed), 60),
            ("episode ids unique", len(ids), 60),
        ]
        for label, got, want in conc_checks:
            ok = got == want
            failures += 0 if ok else 1
            print(f"  [{'ok ' if ok else 'FAIL'}] {label}: {got} (want {want})")

        print()
        if failures:
            print(f"{failures} check(s) FAILED.")
            return 1
        print("Pipeline smoke test passed: rollout -> trace -> replay -> E1 (serial + concurrent).")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
