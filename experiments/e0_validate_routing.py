"""E0 — validate multi-backend routing (2+ GPUs).

Two things must hold, and they fail in different ways:

  round_robin — requests spread evenly, so N GPUs give ~N x throughput. A router
                that silently favours one backend halves the speedup with no
                error anywhere.
  per_agent   — each agent binds to ONE model, stably across processes. If the
                binding drifted between runs, agent identity and model identity
                would be confounded inside a single episode and the traces would
                be uninterpretable.

Run:  python experiments/e0_validate_routing.py
"""

from __future__ import annotations

import sys
import threading
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from puppeteer_stop.agents import AGENT_NAMES  # noqa: E402
from puppeteer_stop.config import PoolSpec  # noqa: E402
from puppeteer_stop.llm import Completion, LLMRouter  # noqa: E402


class FakeClient:
    """Stands in for LLMClient; records how many calls it received."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.calls = 0
        self._lock = threading.Lock()

    def complete(self, messages, **kw) -> Completion:
        with self._lock:
            self.calls += 1
        return Completion(
            text="FINAL ANSWER: 1",
            prompt_tokens=10,
            completion_tokens=5,
            model=self.model,
            model_version=self.model,
        )


def main() -> int:
    failures = 0

    def check(label, got, want) -> None:
        nonlocal failures
        ok = got == want
        failures += 0 if ok else 1
        print(f"  [{'ok ' if ok else 'FAIL'}] {label}: {got!r} (want {want!r})")

    print("== backend expansion ==")
    # One model, two URLs -> replication across two GPUs.
    spec = PoolSpec(
        name="local",
        model="Qwen/Qwen2.5-3B-Instruct",
        api="openai",
        base_url="http://localhost:8000/v1,http://localhost:8001/v1",
    )
    check("replication -> 2 backends", len(spec.backends), 2)
    check("same model on both", len({m for m, _ in spec.backends}), 1)
    check("distinct urls", len({u for _, u in spec.backends}), 2)

    # Two models, two URLs -> heterogeneous pool.
    het = PoolSpec(
        name="local",
        model="Qwen/Qwen2.5-3B-Instruct,mistralai/Mistral-7B-Instruct-v0.3",
        api="openai",
        base_url="http://localhost:8000/v1,http://localhost:8001/v1",
        routing="per_agent",
    )
    check("heterogeneous -> 2 backends", len(het.backends), 2)
    check("two distinct models", len({m for m, _ in het.backends}), 2)

    # Single backend must stay a plain client, not a router.
    solo = PoolSpec(name="local", model="a", api="openai", base_url="http://x/v1")
    check("single backend stays single", len(solo.backends), 1)

    print("== round_robin spreads load ==")
    clients = [FakeClient("m0"), FakeClient("m1")]
    router = LLMRouter(clients, strategy="round_robin")
    for i in range(200):
        router.complete([{"role": "user", "content": "x"}], route_key=f"agent{i % 8}")
    counts = [c.calls for c in clients]
    check("all requests served", sum(counts), 200)
    balanced = max(counts) - min(counts) <= 1
    print(f"  [{'ok ' if balanced else 'FAIL'}] balanced across backends: {counts}")
    failures += 0 if balanced else 1

    print("== round_robin is thread-safe ==")
    clients = [FakeClient("m0"), FakeClient("m1")]
    router = LLMRouter(clients, strategy="round_robin")

    def hammer() -> None:
        for _ in range(100):
            router.complete([{"role": "user", "content": "x"}])

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    counts = [c.calls for c in clients]
    check("no dropped/duplicated dispatches", sum(counts), 800)
    skew = abs(counts[0] - counts[1])
    print(f"  [{'ok ' if skew <= 8 else 'FAIL'}] balanced under contention: {counts} (skew {skew})")
    failures += 0 if skew <= 8 else 1

    print("== per_agent binding is stable ==")
    def build() -> LLMRouter:
        return LLMRouter([FakeClient("qwen"), FakeClient("mistral")], strategy="per_agent")

    a1 = build().assignments(AGENT_NAMES)
    a2 = build().assignments(AGENT_NAMES)
    check("identical across router instances", a1, a2)
    check("every agent bound", len(a1), len(AGENT_NAMES))
    # sha256-based, so it must also be stable across processes — a plain hash()
    # would be salted per process and silently reshuffle every run.
    expected_reasoning = a1["reasoning"]
    for _ in range(5):
        check(f"'reasoning' stable on rebuild", build().assignments(["reasoning"])["reasoning"],
              expected_reasoning)

    spread = Counter(a1.values())
    used_both = len(spread) == 2
    print(f"  [{'ok ' if used_both else 'FAIL'}] both models used: {dict(spread)}")
    failures += 0 if used_both else 1

    print("== per_agent: same agent always same model ==")
    router = build()
    seen = {
        router.complete([{"role": "user", "content": "x"}], route_key="critique").model
        for _ in range(50)
    }
    check("one model for 'critique' over 50 calls", len(seen), 1)

    print("== bad config is rejected ==")
    try:
        LLMRouter([], strategy="round_robin")
        check("empty client list rejected", False, True)
    except ValueError:
        check("empty client list rejected", True, True)
    try:
        LLMRouter([FakeClient("a")], strategy="nonsense")
        check("unknown strategy rejected", False, True)
    except ValueError:
        check("unknown strategy rejected", True, True)

    print()
    if failures:
        print(f"{failures} check(s) FAILED — multi-GPU routing is not trustworthy.")
        return 1
    print("Routing validated (replication, per-agent binding, thread safety).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
