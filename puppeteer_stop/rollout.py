"""Trajectory generation.

Runs each task to a *fixed* depth T regardless of what any policy would have
chosen, recording `answer@t` at every step. Fixed depth is the point: E1 asks
where halting *would* have been best, which is unanswerable if the rollout
already halted somewhere.

The corpus produced here is generated once and replayed by every downstream
experiment, so this is the only expensive module in the project.
"""

from __future__ import annotations

import random
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Sequence

from . import agents
from .agents import Agent, build_messages
from .llm import LLMClient
from .readout import prefix_readouts
from .tasks import Task, extract_answer, grade, render_question
from .trace import EpisodeTrace, StepRecord, TraceWriter

SelectionPolicy = Callable[[int, Sequence[str], random.Random], Agent]


# --------------------------------------------------------------------------
# Selection policies (fixed — no learning here)
# --------------------------------------------------------------------------

def policy_random(step: int, history: Sequence[str], rng: random.Random) -> Agent:
    """Uniform over the pool. The cleanest control: it decouples the stopping
    question from any selection quality."""
    return rng.choice(agents.POOL)


def policy_roundrobin(step: int, history: Sequence[str], rng: random.Random) -> Agent:
    """Deterministic cycle through the pool."""
    return agents.POOL[step % len(agents.POOL)]


# A plausible collaboration prior: plan, reason, then alternate between
# critique/modify and consolidation. This stands in for the paper's untrained
# "Initialized" policy, which is the zero-shot preference of a 70B reward model
# and cannot be reproduced without it. It is a *proxy*, and is labelled as such
# in every result.
_HEURISTIC_STAGES = [
    ("planning", "reasoning"),
    ("reasoning",),
    ("critique", "question"),
    ("modify", "reasoning"),
    ("summarize", "critique"),
    ("conclude", "summarize"),
]


def policy_heuristic(step: int, history: Sequence[str], rng: random.Random) -> Agent:
    stage = _HEURISTIC_STAGES[min(step, len(_HEURISTIC_STAGES) - 1)]
    return agents.get(rng.choice(stage))


POLICIES: dict[str, SelectionPolicy] = {
    "random": policy_random,
    "roundrobin": policy_roundrobin,
    "heuristic": policy_heuristic,
}


# --------------------------------------------------------------------------
# Episode generation
# --------------------------------------------------------------------------

def episode_id(task: Task, policy: str, pool: str, seed: int) -> str:
    return f"{task.task_id}|{policy}|{pool}|s{seed}"


def run_episode(
    task: Task,
    client: LLMClient,
    *,
    policy: str = "heuristic",
    pool: str = "strong",
    depth: int = 6,
    seed: int = 0,
) -> EpisodeTrace:
    """Run one task to fixed depth, recording answer@t at every step."""
    rng = random.Random(f"{task.task_id}:{policy}:{seed}")
    select = POLICIES[policy]
    question = render_question(task)

    history: list[str] = []
    parsed: list[str | None] = []
    steps: list[StepRecord] = []

    for step in range(depth):
        agent = select(step, history, rng)
        messages = build_messages(agent, question, history)
        completion = client.complete(messages, seed=seed)

        answer = extract_answer(completion.text, task.kind)
        parsed.append(answer)

        # Recomputed each step rather than cached, so the record is
        # self-consistent even if a run is resumed mid-episode.
        last_curve = prefix_readouts(parsed, "last")
        vote_curve = prefix_readouts(parsed, "vote")

        steps.append(
            StepRecord(
                step=step,
                agent=agent.name,
                model=completion.model,
                model_version=completion.model_version,
                output=completion.text,
                parsed_answer=answer,
                answer_last=last_curve[-1],
                answer_vote=vote_curve[-1],
                correct_last=grade(last_curve[-1], task.gold, task.kind),
                correct_vote=grade(vote_curve[-1], task.gold, task.kind),
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                cached=completion.cached,
                latency_s=completion.latency_s,
            )
        )
        history.append(f"[{agent.name}] {completion.text}")

    return EpisodeTrace(
        episode_id=episode_id(task, policy, pool, seed),
        task_id=task.task_id,
        dataset=task.dataset,
        kind=task.kind,
        question=question,
        gold=task.gold,
        pool=pool,
        rollout_policy=policy,
        seed=seed,
        steps=steps,
        meta={"depth": depth, "agent_pool": list(agents.AGENT_NAMES)},
    )


def generate_corpus(
    tasks: Iterable[Task],
    client: LLMClient,
    writer: TraceWriter,
    *,
    policy: str = "heuristic",
    pool: str = "strong",
    depth: int = 6,
    seed: int = 0,
    concurrency: int = 1,
    progress_every: int = 10,
) -> int:
    """Generate trajectories, skipping episodes already in the trace file.

    Resumability is the whole point: a killed Kaggle session must cost one
    episode, not the corpus.

    `concurrency` runs several *episodes* at once. Steps within an episode stay
    strictly sequential — step t+1 conditions on step t — but tasks are
    independent, so that is the axis to parallelize. This matters enormously
    against a self-hosted vLLM server: one request at a time leaves the GPU
    almost entirely idle, since throughput there comes from continuous batching.
    Against a rate-limited hosted API, keep it low.
    """
    tasks = [t for t in tasks if not writer.has(episode_id(t, policy, pool, seed))]
    if not tasks:
        return 0

    written = 0
    failures = 0
    counter_lock = threading.Lock()

    def one(task: Task) -> None:
        nonlocal written, failures
        try:
            trace = run_episode(
                task, client, policy=policy, pool=pool, depth=depth, seed=seed
            )
        except Exception as exc:  # noqa: BLE001
            # One bad task must not end the run. Record nothing and move on;
            # the episode is simply regenerated on the next pass.
            with counter_lock:
                failures += 1
            print(f"  !! {task.task_id}: {type(exc).__name__}: {exc}")
            return

        writer.write(trace)
        with counter_lock:
            written += 1
            n = written
        if progress_every and n % progress_every == 0:
            correct = sum(s.correct_last for s in trace.steps)
            print(
                f"  [{n}/{len(tasks)}] last={trace.steps[-1].answer_last} "
                f"gold={trace.gold} correct_steps={correct}/{len(trace.steps)} "
                f"tokens={trace.total_tokens:,}"
            )

    if concurrency <= 1:
        for task in tasks:
            one(task)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool_exec:
            list(pool_exec.map(one, tasks))

    if failures:
        print(f"  ({failures} episode(s) failed and will be retried on the next run)")
    return written
