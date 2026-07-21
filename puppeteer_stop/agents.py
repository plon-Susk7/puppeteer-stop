"""The agent pool.

Role prompts and reasoning-pattern prompts are ported from Appendix B of the
paper (Figures 13, 15, 16) rather than rewritten, so that any difference in
results is attributable to orchestration rather than to prompt engineering.
Tool-use agents (Figure 14) are deliberately omitted from the first phase: they
add failure modes (network, sandboxing) that are orthogonal to the stopping
question.

Every reasoning pattern in the paper concludes with `FINAL ANSWER:`, which is
what makes `answer@t` cheap to read out at every step — the whole diagnostic
depends on this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

TERMINATOR = "Terminator"


@dataclass(frozen=True)
class Agent:
    """An atomic reasoning behavior. `a = (m, r, t)` in the paper's notation.

    `name` is the reasoning pattern r; the model m is bound at rollout time by
    the pool configuration, so the same agent can be instantiated against a
    strong or cheap backend without changing its definition.
    """

    name: str
    role: str
    action: str
    emits_answer: bool = True


# Figure 13 — Reasoning Agent Role Prompts
_ROLES = {
    "planning": (
        "You are an expert in task decomposition and planning. Responsible for "
        "generating structured plans to solve complex tasks (planning)."
    ),
    "reasoning": (
        "You are an expert in logical reasoning. Responsible for synthesizing "
        "solutions to sub-problems (reasoning)."
    ),
    "critique": (
        "You are an expert in critique and verification. Responsible for identifying "
        "flaws and providing feedback on prior reasoning (critique)."
    ),
    "reflect": (
        "You are an expert in metacognitive reflection. Responsible for analyzing the "
        "overall reasoning trajectory and proposing improvements (reflect)."
    ),
    "question": (
        "You are an expert in problem decomposition. Responsible for generating "
        "clarifying or follow-up sub-questions (question)."
    ),
    "summarize": (
        "You are an expert in summarization. Responsible for generating concise "
        "summaries of intermediate results (summarize)."
    ),
    "conclude": (
        "You are an expert in synthesis. Responsible for producing the final "
        "conclusions based on collective reasoning outcomes (conclude)."
    ),
    "modify": (
        "You are an expert in error analysis and correction. Responsible for "
        "identifying errors and revising prior outputs accordingly (modify)."
    ),
}

_ANSWER_TEMPLATE = (
    "Conclude with:\n"
    "REASONING RESULT: [YOUR REASONING RESULT].\n"
    "FINAL ANSWER: [YOUR FINAL ANSWER]."
)

# Figures 15 and 16 — Reasoning-pattern Prompts
_ACTIONS = {
    "planning": (
        "Decompose the question and plan the next steps to address the question. "
        "You should complete your planning using the following template:\n"
        "REASONING RESULT: [YOUR REASONING RESULT].\n" + _ANSWER_TEMPLATE
    ),
    "reasoning": (
        "Now, you need to continue the reasoning to get closer to the correct answer. "
        "You need to follow the direction of the reasoning path and go forward.\n"
        + _ANSWER_TEMPLATE
    ),
    "critique": (
        "You need to critique the previous reasoning. Consider plausibility, "
        "correctness of each step, and whether the conclusion follows.\n"
        + _ANSWER_TEMPLATE
    ),
    "reflect": (
        "You will be provided with a previous reasoning attempt. In a few sentences, "
        "diagnose the potential cause of failure or discrepancy, and outline a new, "
        "concise, high-level plan to prevent the same issue. Use complete sentences. "
        "Reflect on the current state of the task and propose the next steps.\n"
        + _ANSWER_TEMPLATE
    ),
    "question": (
        "Your task is to propose the next sub-question along with its answer. Ensure "
        "it logically follows from the previous reasoning and addresses any gaps. "
        "Provide a well-reasoned answer, supported by evidence or logical arguments.\n"
        + _ANSWER_TEMPLATE
    ),
    "summarize": (
        "You need to summarize previous results and provide some intermediate "
        "conclusions. Summarize the reasoning paths and provide a final conclusion.\n"
        + _ANSWER_TEMPLATE
    ),
    "conclude": (
        "You need to conclude the task and provide a final answer.\n" + _ANSWER_TEMPLATE
    ),
    "modify": (
        "You need to identify and correct errors in the previous reasoning. "
        "Explicitly point out and correct any errors, misconceptions, or inaccuracies. "
        "State which part of the previous reasoning was incorrect, why it was "
        "incorrect, and what the correct understanding is.\n" + _ANSWER_TEMPLATE
    ),
}

POOL: tuple[Agent, ...] = tuple(
    Agent(name=name, role=_ROLES[name], action=_ACTIONS[name]) for name in _ACTIONS
)

AGENT_NAMES: tuple[str, ...] = tuple(a.name for a in POOL)


def get(name: str) -> Agent:
    for agent in POOL:
        if agent.name == name:
            return agent
    raise KeyError(f"unknown agent {name!r}; pool is {AGENT_NAMES}")


def build_messages(
    agent: Agent,
    question: str,
    history: Sequence[str],
    *,
    max_history_chars: int = 6000,
) -> list[dict]:
    """Assemble the chat payload for one activation.

    `history` is the accumulated context (the paper's `S_t`). It is truncated
    from the *front* so the most recent reasoning always survives — dropping the
    latest step would break the sequential dependency the whole method assumes.
    """
    prior = "\n\n".join(history)
    if len(prior) > max_history_chars:
        prior = "...[earlier reasoning truncated]...\n" + prior[-max_history_chars:]

    previous = prior if prior else "(no previous reasoning yet)"
    user = (
        f"QUESTION:\n{question}\n\n"
        f"{agent.action}\n\n"
        f"*Your previous reasoning was: {previous}*"
    )
    return [
        {"role": "system", "content": agent.role},
        {"role": "user", "content": user},
    ]
