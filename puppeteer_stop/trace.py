"""Append-only orchestration traces.

One JSON object per episode, one line each. JSONL rather than a database
because the corpus is the project's primary artifact: it ships as a versioned
Kaggle Dataset, it is inspectable with `head`, and a truncated write costs one
episode rather than the file.

Field names follow the orchestration-trace vocabulary from arXiv:2605.02801
(the survey that named the stopping gap) so the corpus is legible to the
community this work is aimed at.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator


@dataclass
class StepRecord:
    """One agent activation."""

    step: int
    agent: str
    model: str
    model_version: str
    output: str
    parsed_answer: str | None       # answer this step emitted, if any
    answer_last: str | None         # answer@t under the `last` readout
    answer_vote: str | None         # answer@t under the `vote` readout
    correct_last: bool
    correct_vote: bool
    prompt_tokens: int
    completion_tokens: int
    cached: bool = False
    latency_s: float = 0.0
    # Populated only when a policy drove selection; None for fixed rollouts.
    select_probs: list[float] | None = None
    stop_prob: float | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class EpisodeTrace:
    """One task, run to fixed depth T."""

    episode_id: str
    task_id: str
    dataset: str
    kind: str
    question: str
    gold: str
    pool: str                       # "strong" | "cheap"
    rollout_policy: str             # "random" | "roundrobin" | "heuristic" | "learned"
    seed: int
    steps: list[StepRecord] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self.steps)

    def to_json(self) -> str:
        payload = asdict(self)
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def from_dict(d: dict) -> "EpisodeTrace":
        steps = [StepRecord(**s) for s in d.pop("steps", [])]
        return EpisodeTrace(**d, steps=steps)


class TraceWriter:
    """Append-only writer that knows what it has already done.

    Resumability is not optional here: Kaggle kills sessions at 12 hours, and
    regenerating a corpus costs real money and GPU quota. `completed()` lets a
    generation loop skip work it already paid for.
    """

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = str(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._seen = self._scan()
        # Corpus generation runs episodes concurrently against a batching
        # server, so appends must be serialized or lines interleave and the
        # JSONL is silently corrupted.
        self._lock = threading.Lock()

    def _scan(self) -> set[str]:
        if not os.path.exists(self.path):
            return set()
        seen: set[str] = set()
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    seen.add(json.loads(line)["episode_id"])
                except (json.JSONDecodeError, KeyError):
                    # A partial final line from a killed session. Skip it; the
                    # episode simply gets regenerated.
                    continue
        return seen

    def completed(self) -> set[str]:
        return set(self._seen)

    def has(self, episode_id: str) -> bool:
        return episode_id in self._seen

    def write(self, trace: EpisodeTrace) -> None:
        line = trace.to_json()
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
            self._seen.add(trace.episode_id)

    def __len__(self) -> int:
        return len(self._seen)


def read_traces(path: str | os.PathLike) -> Iterator[EpisodeTrace]:
    """Stream episodes back for offline analysis."""
    with open(str(path), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield EpisodeTrace.from_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue


def load_traces(path: str | os.PathLike) -> list[EpisodeTrace]:
    return list(read_traces(path))
