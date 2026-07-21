"""Benchmark loaders and graders for GSM-Hard and MMLU-Pro.

Graders are where silent failures live. GSM-Hard answers are deliberately huge
and often floating point, so naive string equality scores a correct answer wrong
and quietly deflates every number downstream. Everything here is built to be
hand-audited: `answer_spans` returns what the grader saw and why, and
`experiments/e0_validate_graders.py` samples items for manual review.

Loading is layered so the same code works locally and on Kaggle:
  1. local JSONL cache (fast, offline, reproducible)
  2. `datasets` library if installed
  3. HuggingFace datasets-server REST API via requests (no extra deps)
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Iterator, Literal

import requests

DATASETS_SERVER = "https://datasets-server.huggingface.co/rows"

GSM_HARD_REPO = "reasoning-machines/gsm-hard"
MMLU_PRO_REPO = "TIGER-Lab/MMLU-Pro"

TaskKind = Literal["numeric", "multiple_choice"]


@dataclass(frozen=True)
class Task:
    task_id: str
    kind: TaskKind
    question: str
    gold: str
    dataset: str
    options: list[str] | None = None  # multiple_choice only

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Answer extraction
# --------------------------------------------------------------------------

# Ordered by trust: an explicit marker beats a bare trailing number, because
# agents that reason out loud emit many incidental numbers.
_NUMERIC_MARKERS = [
    re.compile(r"####\s*([-+]?[\d,]*\.?\d+(?:[eE][-+]?\d+)?)"),
    re.compile(r"\\boxed\{\s*([-+]?[\d,]*\.?\d+(?:[eE][-+]?\d+)?)\s*\}"),
    re.compile(
        r"(?:final\s+answer|answer)\s*(?:is)?\s*[:=]?\s*\$?\s*"
        r"([-+]?[\d,]*\.?\d+(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    ),
]
_ANY_NUMBER = re.compile(r"[-+]?[\d,]*\.?\d+(?:[eE][-+]?\d+)?")

_CHOICE_MARKERS = [
    re.compile(r"\\boxed\{\s*([A-J])\s*\}"),
    re.compile(r"(?:final\s+answer|answer)\s*(?:is)?\s*[:=]?\s*\(?\s*([A-J])\b", re.IGNORECASE),
    re.compile(r"^\s*\(?([A-J])[\).]\s", re.MULTILINE),
]
_BARE_CHOICE = re.compile(r"\b([A-J])\b")


def _to_float(raw: str) -> float | None:
    try:
        return float(raw.replace(",", "").strip())
    except ValueError:
        return None


def extract_numeric(text: str) -> float | None:
    """Pull a numeric answer out of free-form model text.

    Markers first; otherwise fall back to the *last* number mentioned, which is
    the usual convention for chain-of-thought output.
    """
    if not text:
        return None
    for pattern in _NUMERIC_MARKERS:
        matches = pattern.findall(text)
        if matches:
            value = _to_float(matches[-1])
            if value is not None:
                return value
    matches = _ANY_NUMBER.findall(text)
    for raw in reversed(matches):
        value = _to_float(raw)
        if value is not None:
            return value
    return None


def extract_choice(text: str) -> str | None:
    """Pull a multiple-choice letter (A-J) out of free-form model text."""
    if not text:
        return None
    for pattern in _CHOICE_MARKERS:
        matches = pattern.findall(text)
        if matches:
            return matches[-1].upper()
    # Last resort: a standalone capital letter near the end of the response.
    tail = text[-200:]
    matches = _BARE_CHOICE.findall(tail)
    if matches:
        return matches[-1].upper()
    return None


def extract_answer(text: str, kind: TaskKind) -> str | None:
    """Normalized answer string, or None when nothing parseable is present.

    None is meaningful and must not be collapsed into "wrong": a critique agent
    legitimately emits no answer, and `answer@t` readout depends on telling those
    two cases apart.
    """
    if kind == "numeric":
        value = extract_numeric(text)
        return None if value is None else repr(value)
    choice = extract_choice(text)
    return choice


# --------------------------------------------------------------------------
# Grading
# --------------------------------------------------------------------------

def grade(prediction: str | None, gold: str, kind: TaskKind) -> bool:
    """True when `prediction` matches `gold`.

    Numeric comparison is relative-tolerance because GSM-Hard gold answers are
    frequently floats produced by a Python interpreter; requiring exact equality
    against a model's rounded restatement understates accuracy.
    """
    if prediction is None:
        return False
    if kind == "multiple_choice":
        return prediction.strip().upper() == gold.strip().upper()

    predicted = _to_float(prediction)
    expected = _to_float(gold)
    if predicted is None or expected is None:
        return False
    if math.isnan(predicted) or math.isnan(expected):
        return False
    if predicted == expected:
        return True
    scale = max(abs(expected), 1.0)
    return abs(predicted - expected) <= 1e-4 * scale


def grade_text(text: str, task: Task) -> bool:
    """Convenience: extract then grade, for raw model output."""
    return grade(extract_answer(text, task.kind), task.gold, task.kind)


def answer_spans(text: str, kind: TaskKind) -> dict[str, Any]:
    """What the grader saw — for the manual audit the plan requires."""
    return {
        "extracted": extract_answer(text, kind),
        "marker_hits": [p.findall(text) for p in
                        (_NUMERIC_MARKERS if kind == "numeric" else _CHOICE_MARKERS)],
        "tail": text[-200:] if text else "",
    }


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _fetch_rows(repo: str, config: str, split: str, limit: int) -> list[dict]:
    """Page through the HF datasets-server REST API (100 rows per request)."""
    rows: list[dict] = []
    offset = 0
    while len(rows) < limit:
        page = min(100, limit - len(rows))
        resp = requests.get(
            DATASETS_SERVER,
            params={
                "dataset": repo,
                "config": config,
                "split": split,
                "offset": offset,
                "length": page,
            },
            timeout=60,
        )
        resp.raise_for_status()
        batch = resp.json().get("rows", [])
        if not batch:
            break
        rows.extend(item["row"] for item in batch)
        offset += len(batch)
    return rows[:limit]


def _raw_rows(repo: str, config: str, split: str, limit: int, cache_path: str) -> list[dict]:
    if os.path.exists(cache_path):
        cached = _load_jsonl(cache_path)
        if len(cached) >= limit:
            return cached[:limit]

    try:  # available on Kaggle, usually not locally
        from datasets import load_dataset  # type: ignore

        ds = load_dataset(repo, config, split=split)
        rows = [dict(ds[i]) for i in range(min(limit, len(ds)))]
    except Exception:
        rows = _fetch_rows(repo, config, split, limit)

    _write_jsonl(cache_path, rows)
    return rows


def load_gsm_hard(limit: int = 500, cache_dir: str = "data") -> list[Task]:
    rows = _raw_rows(
        GSM_HARD_REPO, "default", "train", limit, os.path.join(cache_dir, "gsm_hard.jsonl")
    )
    tasks = []
    for i, row in enumerate(rows):
        target = row.get("target")
        if target is None:
            continue
        tasks.append(
            Task(
                task_id=f"gsmhard-{i:05d}",
                kind="numeric",
                question=str(row["input"]).strip(),
                gold=repr(float(target)),
                dataset="gsm-hard",
            )
        )
    return tasks


def load_mmlu_pro(limit: int = 300, cache_dir: str = "data") -> list[Task]:
    rows = _raw_rows(
        MMLU_PRO_REPO, "default", "test", limit, os.path.join(cache_dir, "mmlu_pro.jsonl")
    )
    letters = "ABCDEFGHIJ"
    tasks = []
    for i, row in enumerate(rows):
        options = list(row.get("options") or [])
        answer = row.get("answer")
        if not options or not answer:
            continue
        tasks.append(
            Task(
                task_id=f"mmlupro-{i:05d}",
                kind="multiple_choice",
                question=str(row["question"]).strip(),
                gold=str(answer).strip().upper(),
                dataset="mmlu-pro",
                options=[f"{letters[j]}. {opt}" for j, opt in enumerate(options)],
            )
        )
    return tasks


def render_question(task: Task) -> str:
    """The task as shown to an agent."""
    if task.options:
        return task.question + "\n\n" + "\n".join(task.options)
    return task.question


def split_tasks(
    tasks: list[Task], n_train: int, seed: int = 0
) -> tuple[list[Task], list[Task]]:
    """Deterministic train/eval split.

    Eval is deliberately the larger half: evaluation costs no gradient steps, and
    the statistical-power analysis says a 5-point effect needs ~240 paired items.
    """
    import random

    shuffled = list(tasks)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:n_train], shuffled[n_train:]


def iter_batches(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
