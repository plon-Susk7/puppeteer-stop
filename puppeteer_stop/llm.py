"""LLM client with an on-disk response cache.

Two backends behind one call site:

  "anthropic" — the official `anthropic` SDK. Both agent pools run here by
                default (Opus 4.8 as the strong/Titan pool, Haiku 4.5 as the
                cheap/Mimas pool).
  "openai"    — OpenAI-compatible chat completions over plain HTTP. Kept for
                self-hosted models (a local vLLM server) and other providers,
                which is the only reason a second backend exists.

**Thinking is deliberately left off.** Each agent is supposed to be one atomic
reasoning behavior — the paper's `r` in `a = (m, r, t)` — so extended thinking
would confound the reasoning-pattern variable with an invisible second reasoning
process, and inflate cost. Omitting the `thinking` parameter yields no thinking
on Opus 4.8 and Haiku 4.5. Sonnet 5 runs adaptive thinking when the parameter is
omitted, so `PSTOP_<POOL>_DISABLE_THINKING=1` sends an explicit disable for it.

The cache is the load-bearing part. Every experiment after trajectory generation
is replay, so a cache hit is the difference between a free re-analysis and paying
for the corpus twice.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading  # noqa: F401  (used by ResponseCache and LLMRouter)
import time
from dataclasses import dataclass
from typing import Any, Sequence

import requests

DEFAULT_TIMEOUT = 180
MAX_RETRIES = 5


@dataclass(frozen=True)
class Completion:
    """One model response, plus the accounting the efficiency claims depend on."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    model_version: str  # provider-reported id; drifts under us, so we record it
    cached: bool = False
    latency_s: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ResponseCache:
    """sqlite-backed cache keyed on everything that can change a response.

    Single file so it ships as a Kaggle Dataset. Thread-safe via a lock rather
    than a connection pool — notebook workloads are effectively single-writer and
    the simplicity is worth more than the throughput.
    """

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = str(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS responses (
                key             TEXT PRIMARY KEY,
                text            TEXT NOT NULL,
                prompt_tokens   INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                model           TEXT NOT NULL,
                model_version   TEXT NOT NULL,
                created_at      REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    @staticmethod
    def key(
        provider: str,
        model: str,
        messages: Sequence[dict],
        temperature: float,
        max_tokens: int,
        seed: int | None,
    ) -> str:
        payload = json.dumps(
            {
                "provider": provider,
                "model": model,
                "messages": list(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Completion | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT text, prompt_tokens, completion_tokens, model, model_version"
                " FROM responses WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return Completion(
            text=row[0],
            prompt_tokens=row[1],
            completion_tokens=row[2],
            model=row[3],
            model_version=row[4],
            cached=True,
        )

    def put(self, key: str, c: Completion) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?,?)",
                (
                    key,
                    c.text,
                    c.prompt_tokens,
                    c.completion_tokens,
                    c.model,
                    c.model_version,
                    time.time(),
                ),
            )
            self._conn.commit()

    def __len__(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]


class LLMClient:
    """Talks to one endpoint. Instantiate one per model in the pool.

    `api` selects the wire format:
      "openai"    — /chat/completions   (OpenAI, OpenRouter, Together, vLLM, ...)
      "anthropic" — /v1/messages
    """

    def __init__(
        self,
        model: str,
        *,
        api: str = "anthropic",
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        cache: ResponseCache | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = 0,
        disable_thinking: bool = False,
    ) -> None:
        if api not in {"openai", "anthropic"}:
            raise ValueError(f"unknown api {api!r}; expected 'openai' or 'anthropic'")
        self.model = model
        self.api = api
        self.base_url = base_url.rstrip("/") if base_url else None
        self.api_key = api_key or (os.environ.get(api_key_env) if api_key_env else None)
        self.cache = cache
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.disable_thinking = disable_thinking
        self._session = requests.Session() if api == "openai" else None
        self._client = self._build_anthropic_client() if api == "anthropic" else None

    def _build_anthropic_client(self):
        try:
            import anthropic
        except ImportError as exc:  # noqa: BLE001
            raise SystemExit(
                "The `anthropic` package is required for the Anthropic backend.\n"
                "  pip install anthropic"
            ) from exc

        kwargs: dict[str, Any] = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        # With no api_key, the SDK resolves credentials from the environment
        # (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an `ant auth login`
        # profile) — so a bare client is correct, not a bug.
        return anthropic.Anthropic(**kwargs)

    def complete(
        self,
        messages: Sequence[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        route_key: str | None = None,  # ignored; kept so a router can substitute
    ) -> Completion:
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        seed = self.seed if seed is None else seed

        key = None
        if self.cache is not None:
            key = ResponseCache.key(
                self.api, self.model, messages, temperature, max_tokens, seed
            )
            hit = self.cache.get(key)
            if hit is not None:
                return hit

        started = time.time()
        if self.api == "anthropic":
            # The SDK retries 429/5xx/connection errors with backoff itself.
            completion = self._request_anthropic(messages, temperature, max_tokens)
        else:
            completion = self._request_with_retries(messages, temperature, max_tokens, seed)
        completion = Completion(**{**completion.__dict__, "latency_s": time.time() - started})

        if self.cache is not None and key is not None:
            self.cache.put(key, completion)
        return completion

    def _request_anthropic(
        self, messages: Sequence[dict], temperature: float, max_tokens: int
    ) -> Completion:
        import anthropic

        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": turns,
        }
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature
        if self.disable_thinking:
            # Needed only for models that run adaptive thinking when the
            # parameter is omitted (Sonnet 5). Opus 4.8 and Haiku 4.5 already
            # run without thinking when it is absent.
            kwargs["thinking"] = {"type": "disabled"}

        try:
            message = self._client.messages.create(**kwargs)
        except anthropic.AuthenticationError as exc:
            raise SystemExit(
                "Anthropic authentication failed. Set ANTHROPIC_API_KEY to a key from\n"
                "console.anthropic.com (a Claude subscription is billed separately and\n"
                "is not an API key), or run `ant auth login`."
            ) from exc
        except anthropic.NotFoundError as exc:
            raise SystemExit(
                f"Model {self.model!r} was not found. Check the model id — current ids "
                "include claude-opus-4-8, claude-sonnet-5, claude-haiku-4-5."
            ) from exc

        # A refusal returns HTTP 200 with an empty or partial `content`, so
        # stop_reason must be checked before reading blocks.
        if message.stop_reason == "refusal":
            text = ""
        else:
            text = "".join(b.text for b in message.content if b.type == "text")

        return Completion(
            text=text,
            prompt_tokens=message.usage.input_tokens,
            completion_tokens=message.usage.output_tokens,
            model=self.model,
            model_version=message.model,
        )

    def _request_with_retries(
        self,
        messages: Sequence[dict],
        temperature: float,
        max_tokens: int,
        seed: int | None,
    ) -> Completion:
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                return self._request(messages, temperature, max_tokens, seed)
            except requests.ConnectionError as exc:
                # Nothing is listening. Retrying cannot help, and with high
                # concurrency a retry storm buries the real cause under a wall
                # of identical failures.
                raise RuntimeError(
                    f"cannot reach the model server at {self.base_url}.\n"
                    f"  Is it running? Start it first (see kaggle/KAGGLE.md Cell 3),\n"
                    f"  then re-run this command — completed episodes are skipped.\n"
                    f"  underlying error: {exc}"
                ) from exc
            except (requests.RequestException, _RetriableStatus) as exc:
                last_error = exc
                if attempt == MAX_RETRIES - 1:
                    break
                # Exponential backoff. Rate limits are the common case and they
                # clear on their own; failing the whole corpus run because of one
                # 429 would cost hours of regeneration.
                time.sleep(min(2**attempt, 30))
        raise RuntimeError(
            f"{self.model}: request failed after {MAX_RETRIES} attempts — "
            f"last error: {type(last_error).__name__}: {last_error}"
        ) from last_error

    def _request(
        self,
        messages: Sequence[dict],
        temperature: float,
        max_tokens: int,
        seed: int | None,
    ) -> Completion:
        base_url = self.base_url or "https://api.openai.com/v1"
        url = f"{base_url}/chat/completions"
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        body: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Local vLLM servers reject unknown fields; only send seed when set.
        if seed is not None:
            body["seed"] = seed

        resp = self._session.post(url, headers=headers, json=body, timeout=DEFAULT_TIMEOUT)
        if resp.status_code in (408, 409, 429) or resp.status_code >= 500:
            raise _RetriableStatus(f"HTTP {resp.status_code}: {resp.text[:300]}")
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return Completion(
            text=choice["message"]["content"] or "",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            model=self.model,
            model_version=data.get("model", self.model),
        )


class _RetriableStatus(Exception):
    """HTTP status that warrants a backoff rather than a hard failure."""


class LLMRouter:
    """Spreads requests across several backends. Substitutable for LLMClient.

    Kaggle gives two T4s, and the same mechanism covers both things you'd want
    to do with them:

      `round_robin` — the same model served on every GPU. Pure throughput:
                      corpus generation time drops roughly linearly in the
                      number of backends.

      `per_agent`   — a different model per GPU, with each agent bound to one
                      of them by a stable hash of its name. This is the paper's
                      non-Mono setting, where agents are driven by a diverse set
                      of models.

    `per_agent` binds deterministically rather than randomly on purpose: an
    agent that answers from a different model on every activation would make the
    resulting trace uninterpretable, since agent identity and model identity
    would be confounded within a single episode.
    """

    def __init__(self, clients: Sequence[LLMClient], strategy: str = "round_robin") -> None:
        if not clients:
            raise ValueError("LLMRouter needs at least one client")
        if strategy not in {"round_robin", "per_agent"}:
            raise ValueError(
                f"unknown routing strategy {strategy!r}; "
                "expected 'round_robin' or 'per_agent'"
            )
        self.clients = list(clients)
        self.strategy = strategy
        self._next = 0
        self._lock = threading.Lock()

    @property
    def model(self) -> str:
        """Label for logs; per-step models are recorded on each StepRecord."""
        names = sorted({c.model for c in self.clients})
        return names[0] if len(names) == 1 else "+".join(names)

    def _pick(self, route_key: str | None) -> LLMClient:
        if len(self.clients) == 1:
            return self.clients[0]
        if self.strategy == "per_agent" and route_key is not None:
            # sha256 rather than hash(): Python salts str hashing per process,
            # so hash() would assign agents to models differently on every run.
            digest = hashlib.sha256(route_key.encode()).digest()
            return self.clients[digest[0] % len(self.clients)]
        with self._lock:
            client = self.clients[self._next % len(self.clients)]
            self._next += 1
        return client

    def complete(
        self,
        messages: Sequence[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        route_key: str | None = None,
    ) -> Completion:
        return self._pick(route_key).complete(
            messages, temperature=temperature, max_tokens=max_tokens, seed=seed
        )

    def assignments(self, keys: Sequence[str]) -> dict[str, str]:
        """Which model each route key resolves to — for logging a run's setup."""
        return {k: self._pick(k).model for k in keys}
