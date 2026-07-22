"""Run configuration and agent-pool wiring.

The two pools map onto the paper's Titan (large) and Mimas (small) subspaces,
which is what makes E4 nearly free: the capability split and the experimental
axis are the same axis.

Defaults need only `ANTHROPIC_API_KEY`:

    strong (Titan)  claude-opus-4-8
    cheap  (Mimas)  claude-haiku-4-5

Every field is overridable per pool, so a self-hosted model can stand in for
either side without touching code (on Kaggle, set these from Kaggle Secrets):

    PSTOP_<POOL>_MODEL / _API / _BASE_URL / _API_KEY / _MAX_TOKENS / _TEMPERATURE
    PSTOP_<POOL>_DISABLE_THINKING=1   # only needed for models that think by default

e.g. a local vLLM server as the cheap pool:

    PSTOP_CHEAP_API=openai
    PSTOP_CHEAP_MODEL=Qwen/Qwen2.5-7B-Instruct
    PSTOP_CHEAP_BASE_URL=http://localhost:8000/v1
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .llm import LLMClient, LLMRouter, ResponseCache

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("PSTOP_DATA_DIR", ROOT / "data"))
TRACE_DIR = Path(os.environ.get("PSTOP_TRACE_DIR", ROOT / "traces"))
CACHE_PATH = Path(os.environ.get("PSTOP_CACHE", DATA_DIR / "responses.sqlite"))

# Fixed rollout depth for the diagnostic. Long enough to observe abandonment,
# short enough that the corpus stays affordable.
DEPTH = 6

DEFAULT_TEMPERATURE = 0.0

# The paper's agents emit a short REASONING RESULT plus a FINAL ANSWER, so this
# is already generous. It matters more than it looks on a self-hosted server:
# a batch generates until the longest request finishes, so max_tokens sets the
# floor on how long every batch takes.
DEFAULT_MAX_TOKENS = 512

# Seconds. Generous because one self-hosted batch can take minutes; too low and
# the client hangs up mid-generation and the whole batch fails.
DEFAULT_TIMEOUT = 900


# Pool presets. `local` is the open-source path: a vLLM server serving one of
# the paper's own Mimas models. Qwen is used rather than LLaMA because the
# Llama-3.2 repos are gated behind manual approval on HuggingFace, which is a
# bad dependency for a time-boxed Kaggle session.
POOL_DEFAULTS: dict[str, dict[str, object]] = {
    "local": {                       # Mimas analogue, self-hosted
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "api": "openai",
        "base_url": "http://localhost:8000/v1",
    },
    "strong": {"model": "claude-opus-4-8", "api": "anthropic"},   # Titan analogue
    "cheap": {"model": "claude-haiku-4-5", "api": "anthropic"},   # Mimas analogue
}

POOL_NAMES = tuple(POOL_DEFAULTS)


@dataclass
class PoolSpec:
    """One agent-model backend."""

    name: str            # "local" | "strong" | "cheap"
    model: str
    api: str = "anthropic"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    disable_thinking: bool = False
    routing: str = "round_robin"
    timeout: int = DEFAULT_TIMEOUT

    @property
    def models(self) -> list[str]:
        return [m.strip() for m in self.model.split(",") if m.strip()]

    @property
    def base_urls(self) -> list[str | None]:
        if not self.base_url:
            return [None]
        return [u.strip() for u in self.base_url.split(",") if u.strip()]

    @property
    def backends(self) -> list[tuple[str, str | None]]:
        """(model, base_url) per backend.

        One model with several URLs is replication — the same model served on
        each GPU, for throughput. Several models is a heterogeneous pool. A
        shorter list is cycled, so `model=A,B` with a single URL is also valid
        (two models behind one server).
        """
        models, urls = self.models, self.base_urls
        width = max(len(models), len(urls))
        return [(models[i % len(models)], urls[i % len(urls)]) for i in range(width)]

    @staticmethod
    def from_env(name: str) -> "PoolSpec | None":
        prefix = f"PSTOP_{name.upper()}"
        preset = POOL_DEFAULTS.get(name, {})
        model = os.environ.get(f"{prefix}_MODEL") or preset.get("model")
        if not model:
            return None
        return PoolSpec(
            name=name,
            model=str(model),
            api=os.environ.get(f"{prefix}_API") or str(preset.get("api", "anthropic")),
            base_url=os.environ.get(f"{prefix}_BASE_URL") or preset.get("base_url"),  # type: ignore[arg-type]
            api_key=os.environ.get(f"{prefix}_API_KEY"),
            temperature=float(os.environ.get(f"{prefix}_TEMPERATURE", DEFAULT_TEMPERATURE)),
            max_tokens=int(os.environ.get(f"{prefix}_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
            disable_thinking=os.environ.get(f"{prefix}_DISABLE_THINKING", "") == "1",
            routing=os.environ.get(f"{prefix}_ROUTING", "round_robin"),
            timeout=int(os.environ.get(f"{prefix}_TIMEOUT", DEFAULT_TIMEOUT)),
        )


@dataclass
class RunConfig:
    pool: PoolSpec
    depth: int = DEPTH
    seed: int = 0
    cache_path: Path = field(default_factory=lambda: CACHE_PATH)

    def client(self) -> LLMClient | LLMRouter:
        """One client per backend, behind a router when there is more than one.

        The cache is shared across backends by design: its key includes the
        model, so replicas of the same model hit the same entries and a rerun
        after a crash costs nothing regardless of which GPU served it first.
        """
        cache = ResponseCache(self.cache_path)
        clients = [
            LLMClient(
                model=model,
                api=self.pool.api,
                base_url=base_url,
                api_key=self.pool.api_key,
                cache=cache,
                temperature=self.pool.temperature,
                max_tokens=self.pool.max_tokens,
                seed=self.seed,
                disable_thinking=self.pool.disable_thinking,
                timeout=self.pool.timeout,
            )
            for model, base_url in self.pool.backends
        ]
        if len(clients) == 1:
            return clients[0]
        return LLMRouter(clients, strategy=self.pool.routing)


def resolve_pool(name: str) -> PoolSpec:
    spec = PoolSpec.from_env(name)
    if spec is None:
        raise SystemExit(
            f"Pool {name!r} has no default and is not configured.\n"
            f"Set PSTOP_{name.upper()}_MODEL (plus _API / _BASE_URL / _API_KEY as needed)."
        )
    if spec.api == "anthropic" and not (
        spec.api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    ):
        # Not fatal: the SDK also resolves an `ant auth login` profile from disk.
        print(
            "note: no ANTHROPIC_API_KEY in the environment — the SDK will fall back to "
            "an `ant auth login` profile if one exists.\n"
            "      A Claude subscription is billed separately and is not an API key; "
            "create one at console.anthropic.com.\n"
        )
    return spec
