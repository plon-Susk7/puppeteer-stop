# puppeteer-stop

**Knowing When to Stop: Learned Termination for Multi-Agent Reasoning**

Research prototype extending *Multi-Agent Collaboration via Evolving Orchestration*
("Puppeteer", Dang & Qian et al., NeurIPS 2025 — [arXiv:2505.19591](https://arxiv.org/abs/2505.19591)).

## The claim

Multi-agent orchestrators conflate *whom to call next* with *whether to continue at
all*, train both from a single end-of-episode scalar, and consequently talk themselves
out of correct answers. This repo measures that loss, then fixes it by making
termination a separately-parameterized, densely-supervised decision.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Instrumentation: cache, traces, graders, readout | **done — all checks green** |
| 1 | Trajectory corpus generation | ready; needs an API key |
| 2 | **E1 — oracle stopping gap** (the headline diagnostic) | code done, awaiting corpus |
| — | B0/B1 baselines (incl. matched-budget self-consistency) | not started |
| 3 | E3 — two-head orchestrator, 2×2 ablation | not started |
| 4 | E4 — pool capability + calibration | not started |

## Verify the install (no API key, no cost)

```bash
python experiments/e0_validate_graders.py    # extraction + grading + dataset loading
python experiments/e0_validate_readout.py    # answer@t and oracle-gap metrics
python experiments/e0_smoke_pipeline.py      # full chain against a mock provider
python experiments/e0_validate_serve_hf.py   # fallback server (stubbed torch, no GPU)
```

All four must pass before generating a corpus.

## Generate a corpus and run E1

Default path is **self-hosted open-source** — no API key, no per-token cost.
Start a vLLM server, then:

```bash
python experiments/e1_oracle_gap.py generate \
    --dataset gsm-hard --limit 300 --pool local --concurrency 24
python experiments/e1_oracle_gap.py analyze --dataset gsm-hard --pool local
```

Full Kaggle walkthrough (server flags, T4 caveats, troubleshooting):
**[`kaggle/KAGGLE.md`](kaggle/KAGGLE.md)**

Generation is resumable — re-running after a killed session skips completed
episodes. Analysis is free replay and can be re-run at will.

## Agent pools

Pools mirror the paper's capability split, which is what makes E4 nearly free —
the cost axis and the experimental axis are the same axis.

| Pool | Paper | Default model | Cost for a 300-task corpus |
|---|---|---|---|
| `local` | Mimas (small) | `Qwen/Qwen2.5-7B-Instruct-AWQ` | free (~1–2 h on a Kaggle T4) |
| `cheap` | Mimas (small) | `claude-haiku-4-5` | ~$6 |
| `strong` | Titan (large) | `claude-opus-4-8` | ~$29 |

Qwen2.5-7B is one of the paper's own Mimas models, so `local` is faithful rather
than a substitution. Llama-3.2 would be too, but those repos are gated behind
manual approval on HuggingFace — a bad dependency inside a 12-hour session.

Every field is overridable per pool:

```bash
export PSTOP_LOCAL_MODEL=mistralai/Mistral-7B-Instruct-v0.3
export PSTOP_LOCAL_BASE_URL=http://localhost:8001/v1
```

### Two notes that affect results

**`--concurrency` is not a tuning knob, it's the difference between 1 hour and
10.** vLLM's throughput comes from continuous batching; one request at a time
leaves the GPU idle. Episodes are internally sequential (step *t+1* conditions
on step *t*) but tasks are independent, so concurrency runs whole episodes in
parallel. Use 16–32 against a local server, 2–4 against a rate-limited API.

**Thinking is deliberately off.** Each agent is one atomic reasoning behavior
(the paper's `r` in `a = (m, r, t)`); extended thinking would confound the
reasoning-pattern variable with a second, invisible reasoning process. Qwen
instruct models don't think by default. On the Anthropic pools, Opus 4.8 and
Haiku 4.5 run without thinking when the parameter is omitted; a model that
thinks by default (Sonnet 5) needs `PSTOP_<POOL>_DISABLE_THINKING=1`.

## Design constraints

- **No vendor SDKs.** Everything speaks HTTP via `requests`, so the same code runs
  against OpenAI-compatible endpoints, Anthropic, or a local vLLM server on Kaggle.
- **Offline-first.** Trajectories are generated once and cached; all analysis and all
  stop-head training is replay over that corpus. Only on-policy selection training
  needs fresh rollouts, and that is explicitly a stretch goal.
- **Everything is resumable.** Kaggle sessions die at 12 hours. No loop may lose more
  than one task's worth of work.

## Layout

```
puppeteer_stop/
  config.py      run configuration
  llm.py         provider-agnostic client + sqlite response cache
  tasks.py       GSM-Hard / MMLU-Pro loaders and graders
  agents.py      agent pool; prompts ported from the paper's Appendix B
  trace.py       append-only JSONL orchestration traces
  readout.py     answer@t  —  `last` and `vote`
  rollout.py     trajectory generation, checkpointed
experiments/     E0-E4 runners
analysis/        oracle gap, calibration, topology metrics
```

Traces are JSONL (append-only, inspectable, ships as a Kaggle Dataset).
The response cache is sqlite (compact, fast, single file).
