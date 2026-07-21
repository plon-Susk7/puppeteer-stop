# Running the corpus on Kaggle with an open-source model

Self-hosted path: a vLLM server on a Kaggle T4 serving one of the paper's own
**Mimas** models, with the corpus generated against it over the OpenAI-compatible
API. No API key, no per-token cost.

## Why Qwen

The paper's Mimas subspace is Qwen-2.5-7B, Qwen-2.5-14B, LLaMA-3.1-8B,
LLaMA-3.2-3B, Mistral-7B, Mistral-Nemo-12B — so a Qwen/Mistral pool is faithful,
not a substitution.

| Model | Repo | Gating | Fits one T4 |
|---|---|---|---|
| **Qwen2.5-7B-Instruct-AWQ** | `Qwen/Qwen2.5-7B-Instruct-AWQ` | open | yes (~5.5 GB, 4-bit) |
| Qwen2.5-3B-Instruct | `Qwen/Qwen2.5-3B-Instruct` | open | yes (~6 GB, fp16) |
| Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | open | tight (~14 GB fp16) |
| Llama-3.2-3B-Instruct | `meta-llama/Llama-3.2-3B-Instruct` | **gated, manual approval** | yes |

Start with **Qwen2.5-7B-Instruct-AWQ**. The Llama repos need an accepted license
and an HF token, and approval is manual — a bad dependency for a 12-hour session.

## Notebook settings

- **Accelerator:** GPU T4 ×2 (or P100)
- **Internet:** ON — required for the model download and the benchmark loaders
- **Persistence:** files in `/kaggle/working` survive only the session; save the
  trace corpus as a Dataset before the session ends (last cell)

> T4 is Turing: **no bfloat16 and no FlashAttention-2**. Pass `--dtype half`.
> Passing `bfloat16` fails at load with a dtype error.

---

## Cell 1 — install

```python
!pip install -q vllm
!pip install -q requests
```

vLLM pulls its own torch build; if the kernel complains about a torch version
mismatch, restart the kernel once (Run → Restart) and re-run from Cell 2. Do not
`pip install torch` yourself — let vLLM pin it.

## Cell 2 — get the code

```python
!git clone https://github.com/plon-Susk7/puppeteer-stop.git /kaggle/working/puppeteer-stop
%cd /kaggle/working/puppeteer-stop
```

The repo is public, so no token is needed. To pick up later changes without a
fresh clone: `!cd /kaggle/working/puppeteer-stop && git pull`.

## Cell 3 — start the vLLM server in the background

```python
import os, subprocess, time, requests

MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")   # one model on GPU 0
server = subprocess.Popen(
    [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--dtype", "half",              # T4 has no bfloat16
        "--quantization", "awq",
        "--max-model-len", "8192",      # history grows ~6 steps; 8k is ample
        "--gpu-memory-utilization", "0.90",
        "--disable-log-requests",       # 1800 requests would flood the log
        "--port", "8000",
    ],
    env=env,
    stdout=open("/kaggle/working/vllm.log", "w"),
    stderr=subprocess.STDOUT,
)

# First run downloads ~5.5 GB, so allow several minutes.
for i in range(120):
    try:
        if requests.get("http://localhost:8000/health", timeout=2).status_code == 200:
            print(f"server up after ~{i * 5}s"); break
    except Exception:
        pass
    if server.poll() is not None:
        print("server died — last 40 log lines:")
        print("".join(open("/kaggle/working/vllm.log").readlines()[-40:]))
        raise SystemExit(1)
    time.sleep(5)
else:
    raise SystemExit("server did not become healthy in 10 minutes")
```

## Cell 4 — sanity-check the server

Do this *before* launching 1,800 requests.

```python
import requests
r = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": MODEL,
    "messages": [{"role": "user", "content": "Reply with exactly: FINAL ANSWER: 42"}],
    "max_tokens": 32, "temperature": 0.0,
}, timeout=120)
print(r.status_code, r.json()["choices"][0]["message"]["content"])
```

## Cell 5 — verify the harness (no GPU, no cost)

```python
!python experiments/e0_validate_graders.py | tail -1
!python experiments/e0_validate_readout.py | tail -1
!python experiments/e0_smoke_pipeline.py  | tail -1
```

All three must print a pass line before you generate anything.

## Cell 6 — generate the corpus

```python
import os
os.environ["PSTOP_LOCAL_MODEL"]    = MODEL
os.environ["PSTOP_LOCAL_API"]      = "openai"
os.environ["PSTOP_LOCAL_BASE_URL"] = "http://localhost:8000/v1"
os.environ["PSTOP_TRACE_DIR"]      = "/kaggle/working/traces"
os.environ["PSTOP_DATA_DIR"]       = "/kaggle/working/data"

!python experiments/e1_oracle_gap.py generate \
    --dataset gsm-hard --limit 300 --pool local --concurrency 24
```

**`--concurrency` is the setting that matters.** vLLM's throughput comes from
continuous batching; one request at a time leaves the T4 almost idle. 24
concurrent episodes turns a multi-hour serial run into roughly 1–2 hours. Raise
it until throughput stops improving or the KV cache saturates (watch
`vllm.log` for preemption warnings).

Resumable: if the session dies, re-run this cell and it skips completed episodes.

## Cell 7 — the E1 result

```python
!python experiments/e1_oracle_gap.py analyze --dataset gsm-hard --pool local
```

Free replay over the corpus — re-run it as often as you like.

## Cell 8 — persist the corpus before the session ends

```python
!ls -la /kaggle/working/traces
```

Then **Save Version** with "Save output", or add
`/kaggle/working/traces` as a new Kaggle Dataset. The trace corpus is the
project's primary artifact — losing it means paying for it again.

---

## Heterogeneous pool (later)

Kaggle gives **two** T4s, so a genuinely heterogeneous pool is one extra server:
Qwen2.5-7B-AWQ on GPU 0, Mistral-7B-Instruct on GPU 1 (`CUDA_VISIBLE_DEVICES="1"`,
`--port 8001`). That reproduces the paper's non-Mono setting, where agents are
driven by a diverse set of models.

Until then you are running the paper's **Puppeteer-Mono** condition — one model
driving every agent — which is a published configuration in Table 1, not a
shortcut.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ValueError: Bfloat16 is only supported on GPUs with compute capability of at least 8.0` | T4 is 7.5 — pass `--dtype half` |
| OOM at startup | Lower `--gpu-memory-utilization` to 0.85, or `--max-model-len` to 4096 |
| Server dies silently | `!tail -50 /kaggle/working/vllm.log` |
| Throughput flat as concurrency rises | KV cache is saturated — check the log for preemption; lower `--max-model-len` to fit more sequences |
| Very slow first request | Model download plus CUDA graph capture; only the first is slow |
| `datasets-server` errors in the loader | Internet is off in notebook settings |
