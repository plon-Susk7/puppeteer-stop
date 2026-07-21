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

## Two serving paths — read this before Cell 1

⚠️ **vLLM is not reliable on T4.** Turing is compute capability **7.5**; vLLM's
V1 engine defaults to attention kernels requiring **≥ 8.0**, and some recent
versions crash at load on `sm_75`. Depending on the version you get, it may work
perfectly or fail immediately.

| | Path A — vLLM | Path B — transformers |
|---|---|---|
| Speed | fast (continuous batching) | slower (windowed batching) |
| Reliability on T4 | version-dependent | works |
| Install | `pip install vllm` (~5 min) | already on Kaggle |
| Model | `Qwen2.5-7B-Instruct-AWQ` | `Qwen2.5-3B-Instruct` |

**Try Path A. If the server does not come up within one attempt, switch to
Path B rather than debugging** — B is in this repo (`kaggle/serve_hf.py`), speaks
the same OpenAI-compatible API, and every later cell is identical.

---

## Cell 1 — install (Path A only)

```python
!pip install -q vllm
```

vLLM pins its own torch build. If the kernel complains about a torch mismatch,
restart once (Run → Restart) and resume at Cell 2. Do **not** `pip install torch`
yourself.

Skip this cell entirely for Path B.

## Cell 1.5 — HuggingFace token (optional, but do it)

Without a token you will see:

```
Warning: You are sending unauthenticated requests to the HF Hub.
Please set a HF_TOKEN to enable higher rate limits and faster downloads.
```

Harmless — the Qwen and Mistral repos are public and still download — but
unauthenticated pulls are rate-limited, and you are fetching several GB inside a
time-boxed session. Set the token and the download stops being the slow part.

1. Create a token at <https://huggingface.co/settings/tokens> (`read` scope)
2. Notebook → **Add-ons → Secrets** → add it as `HF_TOKEN`
3. Load it **before** starting the server:

```python
from kaggle_secrets import UserSecretsClient
import os
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

A token is also what unlocks the **gated** repos (`meta-llama/*`) once you want
LLaMA-3.2-3B in the pool for the heterogeneous setting — but those additionally
need the license accepted and manual approval on the model page, so request
access well before you need it.

Weights are cached in `~/.cache/huggingface` for the life of the session, so
setting the token after a download has already completed changes nothing.

## Cell 2 — get the code

```python
!git clone https://github.com/plon-Susk7/puppeteer-stop.git /kaggle/working/puppeteer-stop
%cd /kaggle/working/puppeteer-stop
```

The repo is public, so no token is needed. To pick up later changes without a
fresh clone: `!cd /kaggle/working/puppeteer-stop && git pull`.

## Cell 3 — start the server in the background

Both paths use the same wait-for-health loop, so run whichever block applies and
continue to Cell 4 either way.

### Path A — vLLM

```python
import os, subprocess, time, requests

MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
LOG   = "/kaggle/working/server.log"

env = dict(
    os.environ,
    CUDA_VISIBLE_DEVICES="0",        # one model on GPU 0
    VLLM_USE_V1="0",                 # V1 engine assumes compute capability >= 8.0
    VLLM_ATTENTION_BACKEND="XFORMERS",  # FlashAttention 2/3 are unavailable on Turing
)
server = subprocess.Popen(
    [
        "vllm", "serve", MODEL,
        "--dtype", "half",               # T4 has no bfloat16
        "--quantization", "awq",
        "--max-model-len", "8192",       # history grows over ~6 steps; 8k is ample
        "--gpu-memory-utilization", "0.90",
        "--disable-log-requests",        # 1800 requests would flood the log
        "--port", "8000",
    ],
    env=env, stdout=open(LOG, "w"), stderr=subprocess.STDOUT,
)
```

### Path B — transformers fallback

```python
import os, subprocess, time, requests

MODEL = "Qwen/Qwen2.5-3B-Instruct"   # fp16 ~6 GB, comfortable on one T4
LOG   = "/kaggle/working/server.log"

server = subprocess.Popen(
    ["python", "kaggle/serve_hf.py",
     "--model", MODEL, "--port", "8000", "--max-batch", "16", "--dtype", "float16"],
    env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"),
    stdout=open(LOG, "w"), stderr=subprocess.STDOUT,
)
```

### Then, for either path — wait for health

```python
# First run downloads several GB, so allow time.
for i in range(180):
    try:
        if requests.get("http://localhost:8000/health", timeout=2).status_code == 200:
            print(f"server up after ~{i * 5}s"); break
    except Exception:
        pass
    if server.poll() is not None:
        print("server died — last 40 log lines:")
        print("".join(open(LOG).readlines()[-40:]))
        raise SystemExit("server failed to start (see log above)")
    time.sleep(5)
else:
    raise SystemExit("server did not become healthy in 15 minutes")
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

## Using both T4s

Kaggle gives two GPUs, and there are two different things worth doing with them.
Both work the same way: start one server per GPU, then give the pool a
comma-separated list of URLs.

### A. Replication — same model twice, for speed

Roughly halves corpus generation time. Nothing about the experiment changes.

```python
import os, subprocess, time, requests

MODEL = "Qwen/Qwen2.5-3B-Instruct"
procs = []
for gpu, port in [(0, 8000), (1, 8001)]:
    procs.append(subprocess.Popen(
        ["python", "/kaggle/working/puppeteer-stop/kaggle/serve_hf.py",
         "--model", MODEL, "--port", str(port), "--max-batch", "16"],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
        stdout=open(f"/kaggle/working/server{port}.log", "w"),
        stderr=subprocess.STDOUT,
    ))

for port in (8000, 8001):
    for _ in range(180):
        try:
            if requests.get(f"http://localhost:{port}/health", timeout=2).status_code == 200:
                print(f"port {port} up"); break
        except Exception: pass
        time.sleep(5)
    else:
        raise SystemExit(f"port {port} never came up")

os.environ["PSTOP_LOCAL_MODEL"]    = MODEL
os.environ["PSTOP_LOCAL_BASE_URL"] = "http://localhost:8000/v1,http://localhost:8001/v1"
os.environ["PSTOP_LOCAL_ROUTING"] = "round_robin"
```

Then raise concurrency, since there are now two servers to keep busy:

```python
!python experiments/e1_oracle_gap.py generate \
    --dataset gsm-hard --limit 300 --pool local --concurrency 48
```

### B. Heterogeneous pool — different model per GPU

This is the paper's **non-Mono** setting, where agents are driven by a diverse
set of models. It is an experimental condition, not an optimization.

```python
os.environ["PSTOP_LOCAL_MODEL"] = (
    "Qwen/Qwen2.5-3B-Instruct,mistralai/Mistral-7B-Instruct-v0.3"
)
os.environ["PSTOP_LOCAL_BASE_URL"] = "http://localhost:8000/v1,http://localhost:8001/v1"
os.environ["PSTOP_LOCAL_ROUTING"] = "per_agent"
```

(Serve Qwen on GPU 0 and Mistral on GPU 1 — same loop as above with a different
model per port. Mistral-7B in fp16 is ~14 GB and tight on a T4; prefer vLLM with
an AWQ build, or a smaller second model.)

`per_agent` binds each agent to **one** model by a stable hash of its name, so
`critique` is always answered by the same model within and across runs. That is
deliberate: an agent whose backing model changed per activation would confound
agent identity with model identity inside a single episode. The binding is
printed at the start of the run and each step's model is recorded in the trace.

`round_robin` would be wrong here — it would scatter agents across models
randomly and make the traces uninterpretable.

### Which am I running?

With one model and one URL you are running the paper's **Puppeteer-Mono**
condition — one model driving every agent. That is a published row in Table 1,
not a shortcut. Replication (A) is still Mono. Only (B) is the multi-model
setting.

### Bigger models via tensor parallelism

To run a model that does not fit on one T4 (e.g. `Qwen2.5-14B-Instruct-AWQ`),
shard it across both GPUs instead of replicating — vLLM only:

```
vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ --tensor-parallel-size 2 --dtype half --quantization awq
```

This gives one bigger model rather than two servers, so throughput does not
improve — spend the second GPU this way only if capability is the bottleneck.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Bfloat16 is only supported on GPUs with compute capability of at least 8.0` | T4 is 7.5 — `--dtype half` (Path A) / `--dtype float16` (Path B) |
| `FA3 is only supported on devices with compute capability >= 8` | vLLM V1 engine — set `VLLM_USE_V1=0` and `VLLM_ATTENTION_BACKEND=XFORMERS` |
| vLLM crashes detecting `sm_75` / CUTLASS error | Known Turing breakage in some vLLM builds. **Switch to Path B** — don't burn quota bisecting versions |
| OOM at startup | Lower `--gpu-memory-utilization` to 0.85 or `--max-model-len` to 4096 (Path A); use the 3B model (Path B) |
| Server dies silently | `!tail -50 /kaggle/working/server.log` |
| Throughput flat as concurrency rises | Path A: KV cache saturated — check the log for preemption. Path B: raise `--max-batch` to match `--concurrency` |
| Path B output is garbled or empty | Left-padding or chat template mismatch — confirm Cell 4 returns clean text before generating a corpus |
| Very slow first request | Model download plus warm-up; only the first is slow |
| `datasets-server` errors in the loader | Internet is off in notebook settings |
