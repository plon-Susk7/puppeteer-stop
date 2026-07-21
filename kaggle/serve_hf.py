"""A minimal OpenAI-compatible server over HuggingFace transformers.

Fallback for when vLLM will not run. vLLM is the faster option, but its V1
engine defaults to attention kernels that require compute capability >= 8.0,
and Kaggle's T4 is 7.5 (Turing) — so depending on the version it may crash at
load. This server has no such constraint: plain `transformers` + `torch`, which
Kaggle already has installed.

It speaks just enough of `/v1/chat/completions` for this project, so the corpus
generator runs against it unchanged:

    python kaggle/serve_hf.py --model Qwen/Qwen2.5-3B-Instruct --port 8000

Requests are **batched**. A background worker drains the queue every few
milliseconds and runs one `generate()` over the whole batch, which is what makes
concurrent episode generation worth anything on a single GPU — one-at-a-time
decoding leaves the card almost idle.

Deliberately not a general-purpose server: no streaming, no sampling knobs
beyond temperature, no auth.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Wait this long for more requests to join a batch before running it. Long
# enough to actually fill batches under concurrency, short enough to be
# invisible next to multi-hundred-token generations.
BATCH_WINDOW_S = 0.02


class _Job:
    __slots__ = (
        "prompt", "max_tokens", "temperature", "done",
        "text", "error", "n_prompt", "n_completion",
    )

    def __init__(self, prompt: str, max_tokens: int, temperature: float) -> None:
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.done = threading.Event()
        self.text: str = ""
        self.error: str | None = None
        self.n_prompt: int = 0
        self.n_completion: int = 0


class BatchEngine:
    def __init__(self, model_id: str, dtype: torch.dtype, max_batch: int) -> None:
        print(f"loading {model_id} ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Decoder-only batch generation requires LEFT padding: with right
        # padding the pad tokens land between the prompt and the first
        # generated token, and short sequences in the batch produce garbage.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # `torch_dtype` was renamed to `dtype`; accept whichever this
        # transformers version wants rather than pinning a version.
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=dtype, device_map="auto"
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=dtype, device_map="auto"
            )
        self.model.eval()
        self.max_batch = max_batch
        self.queue: queue.Queue[_Job] = queue.Queue()
        # HuggingFace "fast" tokenizers are Rust-backed with interior
        # mutability. Calling one from two threads at once -- especially when a
        # call mutates padding state, as batch encoding does -- panics with
        # "RuntimeError: Already borrowed". All tokenizer access goes through
        # this lock.
        self.tok_lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()
        print(f"ready: {model_id} on {self.model.device}", flush=True)

    def render(self, messages: list[dict]) -> str:
        """Chat template, under the tokenizer lock (handler threads call this)."""
        with self.tok_lock:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def submit(self, job: _Job) -> None:
        self.queue.put(job)

    def _drain(self) -> list[_Job]:
        batch = [self.queue.get()]           # block for the first
        deadline = time.time() + BATCH_WINDOW_S
        while len(batch) < self.max_batch:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                batch.append(self.queue.get(timeout=remaining))
            except queue.Empty:
                break
        return batch

    def _loop(self) -> None:
        while True:
            batch = self._drain()
            try:
                self._run(batch)
            except Exception as exc:  # noqa: BLE001
                for job in batch:
                    job.error = f"{type(exc).__name__}: {exc}"
            finally:
                for job in batch:
                    job.done.set()

    @torch.inference_mode()
    def _run(self, batch: list[_Job]) -> None:
        # One generate() per batch, so every job shares the longest max_tokens
        # and the first temperature. Fine here: the corpus is generated with a
        # single fixed config.
        max_new = max(j.max_tokens for j in batch)
        temperature = batch[0].temperature

        with self.tok_lock:
            enc = self.tokenizer(
                [j.prompt for j in batch], return_tensors="pt", padding=True
            ).to(self.model.device)

        pad_id = self.tokenizer.pad_token_id
        kwargs = {"max_new_tokens": max_new, "pad_token_id": pad_id}
        if temperature and temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature)
        else:
            kwargs.update(do_sample=False)

        out = self.model.generate(**enc, **kwargs)
        prompt_len = enc["input_ids"].shape[1]

        with self.tok_lock:
            for job, seq in zip(batch, out):
                generated = seq[prompt_len:]
                job.text = self.tokenizer.decode(generated, skip_special_tokens=True)
                job.n_prompt = prompt_len
                # Count real generated tokens here rather than re-tokenizing the
                # text in the handler thread: it is exact, free, and keeps every
                # tokenizer call on this one thread.
                n = len(generated)
                while n > 0 and int(generated[n - 1]) == pad_id:
                    n -= 1
                job.n_completion = n

        print(f"batch={len(batch):2d} prompt_len={prompt_len:5d} new<={max_new}", flush=True)


class Handler(BaseHTTPRequestHandler):
    engine: BatchEngine = None       # type: ignore[assignment]
    model_id: str = ""

    def log_message(self, *args) -> None:  # noqa: D102 - silence per-request noise
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("/health", "/v1/models"):
            self._send(200, {"status": "ok", "model": self.model_id})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("content-length", 0))
            req = json.loads(self.rfile.read(length))
            messages = req["messages"]
        except Exception as exc:  # noqa: BLE001
            self._send(400, {"error": f"bad request: {exc}"})
            return

        prompt = self.engine.render(messages)
        job = _Job(
            prompt=prompt,
            max_tokens=int(req.get("max_tokens", 700)),
            temperature=float(req.get("temperature", 0.0)),
        )
        self.engine.submit(job)
        job.done.wait()

        if job.error:
            self._send(500, {"error": job.error})
            return

        self._send(
            200,
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "model": self.model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": job.text},
                        "finish_reason": "stop",
                    }
                ],
                # Usage drives every efficiency claim in the project, so it is
                # reported honestly rather than zero-filled.
                "usage": {
                    "prompt_tokens": job.n_prompt,
                    "completion_tokens": job.n_completion,
                    "total_tokens": job.n_prompt + job.n_completion,
                },
            },
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument(
        "--dtype",
        default="float16",
        choices=("float16", "bfloat16", "float32"),
        help="T4 is Turing and has no bfloat16 — keep float16 there",
    )
    args = ap.parse_args()

    Handler.engine = BatchEngine(
        args.model, getattr(torch, args.dtype), args.max_batch
    )
    Handler.model_id = args.model

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"listening on http://localhost:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
