"""E0 — validate the transformers fallback server without a GPU.

`kaggle/serve_hf.py` is the path that runs when vLLM won't start on a T4, so a
bug in it costs a Kaggle session. This stubs out torch and transformers and
exercises the parts that can actually be wrong: HTTP routing, chat-template
handling, response shape, usage accounting, and — the real risk — whether
concurrent requests are genuinely batched into one generate() call.

Run:  python experiments/e0_validate_serve_hf.py
"""

from __future__ import annotations

import json
import sys
import threading
import time
import types
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kaggle"))


def install_stubs() -> dict:
    """Minimal fakes for torch / transformers, plus a record of generate() calls."""
    record = {
        "batch_sizes": [],
        "padding_side": None,
        "left_padded": False,
        "concurrent_use": False,
    }

    torch = types.ModuleType("torch")
    torch.float16 = "float16"
    torch.bfloat16 = "bfloat16"
    torch.float32 = "float32"

    class _NoGrad:
        def __call__(self, fn):
            return fn

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    torch.inference_mode = _NoGrad
    sys.modules["torch"] = torch

    class FakeEncoding(dict):
        def to(self, _device):
            return self

    class FakeTokenizer:
        """Mimics a Rust-backed fast tokenizer, including its thread hostility.

        Real HuggingFace fast tokenizers raise `RuntimeError: Already borrowed`
        when used from two threads at once. Reproducing that here is the point:
        an earlier version of the server tokenized in the HTTP handler while the
        batch worker was tokenizing too, and only blew up under real load.
        """

        def __init__(self):
            self.padding_side = "right"
            self.pad_token = None
            self.eos_token = "<eos>"
            self.pad_token_id = 0
            self._busy = threading.Lock()

        def _borrow(self):
            if not self._busy.acquire(blocking=False):
                record["concurrent_use"] = True
                raise RuntimeError("Already borrowed")
            return self._busy

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            lock = self._borrow()
            try:
                time.sleep(0.002)   # widen the window so races actually surface
                return "|".join(f"{m['role']}:{m['content']}" for m in messages) + "|assistant:"
            finally:
                lock.release()

        def __call__(self, text, return_tensors=None, padding=False):
            lock = self._borrow()
            try:
                texts = [text] if isinstance(text, str) else list(text)
                if return_tensors is None:
                    return {"input_ids": [1] * max(1, len(texts[0]) // 4)}
                record["batch_sizes"].append(len(texts))
                record["padding_side"] = self.padding_side
                record["left_padded"] = self.padding_side == "left"
                time.sleep(0.005)
                width = max(len(t) for t in texts)
                return FakeEncoding(input_ids=_FakeTensor([[0] * width for _ in texts]))
            finally:
                lock.release()

        def decode(self, seq, skip_special_tokens=True):
            lock = self._borrow()
            try:
                return f"FINAL ANSWER: {int(seq[0])}"
            finally:
                lock.release()

    class _FakeTensor(list):
        @property
        def shape(self):
            return (len(self), len(self[0]) if self else 0)

    class FakeModel:
        device = "cpu"
        # How many real tokens each row "generates" before padding. Rows differ
        # so the trailing-pad trim is actually exercised rather than assumed.
        REAL = [3, 5]

        def eval(self):
            return self

        def generate(self, input_ids=None, max_new_tokens=None, pad_token_id=0, **kw):
            width = input_ids.shape[1]
            rows = []
            for i in range(len(input_ids)):
                n_real = self.REAL[i % len(self.REAL)]
                prompt = [pad_token_id] * width
                # Real token ids deliberately != pad_token_id.
                gen = [101 + i] * n_real
                pad = [pad_token_id] * (max_new_tokens - n_real)
                rows.append(prompt + gen + pad)
            return rows

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *_a, **_k: FakeTokenizer()
    )
    transformers.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda *_a, **_k: FakeModel()
    )
    sys.modules["transformers"] = transformers
    return record


def post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def main() -> int:
    record = install_stubs()
    import serve_hf  # noqa: E402  (import after stubs are installed)

    from http.server import ThreadingHTTPServer

    serve_hf.Handler.engine = serve_hf.BatchEngine("fake/model", "float16", max_batch=16)
    serve_hf.Handler.model_id = "fake/model"
    server = ThreadingHTTPServer(("127.0.0.1", 0), serve_hf.Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    failures = 0

    def check(label, got, want) -> None:
        nonlocal failures
        ok = got == want
        failures += 0 if ok else 1
        print(f"  [{'ok ' if ok else 'FAIL'}] {label}: {got!r} (want {want!r})")

    print("== left padding ==")
    # Set by the tokenizer stub at construction; decoder-only batch generation
    # produces garbage with right padding, so this is load-bearing.
    check("padding_side is left", serve_hf.Handler.engine.tokenizer.padding_side, "left")
    check("pad_token backfilled from eos", serve_hf.Handler.engine.tokenizer.pad_token, "<eos>")

    print("== health ==")
    with urllib.request.urlopen(f"{base}/health", timeout=10) as r:
        check("health status", r.status, 200)

    print("== single completion ==")
    body = post(
        f"{base}/v1/chat/completions",
        {
            "model": "fake/model",
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "what is it"},
            ],
            "max_tokens": 32,
            "temperature": 0.0,
        },
    )
    check("has one choice", len(body["choices"]), 1)
    check("role", body["choices"][0]["message"]["role"], "assistant")
    check(
        "content parses as an answer",
        body["choices"][0]["message"]["content"].startswith("FINAL ANSWER:"),
        True,
    )
    check("usage keys present",
          sorted(body["usage"]) == ["completion_tokens", "prompt_tokens", "total_tokens"], True)
    check("prompt_tokens non-zero", body["usage"]["prompt_tokens"] > 0, True)
    # The fake model emits 3 real tokens then pads; anything else means the
    # trailing-pad trim is wrong and every efficiency number would be inflated.
    check("completion_tokens excludes padding", body["usage"]["completion_tokens"], 3)
    check(
        "total = prompt + completion",
        body["usage"]["total_tokens"],
        body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"],
    )

    print("== concurrent requests are batched ==")
    record["batch_sizes"].clear()
    results: list[dict] = []
    lock = threading.Lock()

    def fire(i: int) -> None:
        out = post(
            f"{base}/v1/chat/completions",
            {
                "model": "fake/model",
                "messages": [{"role": "user", "content": f"task {i}"}],
                "max_tokens": 16,
            },
        )
        with lock:
            results.append(out)

    threads = [threading.Thread(target=fire, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    check("all 12 responded", len(results), 12)
    check("all well-formed", all(r["choices"][0]["message"]["content"] for r in results), True)
    # The regression this file exists for: tokenizing from handler threads while
    # the batch worker tokenizes raises "Already borrowed" under real load.
    check("tokenizer never used concurrently", record["concurrent_use"], False)
    check(
        "completion_tokens reported",
        all(r["usage"]["completion_tokens"] > 0 for r in results),
        True,
    )
    largest = max(record["batch_sizes"]) if record["batch_sizes"] else 0
    batched = largest > 1
    print(f"  [{'ok ' if batched else 'FAIL'}] batched 12 concurrent requests: "
          f"largest batch={largest}, generate() calls={len(record['batch_sizes'])} (want >1)")
    failures += 0 if batched else 1

    print("== error handling ==")
    try:
        post(f"{base}/v1/chat/completions", {"nope": True})
        check("malformed request rejected", False, True)
    except urllib.error.HTTPError as exc:
        check("malformed request rejected", exc.code, 400)

    server.shutdown()
    print()
    if failures:
        print(f"{failures} check(s) FAILED — do not rely on serve_hf.py yet.")
        return 1
    print("serve_hf.py validated (routing, batching, left-padding, usage accounting).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
