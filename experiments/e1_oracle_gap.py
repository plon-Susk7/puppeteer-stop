"""E1 — generate the trajectory corpus and measure the oracle stopping gap.

Generation is the only expensive step in the project; analysis is free replay,
so the two are separate subcommands and analysis can be re-run without cost.

    # one-time corpus (resumable — safe to re-run after a killed session)
    python experiments/e1_oracle_gap.py generate --dataset gsm-hard --limit 300

    # free, repeatable
    python experiments/e1_oracle_gap.py analyze --dataset gsm-hard

Decision gate (from the plan): gap >= 3 pts OR abandonment >= 5% -> proceed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.oracle_gap import report  # noqa: E402
from puppeteer_stop.config import (  # noqa: E402
    DEPTH,
    POOL_NAMES,
    TRACE_DIR,
    RunConfig,
    resolve_pool,
)
from puppeteer_stop.rollout import POLICIES, generate_corpus  # noqa: E402
from puppeteer_stop.tasks import load_gsm_hard, load_mmlu_pro  # noqa: E402
from puppeteer_stop.trace import TraceWriter, load_traces  # noqa: E402

LOADERS = {"gsm-hard": load_gsm_hard, "mmlu-pro": load_mmlu_pro}


def trace_path(dataset: str, policy: str, pool: str) -> Path:
    return Path(TRACE_DIR) / f"{dataset}__{policy}__{pool}.jsonl"


def cmd_generate(args: argparse.Namespace) -> int:
    tasks = LOADERS[args.dataset](limit=args.limit)
    if not tasks:
        print(f"No tasks loaded for {args.dataset}.")
        return 1

    spec = resolve_pool(args.pool)
    client = RunConfig(pool=spec, depth=args.depth, seed=args.seed).client()
    path = trace_path(args.dataset, args.policy, args.pool)
    writer = TraceWriter(path)

    already = len(writer)
    print(f"corpus: {path}")
    print(f"  dataset={args.dataset}  tasks={len(tasks)}  depth={args.depth}")
    print(f"  policy={args.policy}  pool={args.pool} ({spec.api}:{spec.model})")
    print(f"  seed={args.seed}  concurrency={args.concurrency}")
    print(f"  already complete: {already}")
    print(f"  ~{(len(tasks) - already) * args.depth} model calls remaining\n")

    written = generate_corpus(
        tasks,
        client,
        writer,
        policy=args.policy,
        pool=args.pool,
        depth=args.depth,
        seed=args.seed,
        concurrency=args.concurrency,
    )
    print(f"\nwrote {written} new episodes; corpus now {len(writer)} episodes")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    path = trace_path(args.dataset, args.policy, args.pool)
    if not path.exists():
        print(f"No corpus at {path}. Run `generate` first.")
        return 1

    traces = load_traces(path)
    if not traces:
        print(f"Corpus at {path} is empty.")
        return 1

    print(f"corpus: {path}  ({len(traces)} episodes)\n")
    reports = {}
    for mode in ("last", "vote"):
        rep = report(traces, mode)  # type: ignore[arg-type]
        reports[mode] = rep
        print(rep.summary())
        print()

    # The two readouts answer different questions; divergence is a finding.
    d_last, d_vote = reports["last"], reports["vote"]
    print("readout comparison")
    print(f"  gap          last={d_last.gap:.4f}   vote={d_vote.gap:.4f}")
    print(f"  abandonment  last={d_last.abandonment_rate:.4f}   "
          f"vote={d_vote.abandonment_rate:.4f}")
    if abs(d_last.abandonment_rate - d_vote.abandonment_rate) >= 0.03:
        higher = "last" if d_last.abandonment_rate > d_vote.abandonment_rate else "vote"
        print(f"  -> `{higher}` loses noticeably more; "
              + ("answers drift and voting rescues them."
                 if higher == "last"
                 else "correct answers are being outvoted."))
    print()
    print(d_last.verdict())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ("generate", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", choices=sorted(LOADERS), default="gsm-hard")
        p.add_argument("--policy", choices=sorted(POLICIES), default="heuristic")
        p.add_argument("--pool", choices=POOL_NAMES, default="local")
        if name == "generate":
            p.add_argument("--limit", type=int, default=300)
            p.add_argument("--depth", type=int, default=DEPTH)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument(
                "--concurrency",
                type=int,
                default=16,
                help="episodes in flight at once; 16-32 for a local vLLM server "
                "(throughput comes from batching), 2-4 for a rate-limited API",
            )

    args = parser.parse_args()
    return cmd_generate(args) if args.cmd == "generate" else cmd_analyze(args)


if __name__ == "__main__":
    sys.exit(main())
