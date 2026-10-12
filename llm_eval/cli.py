"""CLI:  python -m llm_eval.cli run --provider groq --prompts v1 v2"""
from __future__ import annotations

import argparse
import sys

from .config import PROVIDERS
from .evaluator import comparison_rows, run_comparison, save_results
from .llm import LLMError
from .prompts import list_prompts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="llm_eval", description="LLM evaluation suite")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Run the evaluation and compare prompt versions")
    r.add_argument("--provider", default="mock", choices=list(PROVIDERS))
    r.add_argument("--model", default=None)
    r.add_argument("--prompts", nargs="+", default=["v1", "v2"])
    r.add_argument("--top-k", type=int, default=3)
    r.add_argument("--limit", type=int, default=None, help="Only the first N questions")
    r.add_argument("--judge", action="store_true", help="Use the LLM as a judge (extra calls)")
    r.add_argument("--fail-under-accuracy", type=float, default=None, help="CI gate, e.g. 0.8")
    r.add_argument("--fail-under-groundedness", type=float, default=None)
    sub.add_parser("list-prompts", help="List available prompt versions")
    args = p.parse_args(argv)

    if args.cmd == "list-prompts":
        print("\n".join(list_prompts()))
        return 0

    def progress(name, i, n):
        print(f"\r[{name}] {i}/{n}", end="", file=sys.stderr, flush=True)
        if i == n:
            print(file=sys.stderr)

    try:
        result = run_comparison(args.prompts, args.provider, args.model, args.top_k,
                                args.limit, args.judge, progress=progress)
    except (LLMError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    path = save_results(result)
    names = list(result["runs"])
    m = result["meta"]
    print(f"\nProvider: {m['provider']} | Model: {m['model']} | {m['n_questions']} questions\n")
    w = max(len(r["metric"]) for r in comparison_rows(result)) + 2
    print("Metric".ljust(w) + "".join(n.ljust(12) for n in names) + "Delta")
    print("-" * (w + 12 * len(names) + 10))
    for row in comparison_rows(result):
        print(row["metric"].ljust(w) + "".join(row[n].ljust(12) for n in names) + row.get("delta", ""))
    print(f"\nSaved: {path}")

    failed = False
    for name in names:
        s = result["runs"][name]["summary"]
        if args.fail_under_accuracy is not None and s["accuracy"] < args.fail_under_accuracy:
            print(f"FAIL: {name} accuracy {s['accuracy']:.0%} < {args.fail_under_accuracy:.0%}", file=sys.stderr)
            failed = True
        if args.fail_under_groundedness is not None and s["groundedness"] < args.fail_under_groundedness:
            print(f"FAIL: {name} groundedness {s['groundedness']:.0%} < {args.fail_under_groundedness:.0%}", file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
