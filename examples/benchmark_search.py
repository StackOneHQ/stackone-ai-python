"""Benchmark: measure SDK search latency with caching.

Runs fetch_tools, local (BM25+TF-IDF) search, and semantic search N times,
reports cold vs warm average latency and the speedup from caching.

Prerequisites:
    - STACKONE_API_KEY environment variable
    - STACKONE_ACCOUNT_ID environment variable

Run with:
    uv run python examples/benchmark_search.py              # default 100 iterations
    uv run python examples/benchmark_search.py -n 50        # fewer for a quick check
"""
from __future__ import annotations

import argparse
import os
import sys
import time

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

from stackone_ai import StackOneToolSet

QUERIES = [
    "list events",
    "cancel a meeting",
    "send a message",
    "get current user",
    "list employees",
]


def bench(fn, n: int) -> tuple[float, float, list[float]]:
    """Run fn() n times. Return (cold, warm_avg, all_times)."""
    times: list[float] = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)

    cold = times[0]
    warm_times = times[1:]
    warm_avg = sum(warm_times) / len(warm_times) if warm_times else cold
    return cold, warm_avg, times


def fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:8.1f}ms"


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark SDK search latency")
    parser.add_argument("--iterations", "-n", type=int, default=100, help="iterations per benchmark (default 100)")
    args = parser.parse_args()
    n = args.iterations

    api_key = os.getenv("STACKONE_API_KEY")
    account_id = os.getenv("STACKONE_ACCOUNT_ID")

    if not api_key:
        print("Set STACKONE_API_KEY to run this benchmark.")
        return 1
    if not account_id:
        print("Set STACKONE_ACCOUNT_ID to run this benchmark.")
        return 1

    print(f"Benchmarking with account {account_id[:8]}..., {n} iterations each\n")

    ts = StackOneToolSet(
        api_key=api_key,
        account_id=account_id,
        search={"method": "auto", "top_k": 5},
    )

    results: list[tuple[str, float, float, float]] = []
    query_idx = 0

    def next_query() -> str:
        nonlocal query_idx
        q = QUERIES[query_idx % len(QUERIES)]
        query_idx += 1
        return q

    # --- 1. fetch_tools ---
    print(f"[1/3] fetch_tools x{n} ...")
    ts.clear_catalog_cache()
    cold, warm_avg, _ = bench(lambda: ts.fetch_tools(), n)
    speedup = cold / warm_avg if warm_avg > 0 else float("inf")
    results.append(("fetch_tools", cold, warm_avg, speedup))
    print(f"       cold={fmt_ms(cold)}  warm_avg={fmt_ms(warm_avg)}  speedup={speedup:.0f}x")

    # --- 2. local search (BM25 + TF-IDF) ---
    print(f"[2/3] search_tools (local) x{n} ...")
    ts.clear_catalog_cache()
    query_idx = 0
    cold, warm_avg, _ = bench(lambda: ts.search_tools(next_query(), search="local"), n)
    speedup = cold / warm_avg if warm_avg > 0 else float("inf")
    results.append(("search (local/BM25)", cold, warm_avg, speedup))
    print(f"       cold={fmt_ms(cold)}  warm_avg={fmt_ms(warm_avg)}  speedup={speedup:.0f}x")

    # --- 3. semantic search (auto) ---
    print(f"[3/3] search_tools (semantic/auto) x{n} ...")
    ts.clear_catalog_cache()
    query_idx = 0
    cold, warm_avg, _ = bench(lambda: ts.search_tools(next_query(), search="auto"), n)
    speedup = cold / warm_avg if warm_avg > 0 else float("inf")
    results.append(("search (semantic)", cold, warm_avg, speedup))
    print(f"       cold={fmt_ms(cold)}  warm_avg={fmt_ms(warm_avg)}  speedup={speedup:.0f}x")

    # --- Summary ---
    print("\n" + "=" * 65)
    print(f"{'Benchmark':<22} {'Cold':>10} {'Warm (avg)':>10} {'Speedup':>10}")
    print("-" * 65)
    for name, c, w, s in results:
        print(f"{name:<22} {fmt_ms(c):>10} {fmt_ms(w):>10} {s:>9.0f}x")
    print("=" * 65)

    print(f"\nWarm = average of {n - 1} calls after the first (cold) call.")
    print("Speedup = cold / warm_avg — shows the benefit of caching.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
