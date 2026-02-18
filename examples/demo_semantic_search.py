"""
Semantic Search Demo - Local BM25 vs Semantic Search

Demonstrates how semantic search understands natural language intent
while local keyword search fails on synonyms and colloquial queries.

Run with local Lambda:
    cd ai-generation/apps/action_search && make run-local
    uv run python examples/demo_semantic_search.py --local

Run with production API:
    STACKONE_API_KEY=xxx uv run python examples/demo_semantic_search.py
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from stackone_ai.semantic_search import (
    SemanticSearchClient,
    SemanticSearchResponse,
    SemanticSearchResult,
)
from stackone_ai.utility_tools import ToolIndex

# Local Lambda URL
DEFAULT_LAMBDA_URL = "http://localhost:4513/2015-03-31/functions/function/invocations"

# Demo queries - the strongest "wow" moments from benchmark results
DEMO_QUERIES = [
    {
        "query": "fire someone",
        "why": "Synonym: 'fire' = terminate employment",
    },
    {
        "query": "ping the team",
        "why": "Intent: 'ping' = send a message",
    },
    {
        "query": "file a new bug",
        "why": "Intent: 'file a bug' = create issue (not file operations)",
    },
    {
        "query": "check my to-do list",
        "why": "Concept: 'to-do list' = list tasks",
    },
    {
        "query": "show me everyone in the company",
        "why": "Synonym: 'everyone in company' = list employees",
    },
    {
        "query": "turn down a job seeker",
        "why": "Synonym: 'turn down' = reject application",
    },
    {
        "query": "approve PTO",
        "why": "Abbreviation: 'PTO' = paid time off request",
    },
    {
        "query": "grab that spreadsheet",
        "why": "Colloquial: 'grab' = download file",
    },
]


@dataclass
class LightweightTool:
    """Minimal tool for BM25 indexing."""

    name: str
    description: str


class LocalLambdaClient:
    """Client for local action_search Lambda."""

    def __init__(self, url: str = DEFAULT_LAMBDA_URL) -> None:
        self.url = url

    def search(
        self,
        query: str,
        connector: str | None = None,
        top_k: int = 5,
    ) -> SemanticSearchResponse:
        payload: dict[str, Any] = {
            "type": "search",
            "payload": {"query": query, "top_k": top_k},
        }
        if connector:
            payload["payload"]["connector"] = connector

        resp = httpx.post(self.url, json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()

        results = [
            SemanticSearchResult(
                action_name=r.get("action_name", ""),
                connector_key=r.get("connector_key", ""),
                similarity_score=r.get("similarity_score", 0.0),
                label=r.get("label", ""),
                description=r.get("description", ""),
            )
            for r in data.get("results", [])
        ]
        return SemanticSearchResponse(
            results=results,
            total_count=data.get("total_count", len(results)),
            query=data.get("query", query),
        )

    def fetch_actions(self) -> list[LightweightTool]:
        """Fetch broad action catalog for BM25 index."""
        seen: dict[str, LightweightTool] = {}
        for q in ["employee", "candidate", "contact", "task", "message", "file", "event", "deal"]:
            try:
                resp = httpx.post(
                    self.url,
                    json={"type": "search", "payload": {"query": q, "top_k": 500}},
                    timeout=30.0,
                )
                for r in resp.json().get("results", []):
                    name = r.get("action_name", "")
                    if name and name not in seen:
                        seen[name] = LightweightTool(name=name, description=r.get("description", ""))
            except Exception:
                continue
        return list(seen.values())


def shorten_name(name: str) -> str:
    """Shorten action name for display.

    bamboohr_1.0.0_bamboohr_list_employees_global -> bamboohr: list_employees
    """
    parts = name.split("_")
    # Find version segment (e.g., "1.0.0") and split around it
    version_idx = None
    for i, p in enumerate(parts):
        if "." in p and any(c.isdigit() for c in p):
            version_idx = i
            break

    if version_idx is not None:
        connector = parts[0]
        # Skip connector + version + repeated connector prefix
        action_parts = parts[version_idx + 1 :]
        # Remove leading connector name if repeated
        if action_parts and action_parts[0].lower().replace("-", "") == connector.lower().replace("-", ""):
            action_parts = action_parts[1:]
        # Remove trailing 'global'
        if action_parts and action_parts[-1] == "global":
            action_parts = action_parts[:-1]
        action = "_".join(action_parts)
        return f"{connector}: {action}"

    return name


def print_header(text: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}")


def print_section(text: str) -> None:
    print(f"\n--- {text} ---\n")


def run_demo(use_local: bool, lambda_url: str, api_key: str | None) -> None:
    # Step 1: Setup
    if use_local:
        client = LocalLambdaClient(url=lambda_url)
        semantic_search = client.search
    else:
        if not api_key:
            print("Error: STACKONE_API_KEY required for production mode")
            print("Use --local flag for local Lambda mode")
            exit(1)
        sem_client = SemanticSearchClient(api_key=api_key)
        semantic_search = sem_client.search
        client = None

    print_header("SEMANTIC SEARCH DEMO")
    print("\n  Comparing Local BM25+TF-IDF vs Semantic Search")
    print("  across 5,144 actions from 200+ connectors\n")

    # Step 2: Build local BM25 index
    print("  Loading action catalog for local BM25 index...")
    if use_local:
        tools = client.fetch_actions()
    else:
        # For production mode, use semantic search to build catalog
        local_client = LocalLambdaClient(url=lambda_url)
        tools = local_client.fetch_actions()

    local_index = ToolIndex(tools)  # type: ignore[arg-type]
    print(f"  Indexed {len(tools)} actions\n")

    input("  Press Enter to start the demo...\n")

    # Step 3: Side-by-side comparison
    print_header("SIDE-BY-SIDE COMPARISON")

    local_hits = 0
    semantic_hits = 0

    for i, demo in enumerate(DEMO_QUERIES, 1):
        query = demo["query"]
        why = demo["why"]

        print(f"\n  [{i}/{len(DEMO_QUERIES)}] Query: \"{query}\"")
        print(f"  Why interesting: {why}")
        print()

        # Local search
        start = time.perf_counter()
        local_results = local_index.search(query, limit=3)
        local_ms = (time.perf_counter() - start) * 1000
        local_names = [shorten_name(r.name) for r in local_results]

        # Semantic search
        start = time.perf_counter()
        sem_response = semantic_search(query=query, top_k=3)
        sem_ms = (time.perf_counter() - start) * 1000
        sem_names = [shorten_name(r.action_name) for r in sem_response.results]
        sem_scores = [f"{r.similarity_score:.2f}" for r in sem_response.results]

        # Display
        w = 38
        print(f"  {'Local BM25 (keyword)':<{w}} | {'Semantic Search (AI)':<{w}}")
        print(f"  {f'{local_ms:.1f}ms':<{w}} | {f'{sem_ms:.1f}ms':<{w}}")
        print(f"  {'-' * w} | {'-' * w}")
        for j in range(min(3, max(len(local_names), len(sem_names)))):
            l_name = local_names[j] if j < len(local_names) else ""
            s_name = sem_names[j] if j < len(sem_names) else ""
            s_score = sem_scores[j] if j < len(sem_scores) else ""
            l_display = f"  {l_name[:w]:<{w}}"
            s_display = f"  {s_name[:w - 8]:<{w - 8}} ({s_score})" if s_name else ""
            print(f"{l_display} |{s_display}")

        input("\n  Press Enter for next query...")

    # Step 4: Summary
    print_header("BENCHMARK RESULTS (94 evaluation tasks)")

    print("""
  Method                    Hit@5        MRR       Avg Latency
  ----------------------------------------------------------
  Local BM25+TF-IDF         66.0%      0.538          1.2ms
  Semantic Search            76.6%      0.634        279.6ms
  ----------------------------------------------------------
  Improvement              +10.6%     +0.096
    """)

    # Step 5: Code examples
    print_header("DEVELOPER API")

    print("""
  # 1. Direct semantic search
  from stackone_ai import StackOneToolSet

  toolset = StackOneToolSet(api_key="xxx")
  tools = toolset.search_tools("fire someone", top_k=5)
  # Returns: terminate_employee, offboard_employee, ...


  # 2. Semantic search with connector filter
  tools = toolset.search_tools(
      "send a message",
      connector="slack",
      top_k=3,
  )
  # Returns: slack_send_message, slack_create_conversation, ...


  # 3. MCP utility tool (for AI agents)
  tools = toolset.fetch_tools()
  utility = tools.utility_tools(use_semantic_search=True)
  # AI agent gets: tool_search (semantic-powered) + tool_execute


  # 4. Inspect results before fetching
  results = toolset.search_action_names("onboard new hire")
  for r in results:
      print(f"{r.action_name}: {r.similarity_score:.2f}")
    """)

    print_header("END OF DEMO")


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search Demo")
    parser.add_argument("--local", action="store_true", help="Use local Lambda")
    parser.add_argument("--lambda-url", default=DEFAULT_LAMBDA_URL, help="Lambda URL")
    args = parser.parse_args()

    api_key = os.environ.get("STACKONE_API_KEY")
    run_demo(use_local=args.local, lambda_url=args.lambda_url, api_key=api_key)


if __name__ == "__main__":
    main()
