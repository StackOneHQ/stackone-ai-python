"""Local BM25 + TF-IDF hybrid keyword search for tool discovery."""

from __future__ import annotations

import bm25s
import numpy as np
from pydantic import BaseModel

from stackone_ai.constants import DEFAULT_HYBRID_ALPHA
from stackone_ai.models import StackOneTool
from stackone_ai.utils.tfidf_index import TfidfDocument, TfidfIndex


class ToolSearchResult(BaseModel):
    """Result from tool_search"""

    name: str
    description: str
    score: float


class ToolIndex:
    """Hybrid BM25 + TF-IDF tool search index"""

    def __init__(self, tools: list[StackOneTool], hybrid_alpha: float | None = None) -> None:
        """Initialize tool index with hybrid search

        Args:
            tools: List of tools to index
            hybrid_alpha: Weight for BM25 in hybrid search (0-1). If not provided,
                uses DEFAULT_HYBRID_ALPHA (0.2), which gives more weight to BM25 scoring
                and has been shown to provide better tool discovery accuracy
                (10.8% improvement in validation testing).
        """
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
        # Use default if not provided, then clamp to [0, 1]
        alpha = hybrid_alpha if hybrid_alpha is not None else DEFAULT_HYBRID_ALPHA
        self.hybrid_alpha = max(0.0, min(1.0, alpha))

        # Prepare corpus for both BM25 and TF-IDF
        corpus = []
        tfidf_docs = []
        self.tool_names = []

        for tool in tools:
            # Extract category and action from tool name
            parts = tool.name.split("_")
            category = parts[0] if parts else ""

            # Extract action types
            action_types = ["create", "update", "delete", "get", "list", "search"]
            actions = [p for p in parts if p in action_types]

            # Combine name, description, category and tags for indexing
            # For TF-IDF: use weighted approach similar to Node.js
            tfidf_text = " ".join(
                [
                    f"{tool.name} {tool.name} {tool.name}",  # boost name
                    f"{category} {' '.join(actions)}",
                    tool.description,
                    " ".join(parts),
                ]
            )

            # For BM25: simpler approach
            bm25_text = " ".join(
                [
                    tool.name,
                    tool.description,
                    category,
                    " ".join(parts),
                    " ".join(actions),
                ]
            )

            corpus.append(bm25_text)
            tfidf_docs.append(TfidfDocument(id=tool.name, text=tfidf_text))
            self.tool_names.append(tool.name)

        # Create BM25 index
        self.bm25_retriever = bm25s.BM25()
        if corpus:
            corpus_tokens = bm25s.tokenize(corpus, stemmer=None, show_progress=False)  # ty: ignore[invalid-argument-type]
            self.bm25_retriever.index(corpus_tokens)

        # Create TF-IDF index
        self.tfidf_index = TfidfIndex()
        if tfidf_docs:
            self.tfidf_index.build(tfidf_docs)

    def search(self, query: str, limit: int = 5, min_score: float = 0.0) -> list[ToolSearchResult]:
        """Search for relevant tools using hybrid BM25 + TF-IDF

        Args:
            query: Natural language query
            limit: Maximum number of results
            min_score: Minimum relevance score (0-1)

        Returns:
            List of search results sorted by relevance
        """
        if not self.tools:
            return []

        # Get more results initially to have better candidate pool for fusion
        fetch_limit = max(50, limit)

        # Tokenize query for BM25
        query_tokens = bm25s.tokenize([query], stemmer=None, show_progress=False)  # ty: ignore[invalid-argument-type]

        # Search with BM25
        bm25_results, bm25_scores = self.bm25_retriever.retrieve(
            query_tokens, k=min(fetch_limit, len(self.tools))
        )

        # Search with TF-IDF
        tfidf_results = self.tfidf_index.search(query, k=min(fetch_limit, len(self.tools)))

        # Build score map for fusion
        score_map: dict[str, dict[str, float]] = {}

        # Add BM25 scores
        for idx, score in zip(bm25_results[0], bm25_scores[0], strict=True):
            tool_name = self.tool_names[idx]
            # Normalize BM25 score to 0-1 range
            normalized_score = float(1 / (1 + np.exp(-score / 10)))
            # Clamp to [0, 1]
            clamped_score = max(0.0, min(1.0, normalized_score))
            score_map[tool_name] = {"bm25": clamped_score}

        # Add TF-IDF scores
        for result in tfidf_results:
            if result.id not in score_map:
                score_map[result.id] = {}
            score_map[result.id]["tfidf"] = result.score

        # Fuse scores: hybrid_score = alpha * bm25 + (1 - alpha) * tfidf
        fused_results: list[tuple[str, float]] = []
        for tool_name, scores in score_map.items():
            bm25_score = scores.get("bm25", 0.0)
            tfidf_score = scores.get("tfidf", 0.0)
            hybrid_score = self.hybrid_alpha * bm25_score + (1 - self.hybrid_alpha) * tfidf_score
            fused_results.append((tool_name, hybrid_score))

        # Sort by score descending
        fused_results.sort(key=lambda x: x[1], reverse=True)

        # Build final results
        search_results = []
        for tool_name, score in fused_results:
            if score < min_score:
                continue

            tool = self.tool_map.get(tool_name)
            if tool is None:
                continue

            search_results.append(
                ToolSearchResult(
                    name=tool.name,
                    description=tool.description,
                    score=score,
                )
            )

            if len(search_results) >= limit:
                break

        return search_results
