"""
Lightweight TF-IDF vector index for offline vector search.
No external dependencies; tokenizes ASCII/latin text, lowercases,
strips punctuation, removes a small stopword set, and builds a sparse index.
"""

from __future__ import annotations

import math
import re
from typing import NamedTuple


class TfidfDocument(NamedTuple):
    """Document for TF-IDF indexing"""

    id: str
    text: str


class TfidfResult(NamedTuple):
    """Search result from TF-IDF index"""

    id: str
    score: float  # cosine similarity (0..1)


# Common English stopwords
STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "for",
    "of",
    "in",
    "on",
    "to",
    "from",
    "by",
    "with",
    "as",
    "at",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "it",
    "this",
    "that",
    "these",
    "those",
    "not",
    "no",
    "can",
    "could",
    "should",
    "would",
    "may",
    "might",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "you",
    "your",
}


def tokenize(text: str) -> list[str]:
    """Tokenize text by lowercasing, removing punctuation, and filtering stopwords

    Args:
        text: Input text to tokenize

    Returns:
        List of tokens
    """
    # Lowercase and replace non-alphanumeric (except underscore) with space
    text = text.lower()
    text = re.sub(r"[^a-z0-9_\s]", " ", text)

    # Split and filter
    tokens = [t for t in text.split() if t and t not in STOPWORDS]
    return tokens


class TfidfIndex:
    """TF-IDF vector index for document search using sparse vectors"""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf: list[float] = []
        self.docs: list[dict[str, str | dict[int, float] | float]] = []

    def build(self, corpus: list[TfidfDocument]) -> None:
        """Build index from a corpus of documents

        Args:
            corpus: List of documents to index
        """
        # Tokenize all documents
        docs_tokens = [tokenize(doc.text) for doc in corpus]

        # Build vocabulary
        for tokens in docs_tokens:
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

        # Compute document frequency (df)
        df: dict[int, int] = {}
        for tokens in docs_tokens:
            seen: set[int] = set()
            for token in tokens:
                term_id = self.vocab.get(token)
                if term_id is not None and term_id not in seen:
                    seen.add(term_id)
                    df[term_id] = df.get(term_id, 0) + 1

        # Compute IDF (inverse document frequency)
        n_docs = len(corpus)
        self.idf = []
        for term_id in range(len(self.vocab)):
            dfi = df.get(term_id, 0)
            # Smoothed IDF
            idf_value = math.log((n_docs + 1) / (dfi + 1)) + 1
            self.idf.append(idf_value)

        # Build document vectors
        self.docs = []
        for doc, tokens in zip(corpus, docs_tokens, strict=True):
            # Compute term frequency (TF)
            tf: dict[int, int] = {}
            for token in tokens:
                term_id = self.vocab.get(token)
                if term_id is not None:
                    tf[term_id] = tf.get(term_id, 0) + 1

            # Build weighted vector (TF-IDF)
            vec: dict[int, float] = {}
            norm_sq = 0.0
            n_tokens = len(tokens)

            for term_id, freq in tf.items():
                if term_id >= len(self.idf) or n_tokens == 0:
                    continue
                idf_val = self.idf[term_id]
                weight = (freq / n_tokens) * idf_val
                if weight > 0:
                    vec[term_id] = weight
                    norm_sq += weight * weight

            norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0

            self.docs.append(
                {
                    "id": doc.id,
                    "vec": vec,
                    "norm": norm,
                }
            )

    def search(self, query: str, k: int = 10) -> list[TfidfResult]:
        """Search for documents similar to the query

        Args:
            query: Search query
            k: Maximum number of results to return

        Returns:
            List of results sorted by score (descending)
        """
        tokens = tokenize(query)
        if not tokens or not self.vocab:
            return []

        # Compute query term frequency
        tf: dict[int, int] = {}
        for token in tokens:
            term_id = self.vocab.get(token)
            if term_id is not None:
                tf[term_id] = tf.get(term_id, 0) + 1

        if not tf:
            return []

        # Build query vector
        q_vec: dict[int, float] = {}
        q_norm_sq = 0.0
        n_tokens = len(tokens)

        for term_id, freq in tf.items():
            if term_id >= len(self.idf):
                continue
            idf_val = self.idf[term_id]
            weight = (freq / n_tokens) * idf_val if n_tokens > 0 else 0
            if weight > 0:
                q_vec[term_id] = weight
                q_norm_sq += weight * weight

        q_norm = math.sqrt(q_norm_sq) if q_norm_sq > 0 else 1.0

        # Compute cosine similarity with each document
        scores: list[TfidfResult] = []
        for doc in self.docs:
            doc_vec = doc["vec"]
            doc_norm = doc["norm"]
            if not isinstance(doc_vec, dict) or not isinstance(doc_norm, (int, float)):
                continue

            # Compute dot product (iterate over smaller map for efficiency)
            dot = 0.0
            small_vec = q_vec if len(q_vec) <= len(doc_vec) else doc_vec
            big_vec = doc_vec if len(q_vec) <= len(doc_vec) else q_vec

            for term_id, weight in small_vec.items():
                other_weight = big_vec.get(term_id)
                if other_weight is not None:
                    dot += weight * other_weight

            # Cosine similarity
            similarity = dot / (q_norm * doc_norm)
            if similarity > 0:
                doc_id = doc["id"]
                if isinstance(doc_id, str):
                    # Clamp to [0, 1]
                    clamped_score = max(0.0, min(1.0, similarity))
                    scores.append(TfidfResult(id=doc_id, score=clamped_score))

        # Sort by score descending and return top k
        scores.sort(key=lambda x: x.score, reverse=True)
        return scores[:k]
