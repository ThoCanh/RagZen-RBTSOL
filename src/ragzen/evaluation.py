"""Dependency-free retrieval metrics for RagZen evaluation suites."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be greater than zero")


def recall_at_k(retrieved: Sequence[str], relevant: Set[str], *, k: int) -> float:
    """Return the fraction of relevant identifiers found in the first *k* results."""
    _validate_k(k)
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]).intersection(relevant)) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: Set[str], *, k: int) -> float:
    """Return reciprocal rank of the first relevant result, or zero when absent."""
    _validate_k(k)
    for rank, identifier in enumerate(retrieved[:k], start=1):
        if identifier in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved: Sequence[str], relevance: Mapping[str, float] | Set[str], *, k: int
) -> float:
    """Compute normalized discounted cumulative gain with binary or graded relevance."""
    _validate_k(k)
    grades = dict.fromkeys(relevance, 1.0) if isinstance(relevance, Set) else dict(relevance)
    if not grades:
        return 0.0

    def discounted_gain(values: Sequence[float]) -> float:
        return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(values, 1))

    actual = [max(0.0, grades.get(identifier, 0.0)) for identifier in retrieved[:k]]
    ideal = sorted((max(0.0, grade) for grade in grades.values()), reverse=True)[:k]
    ideal_gain = discounted_gain(ideal)
    return discounted_gain(actual) / ideal_gain if ideal_gain else 0.0


def citation_precision(citations: Sequence[str], source_ids: Set[str]) -> float:
    """Return the fraction of citations that point to a supplied source identifier."""
    if not citations:
        return 0.0
    return sum(identifier in source_ids for identifier in citations) / len(citations)


@dataclass(frozen=True, slots=True)
class RetrievalEvaluation:
    """A compact metric report for one retrieval query."""

    recall: float
    reciprocal_rank: float
    ndcg: float
    k: int


def evaluate_retrieval(
    retrieved: Sequence[str],
    relevant: Set[str],
    *,
    k: int = 10,
    graded_relevance: Mapping[str, float] | None = None,
) -> RetrievalEvaluation:
    """Evaluate one ranked list using Recall@K, MRR and nDCG@K."""
    relevance: Mapping[str, float] | Set[str] = (
        graded_relevance if graded_relevance is not None else relevant
    )
    return RetrievalEvaluation(
        recall=recall_at_k(retrieved, relevant, k=k),
        reciprocal_rank=reciprocal_rank(retrieved, relevant, k=k),
        ndcg=ndcg_at_k(retrieved, relevance, k=k),
        k=k,
    )
