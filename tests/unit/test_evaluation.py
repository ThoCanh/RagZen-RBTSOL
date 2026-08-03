from __future__ import annotations

import pytest

from ragzen.evaluation import (
    citation_precision,
    evaluate_retrieval,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_retrieval_metrics_binary_and_graded() -> None:
    retrieved = ["noise", "a", "b"]
    relevant = {"a", "b", "missing"}
    report = evaluate_retrieval(
        retrieved,
        relevant,
        k=3,
        graded_relevance={"a": 3.0, "b": 1.0, "missing": 2.0},
    )
    assert report.recall == pytest.approx(2 / 3)
    assert report.reciprocal_rank == 0.5
    assert 0.0 < report.ndcg < 1.0
    assert report.k == 3
    assert citation_precision(["a", "bad", "b"], {"a", "b"}) == pytest.approx(2 / 3)


def test_retrieval_metrics_empty_and_invalid_inputs() -> None:
    assert recall_at_k([], set(), k=1) == 0.0
    assert reciprocal_rank(["x"], {"y"}, k=1) == 0.0
    assert ndcg_at_k([], set(), k=1) == 0.0
    assert ndcg_at_k(["x"], {"x": -1.0}, k=1) == 0.0
    assert citation_precision([], {"x"}) == 0.0
    for metric in (recall_at_k, reciprocal_rank, ndcg_at_k):
        with pytest.raises(ValueError, match="greater than zero"):
            metric([], set(), k=0)
