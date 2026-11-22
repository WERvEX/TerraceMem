"""Tests for offline evaluation metrics, ablation runner, memory benchmark & eval reports API."""

from __future__ import annotations

from pathlib import Path

import pytest
from evals.memory_benchmark import run_memory_benchmark
from evals.metrics import (
    abstention_f1,
    citation_validity,
    false_invalidation_rate,
    mrr_at_k,
    ndcg_at_k,
    percentile_ms,
    prompt_token_reduction_pct,
    recall_at_k,
)
from evals.rag_ablation import run_rag_ablation
from fastapi.testclient import TestClient

from terracemem.api.app import create_app
from terracemem.config import Settings


def test_eval_metric_calculators() -> None:
    assert recall_at_k(["a", "a", "b", "c"], ["b", "d"], k=3) == pytest.approx(0.5)
    assert mrr_at_k(["x", "b", "b"], ["b"], k=5) == pytest.approx(0.5)
    assert ndcg_at_k(["a", "a", "b"], ["a"], k=5) == pytest.approx(1.0)
    assert citation_validity(["c1", "c2"], ["c1", "c2", "c3"]) == pytest.approx(1.0)
    assert citation_validity(["c1", "fake"], ["c1"]) == pytest.approx(0.5)
    assert citation_validity([], ["c1"]) == 0.0

    f1, far = abstention_f1([(True, True), (True, False), (False, False), (False, False)])
    assert f1 == pytest.approx(2 / 3, rel=1e-4)
    assert far == pytest.approx(0.5)
    assert percentile_ms([10.0, 20.0, 30.0, 40.0], 50) == pytest.approx(25.0)
    assert prompt_token_reduction_pct(1000, 350) == pytest.approx(65.0)
    assert false_invalidation_rate(4, 1) == pytest.approx(0.25)


async def test_dry_run_rag_and_memory_evals_and_reports_endpoint(tmp_path: Path) -> None:
    dataset = Path("evals/datasets/tech_docs_hardset.json")
    rag_report = await run_rag_ablation(
        dataset_path=dataset,
        config_names=["B2", "B5"],
        output_dir=tmp_path,
        dry_run=True,
    )
    assert rag_report["benchmark"] == "rag_ablation"
    assert len(rag_report["results"]) == 2
    assert (tmp_path / "rag_ablation_latest.json").exists()
    assert (tmp_path / "rag_ablation_latest.md").exists()

    mem_report = await run_memory_benchmark(
        longmemeval_path=None,
        output_dir=tmp_path,
        dry_run=True,
    )
    assert mem_report["benchmark"] == "memory_benchmark"
    assert len(mem_report["results"]) == 4
    assert (tmp_path / "memory_benchmark_latest.json").exists()

    app = create_app(Settings(storage_backend="memory", eval_reports_dir=str(tmp_path)))
    with TestClient(app) as client:
        resp = client.get("/api/v1/eval-reports")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2
