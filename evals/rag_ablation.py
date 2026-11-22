"""Reproducible 7-group Agentic RAG ablation suite (B0..B5 + B_kw_only)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from evals.metrics import (
        abstention_f1,
        citation_validity,
        mrr_at_k,
        ndcg_at_k,
        percentile_ms,
        recall_at_k,
    )
except ModuleNotFoundError:
    from metrics import (
        abstention_f1,
        citation_validity,
        mrr_at_k,
        ndcg_at_k,
        percentile_ms,
        recall_at_k,
    )
from terracemem.config import Settings
from terracemem.db.repositories import RetrievalMode
from terracemem.domain.models import ChunkRecord, ParsedDocument, RagDecisionStatus, RetrievalFilter
from terracemem.rag.fsm import RagRequest
from terracemem.runtime import TerraceMemRuntime
from terracemem.utils.ids import new_uuid7
from terracemem.utils.tokens import estimate_tokens


@dataclass(frozen=True)
class AblationConfig:
    """Configuration defining one row of the 7-group RAG ablation matrix."""

    name: str
    flat_chunking: bool
    retrieval_mode: RetrievalMode
    enable_enrichment: bool
    skip_embeddings: bool
    enable_rerank: bool
    enable_parent_expand: bool
    enable_graph: bool
    enable_outline_nav: bool
    enable_crag: bool
    strict_citation: bool


ABLATION_MATRIX: dict[str, AblationConfig] = {
    "B0": AblationConfig(
        name="B0",
        flat_chunking=True,
        retrieval_mode="dense_only",
        enable_enrichment=False,
        skip_embeddings=False,
        enable_rerank=False,
        enable_parent_expand=False,
        enable_graph=False,
        enable_outline_nav=False,
        enable_crag=False,
        strict_citation=False,
    ),
    "B1": AblationConfig(
        name="B1",
        flat_chunking=False,
        retrieval_mode="dense_only",
        enable_enrichment=False,
        skip_embeddings=False,
        enable_rerank=False,
        enable_parent_expand=False,
        enable_graph=False,
        enable_outline_nav=False,
        enable_crag=False,
        strict_citation=False,
    ),
    "B_kw_only": AblationConfig(
        name="B_kw_only",
        flat_chunking=False,
        retrieval_mode="lexical_only",
        enable_enrichment=True,
        skip_embeddings=True,
        enable_rerank=False,
        enable_parent_expand=False,
        enable_graph=False,
        enable_outline_nav=False,
        enable_crag=False,
        strict_citation=False,
    ),
    "B2": AblationConfig(
        name="B2",
        flat_chunking=False,
        retrieval_mode="hybrid",
        enable_enrichment=False,
        skip_embeddings=False,
        enable_rerank=False,
        enable_parent_expand=False,
        enable_graph=False,
        enable_outline_nav=False,
        enable_crag=False,
        strict_citation=False,
    ),
    "B3": AblationConfig(
        name="B3",
        flat_chunking=False,
        retrieval_mode="hybrid",
        enable_enrichment=False,
        skip_embeddings=False,
        enable_rerank=True,
        enable_parent_expand=True,
        enable_graph=False,
        enable_outline_nav=False,
        enable_crag=False,
        strict_citation=False,
    ),
    "B4": AblationConfig(
        name="B4",
        flat_chunking=False,
        retrieval_mode="hybrid",
        enable_enrichment=True,
        skip_embeddings=False,
        enable_rerank=True,
        enable_parent_expand=True,
        enable_graph=True,
        enable_outline_nav=True,
        enable_crag=False,
        strict_citation=False,
    ),
    "B5": AblationConfig(
        name="B5",
        flat_chunking=False,
        retrieval_mode="hybrid",
        enable_enrichment=True,
        skip_embeddings=False,
        enable_rerank=True,
        enable_parent_expand=True,
        enable_graph=True,
        enable_outline_nav=True,
        enable_crag=True,
        strict_citation=True,
    ),
}

_DEFAULT_DATASET = Path(__file__).resolve().parent / "datasets" / "tech_docs_hardset.json"


def _make_flat_document(
    *,
    external_key: str,
    title: str,
    content: str,
    chunk_chars: int = 240,
) -> ParsedDocument:
    """Naive fixed-window flat chunker used strictly for B0 baseline comparison."""
    doc_id = new_uuid7()
    cleaned = content.replace("#", "").strip()
    chunks: list[ChunkRecord] = []
    for idx, start in enumerate(range(0, len(cleaned), chunk_chars)):
        piece = cleaned[start : start + chunk_chars].strip()
        if not piece:
            continue
        chunks.append(
            ChunkRecord(
                document_id=doc_id,
                section_id=None,
                chunk_index=idx,
                raw_text=piece,
                contextual_text=piece,
                token_count=estimate_tokens(piece),
            )
        )
    return ParsedDocument(
        id=doc_id,
        external_key=external_key,
        title=title,
        content_hash=f"flat:{hash(content)}",
        sections=[],
        chunks=chunks,
    )


async def run_rag_ablation(
    *,
    dataset_path: Path,
    config_names: list[str],
    output_dir: Path,
    dry_run: bool = True,
    resume: bool = False,
) -> dict[str, Any]:
    raw_dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    docs = raw_dataset["documents"]
    queries = raw_dataset["queries"]

    output_dir.mkdir(parents=True, exist_ok=True)
    json_out = output_dir / "rag_ablation_latest.json"
    md_out = output_dir / "rag_ablation_latest.md"

    existing_by_name: dict[str, dict[str, Any]] = {}
    if resume and json_out.exists():
        prev = json.loads(json_out.read_text(encoding="utf-8"))
        for row in prev.get("results", []):
            existing_by_name[str(row["config"])] = row

    settings = Settings(
        storage_backend="memory" if dry_run else "auto",
        llm_api_key="" if dry_run else Settings().llm_api_key,
        embedding_api_key="" if dry_run else Settings().embedding_api_key,
        reranker_enabled=False if dry_run else Settings().reranker_enabled,
    )
    runtime = TerraceMemRuntime(settings)
    await runtime.startup()

    results: list[dict[str, Any]] = []
    try:
        for cfg_name in config_names:
            if resume and cfg_name in existing_by_name:
                results.append(existing_by_name[cfg_name])
                continue

            cfg = ABLATION_MATRIX[cfg_name]
            namespace = f"eval_{cfg.name.lower()}"
            doc_id_to_ext_key: dict[str, str] = {}
            ingest_tokens = 0

            for doc in docs:
                ext_key = str(doc["external_key"])
                title = str(doc["title"])
                content = str(doc["content"])
                ingest_tokens += estimate_tokens(content)

                if cfg.flat_chunking:
                    parsed = _make_flat_document(
                        external_key=ext_key,
                        title=title,
                        content=content,
                    )
                    res = await runtime.ingestion.ingest_parsed(
                        parsed,
                        namespace=namespace,
                        enable_enrichment=False,
                        enable_graph=False,
                        skip_embeddings=cfg.skip_embeddings,
                    )
                else:
                    res = await runtime.ingestion.ingest_text(
                        content,
                        title=title,
                        external_key=ext_key,
                        namespace=namespace,
                        max_chunk_tokens=180,
                        enable_enrichment=cfg.enable_enrichment,
                        enable_graph=cfg.enable_graph,
                        skip_embeddings=cfg.skip_embeddings,
                    )
                doc_id_to_ext_key[str(res.document_id)] = ext_key

            recall5_scores: list[float] = []
            recall10_scores: list[float] = []
            mrr10_scores: list[float] = []
            ndcg10_scores: list[float] = []
            cit_scores: list[float] = []
            abstain_pairs: list[tuple[bool, bool]] = []
            latencies_ms: list[float] = []
            term_hits = 0
            answerable_count = 0

            for q in queries:
                question = str(q["question"])
                answerable = bool(q["answerable"])
                rel_keys = [str(k) for k in q.get("relevant_external_keys", [])]
                expected_terms = [str(t).lower() for t in q.get("expected_terms", [])]

                t0 = time.monotonic()
                search_res = await runtime.retrieval.search(
                    question,
                    RetrievalFilter(namespace=namespace),
                    mode=cfg.retrieval_mode,
                    top_k=10,
                    enable_graph=cfg.enable_graph,
                    enable_rerank=cfg.enable_rerank,
                    enable_parent_expand=cfg.enable_parent_expand,
                )
                rag_res = await runtime.rag.run(
                    RagRequest(
                        question=question,
                        namespace=namespace,
                        top_k=6,
                        retrieval_mode=cfg.retrieval_mode,
                        enable_rerank=cfg.enable_rerank,
                        enable_parent_expand=cfg.enable_parent_expand,
                        enable_graph_expansion=cfg.enable_graph,
                        enable_outline_navigation=cfg.enable_outline_nav,
                        enable_crag=cfg.enable_crag,
                        strict_citation_mode=cfg.strict_citation,
                        channel="eval",
                    )
                )
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                latencies_ms.append(elapsed_ms)

                retrieved_ext_keys = [
                    doc_id_to_ext_key.get(str(ev.document_id), "") for ev in search_res.evidence
                ]
                if answerable and rel_keys:
                    answerable_count += 1
                    recall5_scores.append(recall_at_k(retrieved_ext_keys, rel_keys, 5))
                    recall10_scores.append(recall_at_k(retrieved_ext_keys, rel_keys, 10))
                    mrr10_scores.append(mrr_at_k(retrieved_ext_keys, rel_keys, 10))
                    ndcg10_scores.append(ndcg_at_k(retrieved_ext_keys, rel_keys, 10))

                    if rag_res.status == RagDecisionStatus.ANSWERED:
                        trace = await runtime.trace_repo.get_trace(rag_res.trace_id)
                        allowed: list[str] = []
                        if trace is not None:
                            for sp in trace.spans:
                                if sp.node_name == "rerank_and_expand":
                                    raw_ids = sp.output_meta.get("chunk_ids", [])
                                    allowed = [str(cid) for cid in raw_ids]
                        if not allowed:
                            allowed = [str(ev.chunk_id) for ev in search_res.evidence]
                        cited = [str(c.chunk_id) for c in rag_res.citations]
                        cit_scores.append(citation_validity(cited, allowed))
                        ans_low = rag_res.answer.lower()
                        if expected_terms and any(t in ans_low for t in expected_terms):
                            term_hits += 1
                    else:
                        cit_scores.append(0.0)

                did_abstain = rag_res.status == RagDecisionStatus.ABSTAINED
                abstain_pairs.append((not answerable, did_abstain))

            abs_f1, far = abstention_f1(abstain_pairs)
            row_summary = {
                "config": cfg.name,
                "queries_evaluated": len(queries),
                "recall_at_5": round(sum(recall5_scores) / max(1, len(recall5_scores)), 4),
                "recall_at_10": round(sum(recall10_scores) / max(1, len(recall10_scores)), 4),
                "mrr_at_10": round(sum(mrr10_scores) / max(1, len(mrr10_scores)), 4),
                "ndcg_at_10": round(sum(ndcg10_scores) / max(1, len(ndcg10_scores)), 4),
                "citation_validity": round(sum(cit_scores) / max(1, len(cit_scores)), 4),
                "answer_term_match": round(term_hits / max(1, answerable_count), 4),
                "abstention_f1": round(abs_f1, 4),
                "false_answer_rate": round(far, 4),
                "p50_latency_ms": round(percentile_ms(latencies_ms, 50), 2),
                "p95_latency_ms": round(percentile_ms(latencies_ms, 95), 2),
                "avg_ingest_tokens": round(ingest_tokens / max(1, len(docs)), 1),
            }
            results.append(row_summary)

        report = {
            "benchmark": "rag_ablation",
            "dataset": raw_dataset.get("name", dataset_path.stem),
            "mode": "dry_run_deterministic" if dry_run else "live_llm",
            "query_count": len(queries),
            "results": results,
        }
        json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        md_out.write_text(_render_markdown(report), encoding="utf-8")
        return report
    finally:
        await runtime.shutdown()


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# TerraceMem RAG Ablation Report ({report['dataset']} — {report['mode']})",
        "",
        (
            "| Config | Recall@5 | MRR@10 | NDCG@10 | Citation Validity "
            "| Term Match | Abstention F1 | FAR | P95 (ms) |"
        ),
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in report["results"]:
        lines.append(
            f"| `{r['config']}` "
            f"| {r['recall_at_5']:.3f} "
            f"| {r['mrr_at_10']:.3f} "
            f"| {r['ndcg_at_10']:.3f} "
            f"| {r['citation_validity']:.3f} "
            f"| {r['answer_term_match']:.3f} "
            f"| {r['abstention_f1']:.3f} "
            f"| {r['false_answer_rate']:.3f} "
            f"| {r['p95_latency_ms']:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run TerraceMem 7-group RAG ablation.")
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument(
        "--configs",
        type=str,
        default="B0,B1,B_kw_only,B2,B3,B4,B5",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("evals/reports"))
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--live", action="store_true", help="Use configured LLM/Embedding APIs.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    cfg_names = [c.strip() for c in args.configs.split(",") if c.strip() in ABLATION_MATRIX]
    if not cfg_names:
        print("Error: no valid configs selected.", file=sys.stderr)
        return 1

    report = asyncio.run(
        run_rag_ablation(
            dataset_path=args.dataset,
            config_names=cfg_names,
            output_dir=args.output_dir,
            dry_run=not args.live,
            resume=args.resume,
        )
    )
    print(_render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
