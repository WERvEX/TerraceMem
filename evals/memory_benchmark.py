"""Reproducible 4-baseline Memory Benchmark (M0..M3) supporting LongMemEval-S & dry-run."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

try:
    from evals.metrics import false_invalidation_rate, percentile_ms, prompt_token_reduction_pct
except ModuleNotFoundError:
    from metrics import false_invalidation_rate, percentile_ms, prompt_token_reduction_pct
from terracemem.config import Settings
from terracemem.llm.providers import DeterministicHashEmbeddingProvider, cosine_similarity
from terracemem.runtime import TerraceMemRuntime
from terracemem.utils.tokens import estimate_tokens


@dataclass(frozen=True)
class MemoryEvalCase:
    """Single multi-session memory benchmark case."""

    case_id: str
    category: str
    turns: list[tuple[str, int, bool]]  # (content, day_offset, is_pinned)
    question: str
    expected_answer_terms: list[str]
    forbidden_stale_terms: list[str]
    protected_expectation: bool = False
    expected_pinned_substring: str | None = None


def _builtin_dry_run_cases() -> list[MemoryEvalCase]:
    return [
        MemoryEvalCase(
            case_id="ku_01",
            category="knowledge-update",
            turns=[
                ("Our primary database is PostgreSQL [slot:project.db].", 0, False),
                ("We have 14 background microservices running.", 2, False),
                ("We migrated our primary database to DuckDB [slot:project.db].", 5, False),
            ],
            question="What is our current primary database?",
            expected_answer_terms=["duckdb"],
            forbidden_stale_terms=["postgresql"],
        ),
        MemoryEvalCase(
            case_id="ku_02",
            category="knowledge-update",
            turns=[
                ("I currently live in Beijing [slot:user.location].", 0, False),
                ("My favorite team sport is badminton.", 3, False),
                ("I relocated to Shanghai this month [slot:user.location].", 10, False),
            ],
            question="Where do I currently live?",
            expected_answer_terms=["shanghai"],
            forbidden_stale_terms=["beijing"],
        ),
        MemoryEvalCase(
            case_id="ku_03",
            category="knowledge-update",
            turns=[
                ("Our project cache uses Memcached [slot:project.cache].", 0, False),
                ("Our project cache now uses Valkey [slot:project.cache].", 8, False),
            ],
            question="Which cache system does the project use now?",
            expected_answer_terms=["valkey"],
            forbidden_stale_terms=["memcached"],
        ),
        MemoryEvalCase(
            case_id="tr_01",
            category="temporal-reasoning",
            turns=[
                ("Our web framework was Flask [slot:project.framework].", 1, False),
                ("We upgraded our web framework to FastAPI [slot:project.framework].", 12, False),
                (
                    "Back in 2024 we initially prototyped on Flask [slot:project.framework].",
                    2,
                    False,
                ),
            ],
            question="Which web framework is active right now?",
            expected_answer_terms=["fastapi"],
            forbidden_stale_terms=["flask"],
        ),
        MemoryEvalCase(
            case_id="ms_01",
            category="multi-session",
            turns=[
                ("I prefer Python for backend tooling [slot:user.language].", 0, False),
                ("My primary editor is Neovim [slot:user.editor].", 4, False),
                ("Today we reviewed CI timeouts and benchmark latency.", 6, False),
                ("Let us prepare the release checklist for Friday.", 8, False),
                ("Please verify telemetry export before tagging.", 9, False),
                ("Remember to include license headers on new files.", 10, False),
                ("We also updated the Docker Compose healthcheck.", 11, False),
                ("Check Alembic revision chain for migration errors.", 12, False),
                ("Verify SSE streaming disconnects cleanly.", 13, False),
                ("Ensure ruff and mypy pass before merge.", 14, False),
                ("Tag v0.1.0 once green.", 15, False),
                ("Archive build logs.", 16, False),
            ],
            question="What programming language and editor do I prefer?",
            expected_answer_terms=["python", "neovim"],
            forbidden_stale_terms=[],
        ),
        MemoryEvalCase(
            case_id="abs_01",
            category="abstention",
            turns=[
                ("Our primary database is PostgreSQL [slot:project.db].", 0, False),
                ("Maybe switch to SQLite later? [slot:project.db]", 3, False),
            ],
            question="Did we switch the database to SQLite?",
            expected_answer_terms=["postgresql"],
            forbidden_stale_terms=["sqlite"],
            protected_expectation=True,
            expected_pinned_substring="PostgreSQL",
        ),
        MemoryEvalCase(
            case_id="pin_01",
            category="single-session-preference",
            turns=[
                (
                    "Never drop production database tables without backup [slot:ops.backup].",
                    0,
                    True,
                ),
                (
                    "We can skip backups and drop production tables quickly [slot:ops.backup].",
                    4,
                    False,
                ),
            ],
            question="What is our rule about dropping production tables?",
            expected_answer_terms=["never drop"],
            forbidden_stale_terms=["skip backups"],
            protected_expectation=True,
            expected_pinned_substring="Never drop",
        ),
        MemoryEvalCase(
            case_id="pref_01",
            category="single-session-preference",
            turns=[
                (
                    "I prefer UTC+8 timezone for all schedule digests [slot:user.timezone].",
                    0,
                    False,
                ),
                ("Today I ran unit tests.", 5, False),
            ],
            question="Which timezone do I prefer for schedules?",
            expected_answer_terms=["utc+8"],
            forbidden_stale_terms=[],
        ),
    ]


def _load_longmemeval(path: Path, *, limit: int = 80) -> list[MemoryEvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases: list[MemoryEvalCase] = []
    for idx, item in enumerate(raw[:limit]):
        q_type = str(item.get("question_type", "multi-session"))
        question = str(item.get("question", ""))
        answer = str(item.get("answer", "")).strip().lower()
        sessions = item.get("haystack_sessions", [])
        turns: list[tuple[str, int, bool]] = []
        for day, sess in enumerate(sessions):
            for msg in sess:
                content = str(msg.get("content", "")).strip()
                if content:
                    turns.append((content[:400], day, False))
        cases.append(
            MemoryEvalCase(
                case_id=str(item.get("question_id", f"lme_{idx}")),
                category=q_type,
                turns=turns,
                question=question,
                expected_answer_terms=[answer] if answer else [],
                forbidden_stale_terms=[],
            )
        )
    return cases


def _judge_context(context_text: str, case: MemoryEvalCase) -> bool:
    low = context_text.lower()
    has_expected = all(term in low for term in case.expected_answer_terms)
    has_forbidden = any(bad in low for bad in case.forbidden_stale_terms)
    return has_expected and not has_forbidden


async def run_memory_benchmark(
    *,
    longmemeval_path: Path | None,
    output_dir: Path,
    dry_run: bool = True,
) -> dict[str, Any]:
    cases = (
        _load_longmemeval(longmemeval_path)
        if (longmemeval_path and longmemeval_path.exists())
        else _builtin_dry_run_cases()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_out = output_dir / "memory_benchmark_latest.json"
    md_out = output_dir / "memory_benchmark_latest.md"

    embedder = DeterministicHashEmbeddingProvider(dim=256)
    base_time = datetime(2026, 1, 1, tzinfo=UTC)

    # First calculate M1 Full Context token count per case for Token Reduction %
    m1_tokens_total = 0
    for case in cases:
        m1_tokens_total += estimate_tokens("\n".join(t[0] for t in case.turns))
    avg_m1_tokens = m1_tokens_total / max(1, len(cases))

    baseline_rows: list[dict[str, Any]] = []

    for baseline in ("M0", "M1", "M2", "M3"):
        cat_hits: dict[str, list[bool]] = {}
        latencies_ms: list[float] = []
        prompt_tokens: list[int] = []
        protected_total = 0
        wrongly_invalidated = 0

        runtime = TerraceMemRuntime(
            Settings(
                storage_backend="memory",
                llm_api_key="" if dry_run else Settings().llm_api_key,
                embedding_api_key="" if dry_run else Settings().embedding_api_key,
            ),
            embedder=embedder,
        )
        await runtime.startup()
        try:
            for case in cases:
                t0 = time.monotonic()
                context_text = ""

                if baseline == "M0":
                    recent = [t[0] for t in case.turns[-10:]]
                    context_text = "\n".join(recent)
                    if case.protected_expectation:
                        protected_total += 1
                        if (
                            case.expected_pinned_substring
                            and case.expected_pinned_substring.lower() not in context_text.lower()
                        ):
                            wrongly_invalidated += 1

                elif baseline == "M1":
                    context_text = "\n".join(t[0] for t in case.turns)

                elif baseline == "M2":
                    # Flat vector memory: stores every turn raw without bi-temporal invalidation
                    vecs = await embedder.embed_texts([t[0] for t in case.turns] + [case.question])
                    q_vec = vecs[-1]
                    scored = [
                        (case.turns[i][0], cosine_similarity(vecs[i], q_vec))
                        for i in range(len(case.turns))
                    ]
                    scored.sort(key=lambda pair: -pair[1])
                    context_text = "\n".join(txt for txt, _ in scored[:4])
                    if case.protected_expectation:
                        protected_total += 1
                        if any(bad in context_text.lower() for bad in case.forbidden_stale_terms):
                            wrongly_invalidated += 1

                elif baseline == "M3":
                    uid = f"user_{case.case_id}"
                    for text, day_offset, is_pin in case.turns:
                        await runtime.memory.commit_turn(
                            user_id=uid,
                            content=text,
                            event_time=base_time + timedelta(days=day_offset),
                            is_pinned=is_pin,
                            sync_extract=True,
                        )
                    recalled = await runtime.memory.recall(
                        user_id=uid,
                        query=case.question,
                        limit=4,
                    )
                    lines = [item.memory.fact_text for item in recalled.semantic_memories]
                    context_text = "\n".join(lines)
                    if case.protected_expectation:
                        protected_total += 1
                        tl = await runtime.memory.timeline(user_id=uid)
                        active_text = " ".join(m.fact_text for m in tl.active_at_as_of)
                        if (
                            case.expected_pinned_substring
                            and case.expected_pinned_substring.lower() not in active_text.lower()
                        ):
                            wrongly_invalidated += 1

                latencies_ms.append((time.monotonic() - t0) * 1000.0)
                prompt_tokens.append(estimate_tokens(context_text))
                ok = _judge_context(context_text, case)
                cat_hits.setdefault(case.category, []).append(ok)
        finally:
            await runtime.shutdown()

        all_outcomes = [ok for lst in cat_hits.values() for ok in lst]
        ku_outcomes = cat_hits.get("knowledge-update", [])
        tr_outcomes = cat_hits.get("temporal-reasoning", [])
        ms_outcomes = cat_hits.get("multi-session", [])
        abs_outcomes = cat_hits.get("abstention", [])

        avg_tokens = sum(prompt_tokens) / max(1, len(prompt_tokens))
        baseline_rows.append(
            {
                "baseline": baseline,
                "cases_evaluated": len(cases),
                "overall_accuracy": round(sum(all_outcomes) / max(1, len(all_outcomes)), 4),
                "knowledge_update_accuracy": round(
                    sum(ku_outcomes) / max(1, len(ku_outcomes)), 4
                ),
                "temporal_reasoning_accuracy": round(
                    sum(tr_outcomes) / max(1, len(tr_outcomes)), 4
                ),
                "multi_session_accuracy": round(sum(ms_outcomes) / max(1, len(ms_outcomes)), 4),
                "abstention_accuracy": round(sum(abs_outcomes) / max(1, len(abs_outcomes)), 4),
                "avg_prompt_tokens": round(avg_tokens, 1),
                "token_reduction_pct": round(
                    prompt_token_reduction_pct(avg_m1_tokens, avg_tokens), 2
                ),
                "false_invalidation_rate": round(
                    false_invalidation_rate(protected_total, wrongly_invalidated), 4
                ),
                "p95_latency_ms": round(percentile_ms(latencies_ms, 95), 2),
            }
        )

    report = {
        "benchmark": "memory_benchmark",
        "dataset": "LongMemEval-S" if longmemeval_path else "builtin_bitemporal_suite",
        "mode": "dry_run_deterministic" if dry_run else "live_llm",
        "case_count": len(cases),
        "results": baseline_rows,
    }
    json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_out.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# TerraceMem Memory Benchmark Report ({report['dataset']} — {report['mode']})",
        "",
        (
            "| Baseline | Overall Acc | Knowledge Update | Temporal | Multi-Session "
            "| Abstention | Token Reduction % | False Invalidation Rate | P95 (ms) |"
        ),
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in report["results"]:
        lines.append(
            f"| `{r['baseline']}` "
            f"| {r['overall_accuracy']:.3f} "
            f"| {r['knowledge_update_accuracy']:.3f} "
            f"| {r['temporal_reasoning_accuracy']:.3f} "
            f"| {r['multi_session_accuracy']:.3f} "
            f"| {r['abstention_accuracy']:.3f} "
            f"| {r['token_reduction_pct']:.1f}% "
            f"| {r['false_invalidation_rate']:.3f} "
            f"| {r['p95_latency_ms']:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run TerraceMem M0..M3 memory benchmark.")
    parser.add_argument("--longmemeval-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("evals/reports"))
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)

    report = asyncio.run(
        run_memory_benchmark(
            longmemeval_path=args.longmemeval_path,
            output_dir=args.output_dir,
            dry_run=not args.live,
        )
    )
    print(_render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
