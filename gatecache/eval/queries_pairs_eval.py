"""Curated query-pair benchmark from ``tests/queries.txt``.

Each consecutive pair is (seed_query, probe_query) with a human-labeled expectation
for whether semantic cache reuse is safe. Compare full-stack vs GPTCache-style paths.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from gatecache.eval.cache_hit_metrics import CacheHitMetrics, compute_cache_hit_metrics
from gatecache.eval.queries_regression import load_queries_from_file
from gatecache.eval.quora_pairs_eval import (
    RoutePolicy,
    _is_cache_hit,
    _routing_blocked,
    _semantic_ok_router,
    _vector_only_lookup,
)
from gatecache.models.context import DEFAULT_SEMANTIC_LOW_WATERMARK, RequestContext
from gatecache.routing.labels import RoutingLabel


@dataclass(frozen=True, slots=True)
class QueryPairScenario:
    pair_id: str
    seed_query: str
    probe_query: str
    expected_cache_hit: bool
    category: str
    notes: str


@dataclass(frozen=True, slots=True)
class QueryPairEvalRow:
    pair_id: str
    seed_query: str
    probe_query: str
    category: str
    expected_cache_hit: bool
    actual_cache_hit: bool
    passed: bool
    source: str
    routing_label: str | None
    routing_confidence: float | None
    routing_blocked: bool
    top_candidate_similarity: float | None
    rejected_reasons: list[str]
    semantic_post_ann_reject_reason: str | None
    semantic_facet_conflict_reason: str | None
    semantic_constraint_risk_reason: str | None
    neighbor_judge_invoked: bool


@dataclass(frozen=True, slots=True)
class QueryPairEvalReport:
    queries_file: str
    model: str
    semantic_threshold: float
    semantic_low_watermark: float
    route_policy: RoutePolicy
    metrics: CacheHitMetrics
    rows: list[QueryPairEvalRow]

    def to_dict(self) -> dict[str, Any]:
        return {
            "queries_file": self.queries_file,
            "model": self.model,
            "semantic_threshold": self.semantic_threshold,
            "semantic_low_watermark": self.semantic_low_watermark,
            "route_policy": self.route_policy,
            "metrics": self.metrics.to_dict(),
            "rows": [asdict(row) for row in self.rows],
        }


# Labeled pairs from tests/queries.txt (seed, probe). Expectations encode safe reuse intent.
_DEFAULT_SCENARIOS: tuple[QueryPairScenario, ...] = (
    QueryPairScenario(
        "lisinopril_paraphrase",
        "What are the common side effects of Lisinopril?",
        "Tell me about the adverse reactions associated with ACE inhibitors like Lisinopril.",
        True,
        "safe_paraphrase",
        "Drug-specific vs class paraphrase — should reuse",
    ),
    QueryPairScenario(
        "metformin_paraphrase",
        "Does Metformin cause stomach issues?",
        "Is gastrointestinal distress a frequent complication of taking Metformin?",
        True,
        "safe_paraphrase",
        "Metformin GI paraphrase — should reuse",
    ),
    QueryPairScenario(
        "hipaa_regulatory",
        "Summarize the HIPAA guidelines for patient data sharing.",
        "Give me a rundown of the regulatory requirements for PHI disclosure under HIPAA.",
        False,
        "skip_cache",
        "PHI/regulatory — router should skip cache",
    ),
    QueryPairScenario(
        "amoxicillin_population",
        "What is the recommended dosage for Pediatric Amoxicillin?",
        "What is the recommended dosage for Adult Amoxicillin?",
        False,
        "facet_conflict",
        "Pediatric vs adult dosing — must not reuse",
    ),
    QueryPairScenario(
        "diabetes_subtype",
        "How do I treat a Type 1 diabetic with low blood sugar?",
        "How do I treat a Type 2 diabetic with low blood sugar?",
        False,
        "facet_conflict",
        "Diabetes subtype change — must not reuse",
    ),
    QueryPairScenario(
        "drug_interaction_entity",
        "Is it safe to take Aspirin with Warfarin?",
        "Is it safe to take Aspirin with Vitamin C?",
        False,
        "facet_conflict",
        "Different co-medication — must not reuse",
    ),
    QueryPairScenario(
        "patient_id",
        "Show me the last lab results for patient ID: 88291.",
        "What were the laboratory findings for patient ID: 11022?",
        False,
        "identifier_conflict",
        "Different patient IDs — must not reuse",
    ),
    QueryPairScenario(
        "patient_name_thread",
        "Can you retrieve the discharge summary for John Doe?",
        "I need the discharge summary for Jane Smith.",
        False,
        "thread_scoped",
        "Different patient names — thread-scoped, must not reuse",
    ),
    QueryPairScenario(
        "er_freshness",
        "What is the current wait time at the St. Jude's ER right now?",
        "How long is the wait at the emergency room at St. Jude's?",
        False,
        "freshness",
        "Live ER wait — freshness-sensitive",
    ),
    QueryPairScenario(
        "personal_glucose",
        "Check my latest blood glucose reading from my Dexcom.",
        "What was my most recent sugar level?",
        False,
        "personalized",
        "Personal device reading — individualized",
    ),
    QueryPairScenario(
        "pe_negation",
        "List the symptoms of a pulmonary embolism.",
        "What symptoms exclude a diagnosis of pulmonary embolism?",
        False,
        "negation",
        "Affirmation vs exclusion — polarity flip",
    ),
    QueryPairScenario(
        "trial_polarity",
        "Which patients are eligible for the clinical trial?",
        "Which patients are ineligible for the clinical trial?",
        False,
        "negation",
        "Eligible vs ineligible — polarity flip",
    ),
)


def default_queries_path() -> Path:
    return Path(__file__).resolve().parents[2] / "tests" / "queries.txt"


def default_queries_pairs_reports_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "docs" / "queries_pairs_eval"


def default_finance_pairs_path() -> Path:
    return Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "finance_adversarial_pairs.json"


def default_finance_pairs_reports_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "docs" / "finance_pairs_eval"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def load_query_pair_scenarios_from_json(path: Path | str) -> list[QueryPairScenario]:
    """Load adversarial seed/probe pairs from a JSON category file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON array in {path}")

    scenarios: list[QueryPairScenario] = []
    for block in payload:
        category = str(block["category"])
        cat_slug = _slug(category)
        pairs = block.get("pairs") or []
        for index, pair in enumerate(pairs):
            scenarios.append(
                QueryPairScenario(
                    pair_id=f"{cat_slug}_{index}",
                    seed_query=str(pair["a"]).strip(),
                    probe_query=str(pair["b"]).strip(),
                    expected_cache_hit=bool(pair.get("expected_cache_hit", False)),
                    category=category,
                    notes=str(pair.get("why_different", "")),
                )
            )
    return scenarios


def load_query_pair_scenarios(
    *,
    queries_file: Path | str | None = None,
    scenarios: tuple[QueryPairScenario, ...] | None = None,
) -> list[QueryPairScenario]:
    path = Path(queries_file) if queries_file is not None else default_queries_path()
    if scenarios is not None:
        return list(scenarios)

    queries = load_queries_from_file(path)
    if len(queries) % 2 != 0:
        raise ValueError(f"Expected even number of queries in {path}, got {len(queries)}")
    if len(queries) // 2 != len(_DEFAULT_SCENARIOS):
        raise ValueError(
            f"queries.txt pair count ({len(queries) // 2}) does not match labeled scenarios ({len(_DEFAULT_SCENARIOS)})"
        )

    out: list[QueryPairScenario] = []
    for index, scenario in enumerate(_DEFAULT_SCENARIOS):
        seed = queries[index * 2]
        probe = queries[index * 2 + 1]
        if seed != scenario.seed_query or probe != scenario.probe_query:
            raise ValueError(
                f"queries.txt pair {index + 1} mismatch:\n"
                f"  file: ({seed!r}, {probe!r})\n"
                f"  expected: ({scenario.seed_query!r}, {scenario.probe_query!r})"
            )
        out.append(scenario)
    return out


def run_queries_pairs_eval(
    *,
    queries_file: Path | str | None = None,
    pairs_json: Path | str | None = None,
    semantic_threshold: float | None = None,
    semantic_low_watermark: float | None = None,
    openai_model: str | None = None,
    route_policy: RoutePolicy = "honest",
    neighbor_judge: Any | None = None,
) -> QueryPairEvalReport:
    from gatecache.cli import build_pipeline

    if pairs_json is not None:
        path = Path(pairs_json)
        scenarios = load_query_pair_scenarios_from_json(path)
    else:
        path = Path(queries_file) if queries_file is not None else default_queries_path()
        scenarios = load_query_pair_scenarios(queries_file=path)

    threshold = semantic_threshold if semantic_threshold is not None else float(os.environ.get("SEMANTIC_THRESHOLD", "0.86"))
    low_watermark = (
        semantic_low_watermark
        if semantic_low_watermark is not None
        else float(os.environ.get("SEMANTIC_LOW_WATERMARK", str(DEFAULT_SEMANTIC_LOW_WATERMARK)))
    )
    model = openai_model or os.environ.get("OPENAI_MODEL", "text-embedding-3-small")
    context = RequestContext(semantic_threshold=threshold, semantic_low_watermark=low_watermark)

    effective_judge = None if route_policy == "vector_only" else neighbor_judge
    rows: list[QueryPairEvalRow] = []

    for scenario in scenarios:
        pipeline = build_pipeline(openai_model=model, embed_cache=True, neighbor_judge=effective_judge)
        if route_policy in {"semantic_ok", "vector_only"}:
            pipeline.router.predict = _semantic_ok_router  # type: ignore[method-assign]

        pipeline.answer_query(scenario.seed_query, context, _seed_answer)
        if route_policy == "vector_only":
            response = _vector_only_lookup(pipeline, scenario.probe_query, context)
        else:
            response = pipeline.lookup_query(scenario.probe_query, context)

        trace = response.trace
        actual_hit = _is_cache_hit(response.source)
        rows.append(
            QueryPairEvalRow(
                pair_id=scenario.pair_id,
                seed_query=scenario.seed_query,
                probe_query=scenario.probe_query,
                category=scenario.category,
                expected_cache_hit=scenario.expected_cache_hit,
                actual_cache_hit=actual_hit,
                passed=actual_hit is scenario.expected_cache_hit,
                source=response.source,
                routing_label=trace.routing_label,
                routing_confidence=trace.routing_confidence,
                routing_blocked=_routing_blocked(response.source, trace),
                top_candidate_similarity=trace.top_candidate_similarity,
                rejected_reasons=list(trace.rejected_reasons),
                semantic_post_ann_reject_reason=trace.semantic_post_ann_reject_reason,
                semantic_facet_conflict_reason=trace.semantic_facet_conflict_reason,
                semantic_constraint_risk_reason=trace.semantic_constraint_risk_reason,
                neighbor_judge_invoked=trace.neighbor_judge_invoked,
            )
        )

    metrics = compute_cache_hit_metrics(
        expected_hits=[row.expected_cache_hit for row in rows],
        actual_hits=[row.actual_cache_hit for row in rows],
        routing_blocked=[row.routing_blocked for row in rows],
    )
    return QueryPairEvalReport(
        queries_file=str(path),
        model=model,
        semantic_threshold=threshold,
        semantic_low_watermark=low_watermark,
        route_policy=route_policy,
        metrics=metrics,
        rows=rows,
    )


def write_queries_pairs_report(
    report: QueryPairEvalReport,
    path: Path | str,
    *,
    judge_enabled: bool,
    run_at: datetime | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    payload["eval_metadata"] = {
        "judge_enabled": judge_enabled,
        "run_at": (run_at or datetime.now(timezone.utc)).isoformat(),
        "report_path": str(output),
        "pair_count": len(report.rows),
        "expected_hits": sum(1 for row in report.rows if row.expected_cache_hit),
        "expected_misses": sum(1 for row in report.rows if not row.expected_cache_hit),
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def _seed_answer(query: str, _context: RequestContext) -> dict[str, Any]:
    return {"answer": f"seed:{query}", "success": True}
