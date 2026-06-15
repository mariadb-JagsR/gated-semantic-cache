"""Quora Question Pairs benchmark for semantic cache precision/recall.

Each row is (question1, question2, is_duplicate). We seed the cache with question1,
then probe with question2. Human duplicate labels become expected cache-hit labels.
"""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from gated_semantic_cache.eval.cache_hit_metrics import CacheHitMetrics, compute_cache_hit_metrics
from gated_semantic_cache.models.context import DEFAULT_SEMANTIC_LOW_WATERMARK, RequestContext
from gated_semantic_cache.observability.tracing import RequestTrace
from gated_semantic_cache.routing.features import normalize_query_text
from gated_semantic_cache.routing.labels import RoutingLabel
from gated_semantic_cache.routing.classifier import RoutingPrediction
from gated_semantic_cache.serving.pipeline import PipelineResponse


RoutePolicy = Literal["honest", "semantic_ok", "vector_only"]

_DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "quora" / "quora_duplicate_questions.tsv"
)
_CACHE_HIT_SOURCES = frozenset({"exact_cache", "semantic_cache", "exact_anchor"})


@dataclass(frozen=True, slots=True)
class QuoraPair:
    row_id: str
    qid1: str
    qid2: str
    question1: str
    question2: str
    is_duplicate: bool


@dataclass(frozen=True, slots=True)
class QuoraPairEvalRow:
    row_id: str
    question1: str
    question2: str
    is_duplicate: bool
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
class QuoraPairEvalReport:
    dataset_path: str
    model: str
    semantic_threshold: float
    semantic_low_watermark: float
    route_policy: RoutePolicy
    limit: int | None
    seed: int
    metrics: CacheHitMetrics
    rows: list[QuoraPairEvalRow]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "model": self.model,
            "semantic_threshold": self.semantic_threshold,
            "semantic_low_watermark": self.semantic_low_watermark,
            "route_policy": self.route_policy,
            "limit": self.limit,
            "seed": self.seed,
            "metrics": self.metrics.to_dict(),
            "rows": [asdict(row) for row in self.rows],
        }


def default_quora_dataset_path() -> Path:
    return _DEFAULT_DATA_PATH


def default_quora_reports_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "docs" / "quora_pairs_eval"


def default_quora_report_path(
    *,
    limit: int | None,
    seed: int,
    judge_enabled: bool,
    route_policy: RoutePolicy,
    semantic_threshold: float,
    run_at: datetime | None = None,
) -> Path:
    stamp = (run_at or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    limit_part = str(limit) if limit is not None else "all"
    if route_policy == "vector_only":
        judge_part = "vector-only"
    else:
        judge_part = "judge-on" if judge_enabled else "no-judge"
    threshold_part = f"thresh{semantic_threshold:.2f}".replace(".", "p")
    filename = f"quora_pairs_limit{limit_part}_seed{seed}_{judge_part}_{route_policy}_{threshold_part}_{stamp}.json"
    return default_quora_reports_dir() / filename


def write_quora_pair_report(
    report: QuoraPairEvalReport,
    path: Path | str,
    *,
    judge_enabled: bool,
    balanced: bool,
    run_at: datetime | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    payload["eval_metadata"] = {
        "judge_enabled": judge_enabled,
        "balanced": balanced,
        "run_at": (run_at or datetime.now(timezone.utc)).isoformat(),
        "report_path": str(output),
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def load_quora_pair_report(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_quora_pairs(
    path: Path | str | None = None,
    *,
    limit: int | None = None,
    seed: int = 42,
    balanced: bool = False,
) -> list[QuoraPair]:
    dataset_path = Path(path) if path is not None else default_quora_dataset_path()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Quora dataset not found at {dataset_path}")

    pairs: list[QuoraPair] = []
    with dataset_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            pairs.append(
                QuoraPair(
                    row_id=str(row["id"]),
                    qid1=str(row["qid1"]),
                    qid2=str(row["qid2"]),
                    question1=str(row["question1"]).strip(),
                    question2=str(row["question2"]).strip(),
                    is_duplicate=str(row["is_duplicate"]).strip() == "1",
                )
            )

    if balanced:
        positives = [pair for pair in pairs if pair.is_duplicate]
        negatives = [pair for pair in pairs if not pair.is_duplicate]
        sample_size = min(len(positives), len(negatives))
        if limit is not None:
            sample_size = min(sample_size, limit // 2)
        rng = random.Random(seed)
        pairs = rng.sample(positives, sample_size) + rng.sample(negatives, sample_size)
        rng.shuffle(pairs)
        if limit is not None:
            pairs = pairs[:limit]
        return pairs

    if limit is not None:
        rng = random.Random(seed)
        if limit >= len(pairs):
            return pairs
        pairs = rng.sample(pairs, limit)
    return pairs


def _semantic_ok_router(_: str) -> RoutingPrediction:
    return RoutingPrediction(
        label=RoutingLabel.SEMANTIC_OK,
        confidence=1.0,
        probabilities={label: float(label is RoutingLabel.SEMANTIC_OK) for label in RoutingLabel},
    )


def _is_cache_hit(source: str) -> bool:
    return source in _CACHE_HIT_SOURCES


def _routing_blocked(source: str, trace: Any) -> bool:
    return source == "miss" and trace.routing_label in {
        RoutingLabel.SKIP_CACHE.value,
        RoutingLabel.EXACT_ONLY.value,
        RoutingLabel.THREAD_SCOPED_ONLY.value,
    }


def _vector_only_context(context: RequestContext) -> RequestContext:
    """Single-threshold admission for GPTCache-style vector-only lookup."""
    return RequestContext(
        namespace=context.namespace,
        agent_version=context.agent_version,
        corpus_version=context.corpus_version,
        tool_or_schema_version=context.tool_or_schema_version,
        thread_scope_key=context.thread_scope_key,
        prior_user_queries=context.prior_user_queries,
        semantic_threshold=context.semantic_threshold,
        semantic_low_watermark=context.semantic_threshold,
        neighbor_judge_similarity_ceiling=context.neighbor_judge_similarity_ceiling,
        neighbor_judge_max_calls=context.neighbor_judge_max_calls,
        neighbor_judge_ambiguity_margin=context.neighbor_judge_ambiguity_margin,
        semantic_ok_min_route_confidence=context.semantic_ok_min_route_confidence,
        exact_only_min_route_confidence=context.exact_only_min_route_confidence,
        exact_context=dict(context.exact_context),
        cache_namespace=context.cache_namespace,
        reuse_scope_key=context.reuse_scope_key,
    )


def _vector_only_lookup(
    pipeline: Any,
    query: str,
    context: RequestContext,
) -> PipelineResponse:
    """GPTCache-style path: embed, ANN top-1, hit iff similarity >= threshold."""
    normalized = normalize_query_text(query)
    trace = RequestTrace(normalized_query=normalized, semantic_lookup_attempted=True)
    trace.routing_label = "VECTOR_ONLY"
    trace.routing_confidence = 1.0

    lookup_context = _vector_only_context(context)
    embedding = pipeline.embedder(normalized)
    lookup = pipeline.semantic_store.semantic_lookup(
        embedding=embedding,
        context=lookup_context,
        required_thread_scope=None,
        top_k=1,
    )
    trace.top_candidate_similarity = lookup.similarity
    trace.second_candidate_similarity = lookup.second_best_similarity
    trace.candidate_count = lookup.candidate_count
    trace.semantic_neighbor_filter_counts = dict(lookup.neighbor_filter_counts)

    if lookup.hit is None or lookup.similarity is None:
        trace.rejected_reasons = list(lookup.rejected_reasons) or ["no_candidate"]
        trace.final_result_source = "miss"
        return PipelineResponse(source="miss", payload={}, trace=trace)

    if lookup.similarity < context.semantic_threshold:
        trace.semantic_post_ann_reject_reason = "below_threshold"
        trace.rejected_reasons = ["below_threshold"]
        trace.final_result_source = "miss"
        return PipelineResponse(source="miss", payload={}, trace=trace)

    trace.final_result_source = "semantic_cache"
    return PipelineResponse(source="semantic_cache", payload=lookup.hit.response_payload, trace=trace)


def run_quora_pairs_eval(
    *,
    dataset_path: Path | str | None = None,
    limit: int | None = 200,
    seed: int = 42,
    balanced: bool = True,
    semantic_threshold: float | None = None,
    semantic_low_watermark: float | None = None,
    openai_model: str | None = None,
    route_policy: RoutePolicy = "semantic_ok",
    neighbor_judge: Any | None = None,
    progress_every: int = 25,
) -> QuoraPairEvalReport:
    from gated_semantic_cache.cli import build_pipeline

    path = Path(dataset_path) if dataset_path is not None else default_quora_dataset_path()
    pairs = load_quora_pairs(path, limit=limit, seed=seed, balanced=balanced)

    threshold = semantic_threshold if semantic_threshold is not None else float(os.environ.get("SEMANTIC_THRESHOLD", "0.86"))
    low_watermark = (
        semantic_low_watermark
        if semantic_low_watermark is not None
        else float(os.environ.get("SEMANTIC_LOW_WATERMARK", str(DEFAULT_SEMANTIC_LOW_WATERMARK)))
    )
    model = openai_model or os.environ.get("OPENAI_MODEL", "text-embedding-3-small")
    context = RequestContext(semantic_threshold=threshold, semantic_low_watermark=low_watermark)

    effective_judge = None if route_policy == "vector_only" else neighbor_judge
    rows: list[QuoraPairEvalRow] = []
    for index, pair in enumerate(pairs, start=1):
        pipeline = build_pipeline(openai_model=model, embed_cache=True, neighbor_judge=effective_judge)
        if route_policy in {"semantic_ok", "vector_only"}:
            pipeline.router.predict = _semantic_ok_router  # type: ignore[method-assign]

        pipeline.answer_query(pair.question1, context, _seed_answer)
        if route_policy == "vector_only":
            response = _vector_only_lookup(pipeline, pair.question2, context)
        else:
            response = pipeline.lookup_query(pair.question2, context)
        trace = response.trace
        actual_hit = _is_cache_hit(response.source)
        expected_hit = pair.is_duplicate
        rows.append(
            QuoraPairEvalRow(
                row_id=pair.row_id,
                question1=pair.question1,
                question2=pair.question2,
                is_duplicate=pair.is_duplicate,
                expected_cache_hit=expected_hit,
                actual_cache_hit=actual_hit,
                passed=actual_hit is expected_hit,
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
        if progress_every > 0 and index % progress_every == 0:
            print(f"quora-pairs-eval: processed {index}/{len(pairs)}", flush=True)

    metrics = compute_cache_hit_metrics(
        expected_hits=[row.expected_cache_hit for row in rows],
        actual_hits=[row.actual_cache_hit for row in rows],
        routing_blocked=[row.routing_blocked for row in rows],
    )
    return QuoraPairEvalReport(
        dataset_path=str(path),
        model=model,
        semantic_threshold=threshold,
        semantic_low_watermark=low_watermark,
        route_policy=route_policy,
        limit=limit,
        seed=seed,
        metrics=metrics,
        rows=rows,
    )


def _seed_answer(query: str, _context: RequestContext) -> dict[str, Any]:
    return {"answer": f"seed:{query}", "success": True}
