from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Callable

from gatecache.cache.exact_cache import ExactCache, ExactCacheValue
from gatecache.cache.semantic_store import SemanticStore
from gatecache.models.cache_entry import SemanticCacheEntry
from gatecache.models.context import RequestContext
from gatecache.observability.tracing import RequestTrace
from gatecache.routing.classifier import RoutingClassifier
from gatecache.routing.features import normalize_query_text
from gatecache.routing.labels import RoutingLabel
from gatecache.serving.constraint_risk import constraint_risk_reason
from gatecache.serving.insert_policy import should_insert_response
from gatecache.serving.neighbor_judge import NeighborJudge, take_neighbor_judge_observation
from gatecache.serving.policy import build_anchor_key, build_exact_key, cache_policy_for_route
from gatecache.serving.routing_resolve import resolve_effective_route_label
from gatecache.serving.query_facets import extract_query_facets, facet_conflict_reason
from gatecache.serving.structured_reuse_gate import (
    compute_structured_critical_signature,
    identifier_pairs_for_metadata,
    structured_reuse_gate,
)
from gatecache.structured_exact.canonical_key import build_structured_key, canonicalize_constraint
from gatecache.structured_exact.schema import StructuredQuery
from gatecache.structured_exact.structured_query import extract_structured_query


LiveAnswerer = Callable[[str, RequestContext], dict[str, Any]]
Embedder = Callable[[str], list[float]]


def _neighbor_judge_preflight(
    *,
    context: RequestContext,
    top_similarity: float,
    second_similarity: float | None,
    calls_used: int,
) -> tuple[bool, str | None]:
    """Return (skip_judge, reason_if_skipped). When skip_judge is True, reuse proceeds without invoking the judge."""
    ceiling = context.neighbor_judge_similarity_ceiling
    if ceiling is not None and top_similarity >= ceiling:
        return True, "neighbor_judge_skipped_strong_similarity"

    margin = context.neighbor_judge_ambiguity_margin
    if margin is not None and second_similarity is not None:
        if top_similarity - second_similarity >= margin:
            return True, "neighbor_judge_skipped_clear_margin"

    max_calls = context.neighbor_judge_max_calls
    if max_calls is not None and calls_used >= max_calls:
        return True, "neighbor_judge_max_calls_exceeded"

    return False, None


def _attach_neighbor_judge_observation(trace: RequestTrace) -> None:
    observation = take_neighbor_judge_observation()
    if observation is None:
        return
    raw = observation.get("raw")
    if raw is not None:
        trace.neighbor_judge_raw_response = str(raw)
    decision = observation.get("decision")
    if isinstance(decision, dict):
        trace.neighbor_judge_decision = decision
    status = observation.get("status")
    if status is not None:
        trace.neighbor_judge_response_status = str(status)


def _run_neighbor_judge(
    judge: NeighborJudge,
    *,
    query: str,
    hit: SemanticCacheEntry,
    context: RequestContext,
    trace: RequestTrace,
) -> str | None:
    reason = judge(query, hit, context)
    _attach_neighbor_judge_observation(trace)
    return reason


@dataclass(slots=True)
class PipelineResponse:
    source: str
    payload: dict[str, Any]
    trace: RequestTrace


class SemanticCachePipeline:
    def __init__(
        self,
        *,
        router: RoutingClassifier,
        exact_cache: ExactCache,
        semantic_store: SemanticStore,
        embedder: Embedder,
        neighbor_judge: NeighborJudge | None = None,
    ) -> None:
        self.router = router
        self.exact_cache = exact_cache
        self.semantic_store = semantic_store
        self.embedder = embedder
        self.neighbor_judge = neighbor_judge

    def answer_query(
        self,
        query: str,
        context: RequestContext,
        live_answerer: LiveAnswerer,
    ) -> PipelineResponse:
        normalized = normalize_query_text(query)
        trace = RequestTrace(normalized_query=normalized, exact_cache_attempted=True)
        exact_key = build_exact_key(normalized, context)
        trace.exact_cache_key_sha256 = exact_key
        exact_hit = self.exact_cache.get(exact_key)
        if exact_hit is not None:
            trace.exact_cache_hit = True
            trace.final_result_source = exact_hit.source
            return PipelineResponse(source=exact_hit.source, payload=exact_hit.payload, trace=trace)

        sq = extract_structured_query(query)
        self._populate_structured_trace(sq, context, trace)

        routing = self.router.predict(query)
        trace.routing_label = routing.label.value
        trace.routing_confidence = routing.confidence
        effective_label, anchor_key, exact_downgraded = resolve_effective_route_label(
            query=query,
            context=context,
            predicted_label=routing.label,
            routing_confidence=routing.confidence,
        )
        if exact_downgraded:
            trace.routing_exact_only_downgraded = True

        if (
            routing.label is RoutingLabel.SEMANTIC_OK
            and context.semantic_ok_min_route_confidence is not None
            and routing.confidence < context.semantic_ok_min_route_confidence
        ):
            trace.routing_semantic_ok_downgraded = True
            return self._live_and_maybe_insert(
                query, normalized, sq, context, RoutingLabel.SKIP_CACHE, live_answerer, trace
            )

        if routing.label is RoutingLabel.SKIP_CACHE:
            return self._live_and_maybe_insert(query, normalized, sq, context, routing.label, live_answerer, trace)

        if effective_label is RoutingLabel.EXACT_ONLY:
            if anchor_key:
                trace.anchor_lookup_key_sha256 = anchor_key
                anchor_entry = self.semantic_store.get_anchor(anchor_key)
                if anchor_entry is not None:
                    trace.final_result_source = "exact_anchor"
                    return PipelineResponse(
                        source="exact_anchor",
                        payload=anchor_entry.response_payload,
                        trace=trace,
                    )
                return self._live_and_maybe_insert(query, normalized, sq, context, routing.label, live_answerer, trace)

            return self._live_and_maybe_insert(query, normalized, sq, context, routing.label, live_answerer, trace)

        if routing.label is RoutingLabel.THREAD_SCOPED_ONLY and not context.thread_scope_key:
            return self._live_and_maybe_insert(query, normalized, sq, context, routing.label, live_answerer, trace)

        required_thread_scope = context.thread_scope_key if routing.label is RoutingLabel.THREAD_SCOPED_ONLY else None
        semantic_hit = self._semantic_lookup(
            query, normalized, sq, context, trace, required_thread_scope=required_thread_scope
        )
        if semantic_hit is not None:
            trace.final_result_source = "semantic_cache"
            return PipelineResponse(source="semantic_cache", payload=semantic_hit.response_payload, trace=trace)

        insert_label = effective_label if exact_downgraded else routing.label
        return self._live_and_maybe_insert(query, normalized, sq, context, insert_label, live_answerer, trace)

    def lookup_query(
        self,
        query: str,
        context: RequestContext,
    ) -> PipelineResponse:
        """Lookup-only path for public cache APIs.

        Unlike ``answer_query()``, this never calls a live answerer and never inserts on a miss.
        """

        normalized = normalize_query_text(query)
        trace = RequestTrace(normalized_query=normalized, exact_cache_attempted=True)
        exact_key = build_exact_key(normalized, context)
        trace.exact_cache_key_sha256 = exact_key
        exact_hit = self.exact_cache.get(exact_key)
        if exact_hit is not None:
            trace.exact_cache_hit = True
            trace.final_result_source = exact_hit.source
            return PipelineResponse(source=exact_hit.source, payload=exact_hit.payload, trace=trace)

        sq = extract_structured_query(query)
        self._populate_structured_trace(sq, context, trace)

        routing = self.router.predict(query)
        trace.routing_label = routing.label.value
        trace.routing_confidence = routing.confidence
        effective_label, anchor_key, exact_downgraded = resolve_effective_route_label(
            query=query,
            context=context,
            predicted_label=routing.label,
            routing_confidence=routing.confidence,
        )
        if exact_downgraded:
            trace.routing_exact_only_downgraded = True

        if (
            routing.label is RoutingLabel.SEMANTIC_OK
            and context.semantic_ok_min_route_confidence is not None
            and routing.confidence < context.semantic_ok_min_route_confidence
        ):
            trace.routing_semantic_ok_downgraded = True
            trace.final_result_source = "miss"
            return PipelineResponse(source="miss", payload={}, trace=trace)

        if routing.label is RoutingLabel.SKIP_CACHE:
            trace.final_result_source = "miss"
            return PipelineResponse(source="miss", payload={}, trace=trace)

        if effective_label is RoutingLabel.EXACT_ONLY:
            if anchor_key:
                trace.anchor_lookup_key_sha256 = anchor_key
                anchor_entry = self.semantic_store.get_anchor(anchor_key)
                if anchor_entry is not None:
                    trace.final_result_source = "exact_anchor"
                    return PipelineResponse(
                        source="exact_anchor",
                        payload=anchor_entry.response_payload,
                        trace=trace,
                    )
                trace.final_result_source = "miss"
                return PipelineResponse(source="miss", payload={}, trace=trace)

            trace.final_result_source = "miss"
            return PipelineResponse(source="miss", payload={}, trace=trace)

        if routing.label is RoutingLabel.THREAD_SCOPED_ONLY and not context.thread_scope_key:
            trace.final_result_source = "miss"
            return PipelineResponse(source="miss", payload={}, trace=trace)

        required_thread_scope = context.thread_scope_key if routing.label is RoutingLabel.THREAD_SCOPED_ONLY else None
        semantic_hit = self._semantic_lookup(
            query, normalized, sq, context, trace, required_thread_scope=required_thread_scope
        )
        if semantic_hit is not None:
            trace.final_result_source = "semantic_cache"
            return PipelineResponse(source="semantic_cache", payload=semantic_hit.response_payload, trace=trace)

        trace.final_result_source = "miss"
        return PipelineResponse(source="miss", payload={}, trace=trace)

    def _semantic_lookup(
        self,
        query: str,
        normalized_query: str,
        sq: StructuredQuery,
        context: RequestContext,
        trace: RequestTrace,
        *,
        required_thread_scope: str | None,
    ) -> SemanticCacheEntry | None:
        trace.semantic_lookup_attempted = True
        embed_start = time.perf_counter()
        embedding = self.embedder(normalized_query)
        trace.embedding_latency_ms = round((time.perf_counter() - embed_start) * 1000, 3)

        search_start = time.perf_counter()
        lookup = self.semantic_store.semantic_lookup(
            embedding=embedding,
            context=context,
            required_thread_scope=required_thread_scope,
            top_k=5,
        )
        trace.ann_latency_ms = round((time.perf_counter() - search_start) * 1000, 3)
        trace.semantic_neighbor_filter_counts = dict(lookup.neighbor_filter_counts)
        trace.top_candidate_similarity = lookup.similarity
        trace.second_candidate_similarity = lookup.second_best_similarity
        trace.candidate_count = lookup.candidate_count

        if lookup.hit is None:
            trace.rejected_reasons = list(lookup.rejected_reasons)
            return None

        if reason := structured_reuse_gate(sq, lookup.hit):
            trace.semantic_post_ann_reject_reason = reason
            trace.rejected_reasons = [reason]
            return None

        if reason := facet_conflict_reason(query, lookup.hit):
            trace.semantic_facet_conflict_reason = reason
            trace.semantic_post_ann_reject_reason = reason
            trace.rejected_reasons = [reason]
            return None

        risk_reason = constraint_risk_reason(query, lookup.hit, sq)
        if risk_reason:
            trace.semantic_constraint_risk_reason = risk_reason
            if self.neighbor_judge is None:
                trace.semantic_post_ann_reject_reason = "constraint_risk_requires_judge"
                trace.rejected_reasons = ["constraint_risk_requires_judge"]
                return None
            max_calls = context.neighbor_judge_max_calls
            if max_calls is not None and trace.neighbor_judge_calls_used >= max_calls:
                trace.semantic_post_ann_reject_reason = "constraint_risk_judge_budget_exceeded"
                trace.rejected_reasons = ["constraint_risk_judge_budget_exceeded"]
                return None
            trace.neighbor_judge_invoked = True
            trace.neighbor_judge_calls_used += 1
            if reason := _run_neighbor_judge(
                self.neighbor_judge,
                query=query,
                hit=lookup.hit,
                context=context,
                trace=trace,
            ):
                trace.semantic_post_ann_reject_reason = reason
                trace.rejected_reasons = [reason]
                return None
            return lookup.hit

        sim = lookup.similarity
        assert sim is not None
        if sim < context.semantic_threshold:
            trace.semantic_post_ann_reject_reason = "semantic_gray_zone_requires_judge"
            if self.neighbor_judge is None:
                trace.rejected_reasons = ["semantic_gray_zone_requires_judge"]
                return None
            max_calls = context.neighbor_judge_max_calls
            if max_calls is not None and trace.neighbor_judge_calls_used >= max_calls:
                trace.semantic_post_ann_reject_reason = "semantic_gray_zone_judge_budget_exceeded"
                trace.rejected_reasons = ["semantic_gray_zone_judge_budget_exceeded"]
                return None
            trace.neighbor_judge_invoked = True
            trace.neighbor_judge_calls_used += 1
            if judge_reason := _run_neighbor_judge(
                self.neighbor_judge,
                query=query,
                hit=lookup.hit,
                context=context,
                trace=trace,
            ):
                trace.semantic_post_ann_reject_reason = judge_reason
                trace.rejected_reasons = [judge_reason]
                return None
            trace.semantic_post_ann_reject_reason = None
            return lookup.hit

        if self.neighbor_judge is not None:
            skip, skip_reason = _neighbor_judge_preflight(
                context=context,
                top_similarity=sim,
                second_similarity=lookup.second_best_similarity,
                calls_used=trace.neighbor_judge_calls_used,
            )
            if skip:
                trace.neighbor_judge_skipped_reason = skip_reason
                return lookup.hit

            trace.neighbor_judge_invoked = True
            trace.neighbor_judge_calls_used += 1
            if reason := _run_neighbor_judge(
                self.neighbor_judge,
                query=query,
                hit=lookup.hit,
                context=context,
                trace=trace,
            ):
                trace.semantic_post_ann_reject_reason = reason
                trace.rejected_reasons = [reason]
                return None

        return lookup.hit

    def _live_and_maybe_insert(
        self,
        query: str,
        normalized: str,
        sq: StructuredQuery,
        context: RequestContext,
        route_label: RoutingLabel,
        live_answerer: LiveAnswerer,
        trace: RequestTrace,
    ) -> PipelineResponse:
        payload = live_answerer(query, context)
        inserted = should_insert_response(
            route_label=route_label,
            context=context,
            success=bool(payload.get("success", True)),
            is_private=bool(payload.get("private", False)),
            is_destructive=bool(payload.get("destructive", False)),
            is_freshness_sensitive=bool(payload.get("freshness_sensitive", False)),
        )
        if inserted:
            exact_key = build_exact_key(normalized, context)
            self.exact_cache.put(exact_key, ExactCacheValue(payload=payload))
            anchor_key = build_anchor_key(query, context) if route_label is RoutingLabel.EXACT_ONLY else None
            if route_label is not RoutingLabel.EXACT_ONLY:
                ns = context.effective_cache_namespace()
                sig = compute_structured_critical_signature(sq)
                id_pairs = identifier_pairs_for_metadata(sq)
                semantic_entry = SemanticCacheEntry(
                    cache_id=hashlib.sha256(
                        f"{ns}|{context.reuse_scope_key}|{context.thread_scope_key}|{normalized}".encode("utf-8")
                    ).hexdigest(),
                    namespace=ns,
                    query_text_original=query,
                    query_text_normalized=normalized,
                    embedding_vector=self.embedder(normalized),
                    response_payload=payload,
                    response_preview=str(payload.get("answer", ""))[:140],
                    created_at=datetime.now(tz=UTC),
                    expires_at=datetime.now(tz=UTC) + timedelta(hours=1),
                    cache_policy_class=cache_policy_for_route(route_label),
                    agent_version=context.agent_version,
                    corpus_version=context.corpus_version,
                    tool_or_schema_version=context.tool_or_schema_version,
                    thread_scope_key=context.thread_scope_key,
                    exact_anchor_key=anchor_key,
                    freshness_class="stable",
                    reuse_scope_key=context.reuse_scope_key,
                    structured_critical_signature=sig,
                    structured_confidence_at_insert=sq.confidence,
                    confidence_metadata={
                        "route_confidence": trace.routing_confidence,
                        "structured_identifier_pairs": id_pairs,
                        "query_facets": extract_query_facets(query),
                    },
                )
                self.semantic_store.insert(semantic_entry)
            elif anchor_key:
                ns = context.effective_cache_namespace()
                semantic_entry = SemanticCacheEntry(
                    cache_id=hashlib.sha256(f"anchor|{anchor_key}".encode("utf-8")).hexdigest(),
                    namespace=ns,
                    query_text_original=query,
                    query_text_normalized=normalized,
                    embedding_vector=[0.0] * self.semantic_store._index.dimension,
                    response_payload=payload,
                    response_preview=str(payload.get("answer", ""))[:140],
                    created_at=datetime.now(tz=UTC),
                    expires_at=datetime.now(tz=UTC) + timedelta(hours=1),
                    cache_policy_class="exact_only",
                    agent_version=context.agent_version,
                    corpus_version=context.corpus_version,
                    tool_or_schema_version=context.tool_or_schema_version,
                    thread_scope_key=context.thread_scope_key,
                    exact_anchor_key=anchor_key,
                    freshness_class="stable",
                    reuse_scope_key=context.reuse_scope_key,
                    structured_critical_signature=compute_structured_critical_signature(sq),
                    structured_confidence_at_insert=sq.confidence,
                    confidence_metadata={
                        "route_confidence": trace.routing_confidence,
                        "structured_identifier_pairs": identifier_pairs_for_metadata(sq),
                        "query_facets": extract_query_facets(query),
                    },
                )
                self.semantic_store.insert(semantic_entry)
            trace.insert_performed = True

        trace.final_result_source = "live"
        return PipelineResponse(source="live", payload=payload, trace=trace)

    def _populate_structured_trace(self, sq: StructuredQuery, context: RequestContext, trace: RequestTrace) -> None:
        trace.structured_extraction_attempted = True
        trace.structured_confidence = sq.confidence
        trace.structured_ambiguity_flags = list(sq.ambiguity_flags)
        trace.structured_constraint_kinds = sorted({c.kind for c in sq.constraints})
        crit = sq.critical_constraints()
        trace.structured_critical_constraint_count = len(crit)
        preview: list[str] = []
        for c in crit:
            if c.confidence < 0.85:
                continue
            preview.append(canonicalize_constraint(c))
            if len(preview) >= 16:
                break
        trace.structured_critical_preview = preview
        ckey = build_structured_key(sq, namespace=context.effective_cache_namespace())
        if ckey is not None:
            trace.structured_canonical_key_sha256 = hashlib.sha256(ckey.encode("utf-8")).hexdigest()
