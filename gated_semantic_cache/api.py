from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from collections.abc import Callable
from typing import Any, Literal, Mapping

from gated_semantic_cache.cache.exact_cache import ExactCache, ExactCacheValue
from gated_semantic_cache.cache.semantic_store import SemanticStore
from gated_semantic_cache.cache.sqlite_persistence import (
    FaissVectorPersistence,
    SqliteCachePersistence,
    hydrate_semantic_store,
)
from gated_semantic_cache.embeddings.backends import (
    caching_embedder,
    embedding_dim_for_openai_model,
    make_openai_embedder,
)
from gated_semantic_cache.eval.datasets import build_routing_dataset
from gated_semantic_cache.routing.classifier import train_default_classifier
from gated_semantic_cache.models.cache_entry import SemanticCacheEntry
from gated_semantic_cache.models.context import DEFAULT_SEMANTIC_LOW_WATERMARK, RequestContext
from gated_semantic_cache.observability.tracing import RequestTrace
from gated_semantic_cache.routing.features import normalize_query_text
from gated_semantic_cache.routing.labels import RoutingLabel
from gated_semantic_cache.serving.llm_judge import default_llm_neighbor_judge_from_env
from gated_semantic_cache.serving.neighbor_judge import NeighborJudge
from gated_semantic_cache.serving.pipeline import Embedder, SemanticCachePipeline
from gated_semantic_cache.serving.policy import build_anchor_key, build_exact_key, cache_policy_for_route
from gated_semantic_cache.serving.query_facets import extract_query_facets
from gated_semantic_cache.serving.structured_reuse_gate import (
    compute_structured_critical_signature,
    identifier_pairs_for_metadata,
)
from gated_semantic_cache.structured_exact.structured_query import extract_structured_query

SemanticMode = Literal["auto", "always", "never"]


@dataclass(frozen=True, slots=True)
class JudgePolicy:
    """Post-retrieval verifier policy for semantic hits.

    ``enabled=True`` is intentionally conservative: if no judge callable is configured,
    gray-zone semantic hits fail closed instead of being reused silently.
    """

    enabled: bool = True
    similarity_floor: float = 0.70
    similarity_ceiling: float | None = None
    ambiguity_margin: float | None = None
    max_calls: int | None = 1
    fail_closed_on_missing_judge: bool = True


@dataclass(frozen=True, slots=True)
class CacheHit:
    source: str
    payload: dict[str, Any]
    trace: dict[str, Any]
    similarity: float | None = None


@dataclass(frozen=True, slots=True)
class CacheEntryRef:
    namespace: str
    exact_key_sha256: str
    semantic_cache_id: str | None
    anchor_key_sha256: str | None
    semantic_indexed: bool
    routing_label: str
    routing_confidence: float
    trace: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PutPolicy:
    """Controls how explicit ``put`` calls index a response.

    Exact lookup storage always happens on ``put``. ``semantic_mode`` only controls ANN
    indexing. In ``auto`` mode, the classifier may narrow indexing or skip semantic indexing.
    """

    semantic_mode: SemanticMode = "always"
    ttl_seconds: int | None = 3600
    metadata: Mapping[str, Any] = field(default_factory=dict)


class SemanticCache:
    """Public get/put API for app-level semantic response caching.

    The app owns live answer generation. This class only performs cache lookup and explicit
    insertion, with namespace isolation and optional caller-supplied scope keys.
    """

    def __init__(
        self,
        *,
        namespace: str,
        pipeline: SemanticCachePipeline,
        neighbor_judge: NeighborJudge | None = None,
        default_judge_policy: JudgePolicy | None = None,
        use_default_llm_judge: bool = True,
        entry_persistence: SqliteCachePersistence | None = None,
        vector_persistence: FaissVectorPersistence | None = None,
    ) -> None:
        if not namespace.strip():
            raise ValueError("namespace must be non-empty")
        self.namespace = namespace
        self._pipeline = pipeline
        self._neighbor_judge = neighbor_judge
        if self._neighbor_judge is None and use_default_llm_judge:
            self._neighbor_judge = default_llm_neighbor_judge_from_env()
        self.default_judge_policy = default_judge_policy or JudgePolicy()
        self._entry_persistence = entry_persistence
        self._vector_persistence = vector_persistence

    def close(self) -> None:
        if self._entry_persistence is not None:
            self._entry_persistence.close()

    @classmethod
    def from_sqlite(
        cls,
        *,
        db_path: Path | str,
        namespace: str,
        openai_model: str = "text-embedding-3-small",
        openai_api_key: str | None = None,
        openai_dimensions: int | None = None,
        embed_cache: bool = False,
        embedder: Callable[[str], list[float]] | None = None,
        embedding_dimension: int | None = None,
        neighbor_judge: NeighborJudge | None = None,
        default_judge_policy: JudgePolicy | None = None,
        use_default_llm_judge: bool = True,
        force_rebuild_index: bool = False,
    ) -> SemanticCache:
        """Open a durable SQLite cache and hydrate exact + semantic stores for ``namespace``."""

        db_path = Path(db_path)
        persistence = SqliteCachePersistence(db_path)
        persistence.init_schema()
        if embedder is None:
            dim = embedding_dim_for_openai_model(openai_model, dimensions=openai_dimensions)
            resolved_embedder = make_openai_embedder(
                model=openai_model,
                api_key=openai_api_key,
                dimensions=openai_dimensions,
            )
        else:
            dim = (
                embedding_dimension
                if embedding_dimension is not None
                else len(embedder("__semantic_cache_embedding_dim_probe__"))
            )
            resolved_embedder = embedder
        if embed_cache:
            resolved_embedder = caching_embedder(resolved_embedder)
        router = train_default_classifier(build_routing_dataset())
        exact_cache = ExactCache()
        semantic_store = SemanticStore(dimension=dim)
        bundle = persistence.load_namespace(namespace, embedding_dimension=dim)
        exact_map = {k: ExactCacheValue(payload=v) for k, v in bundle.exact_payloads.items()}
        exact_cache.replace_all(exact_map)
        vp = FaissVectorPersistence(db_path)
        hydrate_semantic_store(
            bundle=bundle,
            semantic_store=semantic_store,
            vector_persistence=vp,
            force_rebuild_index=force_rebuild_index,
        )
        pipeline = SemanticCachePipeline(
            router=router,
            exact_cache=exact_cache,
            semantic_store=semantic_store,
            embedder=resolved_embedder,
            neighbor_judge=None,
        )
        return cls(
            namespace=namespace,
            pipeline=pipeline,
            neighbor_judge=neighbor_judge,
            default_judge_policy=default_judge_policy,
            use_default_llm_judge=use_default_llm_judge,
            entry_persistence=persistence,
            vector_persistence=vp,
        )

    @classmethod
    def from_components(
        cls,
        *,
        namespace: str,
        router: Any,
        exact_cache: ExactCache,
        semantic_store: SemanticStore,
        embedder: Embedder,
        neighbor_judge: NeighborJudge | None = None,
        default_judge_policy: JudgePolicy | None = None,
        use_default_llm_judge: bool = True,
    ) -> SemanticCache:
        pipeline = SemanticCachePipeline(
            router=router,
            exact_cache=exact_cache,
            semantic_store=semantic_store,
            embedder=embedder,
            neighbor_judge=None,
        )
        return cls(
            namespace=namespace,
            pipeline=pipeline,
            neighbor_judge=neighbor_judge,
            default_judge_policy=default_judge_policy,
            use_default_llm_judge=use_default_llm_judge,
            entry_persistence=None,
            vector_persistence=None,
        )

    def get(
        self,
        query: str,
        *,
        namespace: str | None = None,
        scope_keys: Mapping[str, str] | None = None,
        semantic_mode: SemanticMode = "always",
        judge_policy: JudgePolicy | None = None,
        semantic_threshold: float | None = None,
        semantic_low_watermark: float | None = None,
        thread_scope_key: str | None = None,
    ) -> CacheHit | None:
        if semantic_mode not in ("auto", "always", "never"):
            raise ValueError("semantic_mode must be one of: auto, always, never")
        policy = judge_policy or self.default_judge_policy
        context = self._context(
            namespace=namespace,
            scope_keys=scope_keys,
            judge_policy=policy,
            semantic_threshold=semantic_threshold,
            semantic_low_watermark=semantic_low_watermark,
            thread_scope_key=thread_scope_key,
        )

        old_judge = self._pipeline.neighbor_judge
        self._pipeline.neighbor_judge = self._active_judge(policy)
        try:
            if semantic_mode == "auto":
                response = self._pipeline.lookup_query(query, context)
            else:
                response = self._lookup_with_forced_semantic_mode(query, context, semantic_mode)
        finally:
            self._pipeline.neighbor_judge = old_judge

        if response.source == "miss":
            return None
        return CacheHit(
            source=response.source,
            payload=response.payload,
            trace=response.trace.to_dict(),
            similarity=response.trace.top_candidate_similarity,
        )

    def put(
        self,
        query: str,
        response: Mapping[str, Any],
        *,
        namespace: str | None = None,
        scope_keys: Mapping[str, str] | None = None,
        policy: PutPolicy | None = None,
        thread_scope_key: str | None = None,
    ) -> CacheEntryRef:
        put_policy = policy or PutPolicy()
        if put_policy.semantic_mode not in ("auto", "always", "never"):
            raise ValueError("semantic_mode must be one of: auto, always, never")
        context = self._context(
            namespace=namespace,
            scope_keys=scope_keys,
            judge_policy=self.default_judge_policy,
            semantic_threshold=self.default_judge_policy.similarity_floor,
            thread_scope_key=thread_scope_key,
        )
        payload = dict(response)
        normalized = normalize_query_text(query)
        trace = RequestTrace(normalized_query=normalized, exact_cache_attempted=False)
        sq = extract_structured_query(query)
        self._pipeline._populate_structured_trace(sq, context, trace)

        routing = self._pipeline.router.predict(query)
        route_label = routing.label
        trace.routing_label = route_label.value
        trace.routing_confidence = routing.confidence

        exact_key = build_exact_key(normalized, context)
        self._pipeline.exact_cache.put(exact_key, ExactCacheValue(payload=payload))
        trace.exact_cache_key_sha256 = exact_key
        trace.insert_performed = True

        semantic_cache_id: str | None = None
        anchor_key = build_anchor_key(query, context) if route_label is RoutingLabel.EXACT_ONLY else None
        if anchor_key is not None:
            trace.anchor_lookup_key_sha256 = anchor_key

        semantic_route = self._semantic_route_for_put(route_label, put_policy.semantic_mode, anchor_key, context)
        entry: SemanticCacheEntry | None = None
        if semantic_route is not None:
            entry = self._build_entry(
                query=query,
                normalized=normalized,
                payload=payload,
                context=context,
                route_label=semantic_route,
                sq=sq,
                trace=trace,
                ttl_seconds=put_policy.ttl_seconds,
                anchor_key=anchor_key if semantic_route is RoutingLabel.EXACT_ONLY else None,
                metadata=put_policy.metadata,
            )
            self._pipeline.semantic_store.insert(entry)
            semantic_cache_id = entry.cache_id

        self._persist_after_put(
            context=context,
            exact_key_sha256=exact_key,
            payload=payload,
            entry=entry,
        )

        return CacheEntryRef(
            namespace=context.effective_cache_namespace(),
            exact_key_sha256=exact_key,
            semantic_cache_id=semantic_cache_id,
            anchor_key_sha256=anchor_key,
            semantic_indexed=semantic_cache_id is not None,
            routing_label=route_label.value,
            routing_confidence=routing.confidence,
            trace=trace.to_dict(),
        )

    def _persist_after_put(
        self,
        *,
        context: RequestContext,
        exact_key_sha256: str,
        payload: dict[str, Any],
        entry: SemanticCacheEntry | None,
    ) -> None:
        if self._entry_persistence is None:
            return
        ns = context.effective_cache_namespace()
        self._entry_persistence.upsert_exact(
            namespace=ns,
            exact_key_sha256=exact_key_sha256,
            scope_fingerprint=_scope_fingerprint_from_exact_context(context.exact_context),
            payload=payload,
        )
        if entry is not None:
            self._entry_persistence.upsert_semantic(entry)
        if entry is not None and entry.exact_anchor_key:
            self._entry_persistence.upsert_anchor(
                namespace=ns,
                anchor_key_sha256=entry.exact_anchor_key,
                cache_id=entry.cache_id,
            )
        if self._vector_persistence is not None:
            self._vector_persistence.save(
                namespace=ns,
                faiss_index=self._pipeline.semantic_store._index,
                cache_ids=self._pipeline.semantic_store._index.cache_ids,
            )

    def _lookup_with_forced_semantic_mode(
        self,
        query: str,
        context: RequestContext,
        semantic_mode: SemanticMode,
    ) -> Any:
        normalized = normalize_query_text(query)
        trace = RequestTrace(normalized_query=normalized, exact_cache_attempted=True)
        exact_key = build_exact_key(normalized, context)
        trace.exact_cache_key_sha256 = exact_key
        exact_hit = self._pipeline.exact_cache.get(exact_key)
        if exact_hit is not None:
            trace.exact_cache_hit = True
            trace.final_result_source = exact_hit.source
            from gated_semantic_cache.serving.pipeline import PipelineResponse

            return PipelineResponse(source=exact_hit.source, payload=exact_hit.payload, trace=trace)
        if semantic_mode == "never":
            trace.final_result_source = "miss"
            from gated_semantic_cache.serving.pipeline import PipelineResponse

            return PipelineResponse(source="miss", payload={}, trace=trace)

        sq = extract_structured_query(query)
        self._pipeline._populate_structured_trace(sq, context, trace)
        trace.routing_label = "FORCED_SEMANTIC"
        trace.routing_confidence = 1.0
        semantic_hit = self._pipeline._semantic_lookup(
            query,
            normalized,
            sq,
            context,
            trace,
            required_thread_scope=None,
        )
        from gated_semantic_cache.serving.pipeline import PipelineResponse

        if semantic_hit is None:
            trace.final_result_source = "miss"
            return PipelineResponse(source="miss", payload={}, trace=trace)
        trace.final_result_source = "semantic_cache"
        return PipelineResponse(source="semantic_cache", payload=semantic_hit.response_payload, trace=trace)

    def _context(
        self,
        *,
        namespace: str | None,
        scope_keys: Mapping[str, str] | None,
        judge_policy: JudgePolicy,
        semantic_threshold: float | None,
        semantic_low_watermark: float | None = None,
        thread_scope_key: str | None = None,
    ) -> RequestContext:
        effective_namespace = namespace or self.namespace
        if not effective_namespace.strip():
            raise ValueError("namespace must be non-empty")
        exact_context = dict(scope_keys or {})
        scope_fingerprint = _scope_fingerprint(scope_keys)
        return RequestContext(
            namespace=effective_namespace,
            cache_namespace=effective_namespace,
            exact_context=exact_context,
            reuse_scope_key=scope_fingerprint,
            thread_scope_key=thread_scope_key if thread_scope_key is not None else scope_fingerprint,
            semantic_threshold=semantic_threshold
            if semantic_threshold is not None
            else judge_policy.similarity_floor,
            semantic_low_watermark=semantic_low_watermark
            if semantic_low_watermark is not None
            else DEFAULT_SEMANTIC_LOW_WATERMARK,
            neighbor_judge_similarity_ceiling=judge_policy.similarity_ceiling,
            neighbor_judge_ambiguity_margin=judge_policy.ambiguity_margin,
            neighbor_judge_max_calls=judge_policy.max_calls,
        )

    def _active_judge(self, policy: JudgePolicy) -> NeighborJudge | None:
        if not policy.enabled:
            return None
        if self._neighbor_judge is not None:
            return self._neighbor_judge
        if not policy.fail_closed_on_missing_judge:
            return None

        def _missing_judge(_query: str, _entry: SemanticCacheEntry, _context: RequestContext) -> str | None:
            return "neighbor_judge_not_configured"

        return _missing_judge

    def _semantic_route_for_put(
        self,
        route_label: RoutingLabel,
        semantic_mode: SemanticMode,
        anchor_key: str | None,
        context: RequestContext,
    ) -> RoutingLabel | None:
        if semantic_mode == "never":
            return None
        if semantic_mode == "always":
            return RoutingLabel.SEMANTIC_OK
        if route_label is RoutingLabel.SKIP_CACHE:
            return None
        if route_label is RoutingLabel.EXACT_ONLY:
            return RoutingLabel.EXACT_ONLY if anchor_key is not None else RoutingLabel.SEMANTIC_OK
        if route_label is RoutingLabel.THREAD_SCOPED_ONLY and context.thread_scope_key is None:
            return None
        return route_label

    def _build_entry(
        self,
        *,
        query: str,
        normalized: str,
        payload: dict[str, Any],
        context: RequestContext,
        route_label: RoutingLabel,
        sq: Any,
        trace: RequestTrace,
        ttl_seconds: int | None,
        anchor_key: str | None,
        metadata: Mapping[str, Any],
    ) -> SemanticCacheEntry:
        ns = context.effective_cache_namespace()
        expires_at = datetime.now(tz=UTC) + timedelta(seconds=ttl_seconds) if ttl_seconds is not None else None
        embedding = (
            [0.0] * self._pipeline.semantic_store._index.dimension
            if route_label is RoutingLabel.EXACT_ONLY
            else self._pipeline.embedder(normalized)
        )
        id_pairs = identifier_pairs_for_metadata(sq)
        confidence_metadata = {
            "route_confidence": trace.routing_confidence,
            "structured_identifier_pairs": id_pairs,
            "query_facets": extract_query_facets(query),
            **dict(metadata),
        }
        cache_id_seed = f"{ns}|{context.reuse_scope_key}|{context.thread_scope_key}|{route_label.value}|{normalized}"
        if anchor_key:
            cache_id_seed = f"anchor|{anchor_key}"
        return SemanticCacheEntry(
            cache_id=hashlib.sha256(cache_id_seed.encode("utf-8")).hexdigest(),
            namespace=ns,
            query_text_original=query,
            query_text_normalized=normalized,
            embedding_vector=embedding,
            response_payload=payload,
            response_preview=str(payload.get("answer", ""))[:140],
            created_at=datetime.now(tz=UTC),
            expires_at=expires_at,
            cache_policy_class=cache_policy_for_route(route_label),
            agent_version=context.agent_version,
            corpus_version=context.corpus_version,
            tool_or_schema_version=context.tool_or_schema_version,
            thread_scope_key=context.thread_scope_key,
            exact_anchor_key=anchor_key,
            freshness_class=str(metadata.get("freshness_class", "stable")),
            reuse_scope_key=context.reuse_scope_key,
            structured_critical_signature=compute_structured_critical_signature(sq),
            structured_confidence_at_insert=sq.confidence,
            confidence_metadata=confidence_metadata,
        )


def _scope_fingerprint(scope_keys: Mapping[str, str] | None) -> str | None:
    if not scope_keys:
        return None
    canonical = json.dumps(dict(scope_keys), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _scope_fingerprint_from_exact_context(exact_context: Mapping[str, str]) -> str | None:
    if not exact_context:
        return None
    return _scope_fingerprint(exact_context)
