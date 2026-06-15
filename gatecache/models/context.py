from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_SEMANTIC_LOW_WATERMARK = 0.70
DEFAULT_EXACT_ONLY_MIN_ROUTE_CONFIDENCE = 0.55
# Bounded thread context forwarded to post-retrieval judge (caller-maintained).
DEFAULT_MAX_PRIOR_USER_QUERIES = 3


@dataclass(slots=True)
class RequestContext:
    """Request-scoped knobs for cache lookup and insertion.

    ``namespace`` is the primary logical tenant/cache partition. ``cache_namespace`` optionally
    overrides it for cache keying and ANN isolation without changing other request metadata.
    ``reuse_scope_key`` further bounds semantic/exact reuse (e.g. agent or product surface).
    """

    namespace: str = "default"
    agent_version: str = "v1"
    corpus_version: str | None = None
    tool_or_schema_version: str | None = None
    thread_scope_key: str | None = None
    # Optional recent user queries in the same thread, oldest first. Callers (e.g. REPL) may populate this
    # for bounded gray-zone judge context; the engine does not reconstruct conversation history itself.
    prior_user_queries: tuple[str, ...] = ()
    # High-confidence cosine threshold: hits at or above this may be served after deterministic gates.
    semantic_threshold: float = 0.86
    # Low watermark for candidate admission. Candidates between this and semantic_threshold enter the
    # gray-zone path and require a bounded neighbor judge; below this they are filtered before gates.
    semantic_low_watermark: float = DEFAULT_SEMANTIC_LOW_WATERMARK
    # When set: after gray-zone handling, neighbor judge (if configured) only runs for similarities in
    # [semantic_threshold, neighbor_judge_similarity_ceiling). At or above the ceiling, the hit is treated
    # as a strong embedding match and the judge is skipped. None disables this band.
    neighbor_judge_similarity_ceiling: float | None = None
    # Max neighbor-judge invocations per request; when exceeded, remaining hits skip the judge and reuse (cheap path).
    neighbor_judge_max_calls: int | None = None
    # When set: skip the judge if (top1_sim - runner_up_sim) >= this margin (clear winner among eligible neighbors).
    # When None, margin-based skip is disabled.
    neighbor_judge_ambiguity_margin: float | None = None
    # When set: if the router returns SEMANTIC_OK but confidence is below this floor, treat as SKIP_CACHE
    # (no semantic/exact/thread reuse) for paraphrase-consistency / calibration experiments.
    semantic_ok_min_route_confidence: float | None = None
    # When EXACT_ONLY is predicted without a resolvable anchor, downgrade to SEMANTIC_OK if router
    # confidence is below this floor. Set None to keep the conservative live-only path.
    exact_only_min_route_confidence: float | None = DEFAULT_EXACT_ONLY_MIN_ROUTE_CONFIDENCE
    exact_context: dict[str, str] = field(default_factory=dict)
    cache_namespace: str | None = None
    reuse_scope_key: str | None = None

    def effective_cache_namespace(self) -> str:
        return self.cache_namespace if self.cache_namespace is not None else self.namespace

    def exact_key_parts(self) -> tuple[tuple[str, str], ...]:
        base = sorted((key, value) for key, value in self.exact_context.items() if value)
        if self.reuse_scope_key:
            return tuple(sorted((*base, ("reuse_scope_key", self.reuse_scope_key))))
        return tuple(base)
