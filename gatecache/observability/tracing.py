from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class RequestTrace:
    normalized_query: str
    exact_cache_attempted: bool = False
    exact_cache_hit: bool = False
    # SHA-256 hex from `build_exact_key` (string exact-cache lookup).
    exact_cache_key_sha256: str | None = None
    routing_label: str | None = None
    routing_confidence: float | None = None
    routing_semantic_ok_downgraded: bool = False
    routing_exact_only_downgraded: bool = False
    semantic_lookup_attempted: bool = False
    embedding_latency_ms: float = 0.0
    ann_latency_ms: float = 0.0
    top_candidate_similarity: float | None = None
    candidate_count: int = 0
    rejected_reasons: list[str] = field(default_factory=list)
    # Per-reason counts for ANN neighbors (includes below_threshold for non-winning neighbors when a hit exists).
    semantic_neighbor_filter_counts: dict[str, int] = field(default_factory=dict)
    final_result_source: str | None = None
    insert_performed: bool = False
    # Structured exact (deterministic extraction + canonical key); runs on every miss after exact-cache.
    structured_extraction_attempted: bool = False
    structured_confidence: float | None = None
    # SHA-256 hex of structured canonical key string when gates pass.
    structured_canonical_key_sha256: str | None = None
    structured_constraint_kinds: list[str] = field(default_factory=list)
    structured_ambiguity_flags: list[str] = field(default_factory=list)
    structured_critical_constraint_count: int = 0
    # Short canonical constraint strings (high-confidence critical only; capped in pipeline).
    structured_critical_preview: list[str] = field(default_factory=list)
    # Anchor key from `build_anchor_key` when EXACT_ONLY path resolves an identifier.
    anchor_lookup_key_sha256: str | None = None
    # Post-ANN structured gate or pluggable neighbor judge (machine-readable).
    semantic_post_ann_reject_reason: str | None = None
    semantic_facet_conflict_reason: str | None = None
    semantic_constraint_risk_reason: str | None = None
    # Second-best eligible cosine similarity (for ambiguity / margin observability).
    second_candidate_similarity: float | None = None
    neighbor_judge_invoked: bool = False
    neighbor_judge_skipped_reason: str | None = None
    neighbor_judge_calls_used: int = 0
    # Populated when SEMANTIC_CACHE_JUDGE_DEBUG=1 and an LLM-backed judge runs.
    neighbor_judge_raw_response: str | None = None
    neighbor_judge_decision: dict[str, Any] | None = None
    neighbor_judge_response_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
