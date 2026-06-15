from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


@dataclass(slots=True)
class SemanticCacheEntry:
    cache_id: str
    namespace: str
    query_text_original: str
    query_text_normalized: str
    embedding_vector: list[float]
    response_payload: dict[str, Any]
    response_preview: str
    created_at: datetime
    expires_at: datetime | None
    cache_policy_class: str
    agent_version: str
    corpus_version: str | None
    tool_or_schema_version: str | None
    thread_scope_key: str | None
    exact_anchor_key: str | None
    freshness_class: str
    reuse_scope_key: str | None = None
    structured_critical_signature: str | None = None
    structured_confidence_at_insert: float | None = None
    validation_status: str = "unverified"
    source_type: str = "live"
    confidence_metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self, now: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        current = now or utc_now()
        return current >= self.expires_at
