"""Pluggable persistence contracts.

Local development uses SQLite plus an optional on-disk FAISS snapshot. Production
(GridGain, managed vector search, etc.) should implement the same protocols without
changing routing or gate logic in ``serving/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from gatecache.models.cache_entry import SemanticCacheEntry


@dataclass(frozen=True, slots=True)
class NamespaceHydrationBundle:
    """Cold-start data for one namespace loaded from durable storage."""

    namespace: str
    exact_payloads: dict[str, dict[str, Any]]  # exact_key_sha256 -> payload dict
    semantic_entries: dict[str, SemanticCacheEntry]  # cache_id -> entry
    anchor_map: dict[str, str]  # anchor_key_sha256 -> cache_id
    embedding_dimension: int


@runtime_checkable
class CacheEntryPersistence(Protocol):
    """Exact + semantic rows (replace with GridGain / JDBC / HTTP in GA)."""

    def init_schema(self) -> None: ...

    def upsert_exact(
        self,
        *,
        namespace: str,
        exact_key_sha256: str,
        scope_fingerprint: str | None,
        payload: dict[str, Any],
    ) -> None: ...

    def upsert_semantic(self, entry: SemanticCacheEntry) -> None: ...

    def upsert_anchor(self, *, namespace: str, anchor_key_sha256: str, cache_id: str) -> None: ...

    def load_namespace(self, namespace: str, *, embedding_dimension: int) -> NamespaceHydrationBundle: ...

    def clear_namespace(self, namespace: str) -> None: ...

    def stats_global(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


@runtime_checkable
class VectorIndexPersistence(Protocol):
    """Optional fast-path snapshot for the ANN index (FAISS today; remote ANN later)."""

    def save(self, *, namespace: str, faiss_index: Any, cache_ids: list[str]) -> None: ...

    def load_or_none(self, *, namespace: str, embedding_dimension: int) -> tuple[Any, list[str]] | None: ...

    def remove(self, *, namespace: str) -> None: ...
