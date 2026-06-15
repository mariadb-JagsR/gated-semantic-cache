from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from gated_semantic_cache.cache.faiss_index import FaissHnswIndex
from gated_semantic_cache.models.cache_entry import SemanticCacheEntry
from gated_semantic_cache.models.context import RequestContext
from gated_semantic_cache.serving.policy import is_policy_compatible


@dataclass(slots=True)
class SemanticLookupResult:
    """Outcome of ANN + filter scan. ``hit`` is set when a candidate passes all filters."""

    hit: SemanticCacheEntry | None
    similarity: float | None
    candidate_count: int
    # Populated when ``hit`` is None: ordered reasons for filtered candidates (debug).
    rejected_reasons: list[str]
    # Counts per filter reason for all scanned neighbors (non-misleading when ``hit`` is set).
    neighbor_filter_counts: dict[str, int]
    # Second-highest similarity among eligible (policy-passing) neighbors, if any.
    second_best_similarity: float | None = None


class SemanticStore:
    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._entries: dict[str, SemanticCacheEntry] = {}
        self._index = FaissHnswIndex(dimension=dimension)
        self._anchor_map: dict[str, str] = {}

    @property
    def dimension(self) -> int:
        return self._dimension

    def clear(self) -> None:
        self._entries.clear()
        self._anchor_map.clear()
        self._index = FaissHnswIndex(dimension=self._dimension)

    def replace_hydrated(
        self,
        *,
        entries: dict[str, SemanticCacheEntry],
        anchor_map: dict[str, str],
        index: FaissHnswIndex,
    ) -> None:
        """Replace in-memory state after loading from a snapshot (internal / persistence)."""

        self._entries = dict(entries)
        self._anchor_map = dict(anchor_map)
        self._index = index

    def insert(self, entry: SemanticCacheEntry) -> None:
        self._entries[entry.cache_id] = entry
        if entry.exact_anchor_key:
            self._anchor_map[entry.exact_anchor_key] = entry.cache_id
        if entry.cache_policy_class != "exact_only":
            self._index.add(entry.cache_id, entry.embedding_vector)

    def get_anchor(self, anchor_key: str) -> SemanticCacheEntry | None:
        cache_id = self._anchor_map.get(anchor_key)
        if cache_id is None:
            return None
        return self._entries.get(cache_id)

    def semantic_lookup(
        self,
        *,
        embedding: list[float],
        context: RequestContext,
        required_thread_scope: str | None,
        top_k: int = 5,
    ) -> SemanticLookupResult:
        rejected_reasons: list[str] = []
        counts: Counter[str] = Counter()
        eligible: list[tuple[SemanticCacheEntry, float]] = []
        candidates = self._index.search(embedding, top_k=top_k)
        top_candidate_similarity = candidates[0].similarity if candidates else None

        for candidate in candidates:
            entry = self._entries.get(candidate.cache_id)
            if entry is None:
                reason = "missing_entry"
                counts[reason] += 1
                rejected_reasons.append(reason)
                continue
            reason = _reject_reason(entry, context, required_thread_scope, candidate.similarity)
            if reason:
                counts[reason] += 1
                rejected_reasons.append(reason)
                continue
            eligible.append((entry, candidate.similarity))

        if not eligible:
            return SemanticLookupResult(
                hit=None,
                similarity=top_candidate_similarity,
                candidate_count=len(candidates),
                rejected_reasons=rejected_reasons,
                neighbor_filter_counts=dict(counts),
                second_best_similarity=None,
            )

        eligible.sort(key=lambda item: item[1], reverse=True)
        best_entry, best_similarity = eligible[0]
        second_best: float | None = None
        if len(eligible) > 1:
            second_best = eligible[1][1]
        # Successful hit: do not surface sub-threshold rejections from other neighbors as if they
        # applied to the winning candidate (observability consistency with ``final_result_source``).
        return SemanticLookupResult(
            hit=best_entry,
            similarity=best_similarity,
            candidate_count=len(candidates),
            rejected_reasons=[],
            neighbor_filter_counts=dict(counts),
            second_best_similarity=second_best,
        )


def _reject_reason(
    entry: SemanticCacheEntry,
    context: RequestContext,
    required_thread_scope: str | None,
    similarity: float,
) -> str | None:
    if entry.namespace != context.effective_cache_namespace():
        return "namespace_mismatch"
    if entry.is_expired():
        return "expired"
    if entry.agent_version != context.agent_version:
        return "agent_version_mismatch"
    if context.corpus_version and entry.corpus_version != context.corpus_version:
        return "corpus_version_mismatch"
    if context.tool_or_schema_version and entry.tool_or_schema_version != context.tool_or_schema_version:
        return "tool_or_schema_version_mismatch"
    if required_thread_scope is not None and entry.thread_scope_key != required_thread_scope:
        return "thread_scope_mismatch"
    if not is_policy_compatible(entry.cache_policy_class, context):
        return "cache_policy_mismatch"
    if context.reuse_scope_key is not None or entry.reuse_scope_key is not None:
        if context.reuse_scope_key != entry.reuse_scope_key:
            return "reuse_scope_mismatch"
    candidate_floor = min(context.semantic_low_watermark, context.semantic_threshold)
    if similarity < candidate_floor:
        return "below_threshold"
    return None
