from __future__ import annotations

import hashlib

from gated_semantic_cache.models.cache_entry import SemanticCacheEntry
from gated_semantic_cache.structured_exact.canonical_key import canonicalize_constraint
from gated_semantic_cache.structured_exact.schema import StructuredQuery

# Below this, skip constraint-based reuse blocking (extraction too weak).
_LOW_STRUCTURED_CONFIDENCE = 0.55
# At or above: require full critical signature match when the cached entry recorded one.
_HIGH_STRUCTURED_CONFIDENCE = 0.85


def compute_structured_critical_signature(sq: StructuredQuery) -> str | None:
    """SHA-256 of sorted high-confidence critical constraints; None if nothing reliable."""
    parts: list[str] = []
    for c in sq.critical_constraints():
        if c.confidence < _HIGH_STRUCTURED_CONFIDENCE:
            continue
        parts.append(canonicalize_constraint(c))
    if not parts:
        return None
    parts.sort()
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _identifier_pairs(sq: StructuredQuery) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in sq.critical_constraints():
        if c.kind != "identifier" or c.confidence < _HIGH_STRUCTURED_CONFIDENCE:
            continue
        key = f"{c.name}:{c.op or '='}"
        out[key] = str(c.value)
    return out


def structured_reuse_gate(sq: StructuredQuery, entry: SemanticCacheEntry) -> str | None:
    """Return a rejection reason if structured signals forbid reusing ``entry``, else None.

    When ``sq.confidence`` is low, the gate is permissive. When high, mismatches on critical
    constraints reject reuse. Medium band tightens only on strong identifier conflicts.
    """
    if not entry.structured_critical_signature:
        return None
    if sq.confidence < _LOW_STRUCTURED_CONFIDENCE:
        return None

    current_sig = compute_structured_critical_signature(sq)
    if sq.confidence >= _HIGH_STRUCTURED_CONFIDENCE:
        if current_sig is None:
            return None
        if current_sig != entry.structured_critical_signature:
            return "structured_critical_mismatch"
        return None

    # Medium confidence: block only on explicit identifier value conflicts.
    if current_sig == entry.structured_critical_signature:
        return None
    left = _identifier_pairs(sq)
    if not left:
        return None
    cached_meta = entry.confidence_metadata or {}
    cached_ids = cached_meta.get("structured_identifier_pairs")
    if not isinstance(cached_ids, dict):
        return None
    for k, v in left.items():
        other = cached_ids.get(k)
        if other is not None and str(other) != str(v):
            return "structured_identifier_mismatch"
    return None


def identifier_pairs_for_metadata(sq: StructuredQuery) -> dict[str, str]:
    return _identifier_pairs(sq)
