from __future__ import annotations

from gated_semantic_cache.structured_exact.ambiguity import detect_ambiguity
from gated_semantic_cache.structured_exact.confidence import score_structured_confidence
from gated_semantic_cache.structured_exact.extractors import (
    extract_anchors,
    extract_categoricals,
    extract_dates,
    extract_dimensions,
    extract_ids,
    extract_numeric_bounds,
    extract_quantities,
)
from gated_semantic_cache.structured_exact.normalize import normalize_query
from gated_semantic_cache.structured_exact.schema import Constraint, StructuredQuery


def extract_structured_query(raw_text: str) -> StructuredQuery:
    normalized_text = normalize_query(raw_text)

    constraints: list[Constraint] = []
    constraints.extend(extract_ids(normalized_text))
    constraints.extend(extract_numeric_bounds(normalized_text))
    constraints.extend(extract_dimensions(normalized_text))
    constraints.extend(extract_quantities(normalized_text))
    constraints.extend(extract_dates(normalized_text))
    constraints.extend(extract_categoricals(normalized_text))
    constraints = _dedupe_constraints(constraints)

    anchors = extract_anchors(normalized_text)
    ambiguity_flags = detect_ambiguity(normalized_text)
    confidence = score_structured_confidence(anchors, tuple(constraints), ambiguity_flags)

    return StructuredQuery(
        normalized_text=normalized_text,
        anchors=anchors,
        constraints=tuple(constraints),
        ambiguity_flags=ambiguity_flags,
        confidence=confidence,
    )


def _dedupe_constraints(constraints: list[Constraint]) -> list[Constraint]:
    seen: dict[tuple[str, str, str | None, str, str | None], Constraint] = {}
    for constraint in constraints:
        key = (
            constraint.kind,
            constraint.name,
            constraint.op,
            str(constraint.value),
            constraint.unit,
        )
        prior = seen.get(key)
        if prior is None or constraint.confidence > prior.confidence:
            seen[key] = constraint
    return list(seen.values())
