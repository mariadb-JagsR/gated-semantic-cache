from __future__ import annotations

from gated_semantic_cache.structured_exact.schema import StructuredQuery


def critical_constraints_match(left: StructuredQuery, right: StructuredQuery) -> bool:
    left_constraints = _critical_constraint_map(left)
    right_constraints = _critical_constraint_map(right)
    if not left_constraints or not right_constraints:
        return False
    return left_constraints == right_constraints


def _critical_constraint_map(query: StructuredQuery) -> dict[str, str]:
    items: dict[str, str] = {}
    for constraint in query.critical_constraints():
        if constraint.confidence < 0.85:
            continue
        key = f"{constraint.kind}:{constraint.name}"
        value = f"{constraint.op or '='}:{constraint.value}:{constraint.unit or ''}"
        items[key] = value
    return items
