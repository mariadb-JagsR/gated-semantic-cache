from __future__ import annotations

from gated_semantic_cache.structured_exact.schema import Constraint, StructuredQuery


def build_structured_key(query: StructuredQuery, namespace: str = "default") -> str | None:
    if query.confidence < 0.55:
        return None

    critical = tuple(
        constraint for constraint in query.critical_constraints() if constraint.confidence >= 0.85
    )
    if not critical or not _has_reliable_structure(critical):
        return None

    parts = [f"ns:{namespace}"]
    for anchor in sorted(query.anchors):
        parts.append(f"anchor:{anchor}")
    for constraint in sorted(critical, key=canonicalize_constraint):
        parts.append(canonicalize_constraint(constraint))
    return "|".join(parts)


def canonicalize_constraint(constraint: Constraint) -> str:
    if constraint.kind == "numeric_bound":
        suffix = "" if not constraint.unit else constraint.unit
        return f"{constraint.name}{constraint.op}{constraint.value}{suffix}"
    if constraint.op:
        return f"{constraint.kind}:{constraint.name}{constraint.op}{constraint.value}"
    return f"{constraint.kind}:{constraint.name}={constraint.value}"


def _has_reliable_structure(constraints: tuple[Constraint, ...]) -> bool:
    strong_kinds = {"identifier", "dimension", "numeric_bound", "quantity", "date_window", "binary_flag"}
    if any(constraint.kind in strong_kinds for constraint in constraints):
        return True
    return len(constraints) >= 2
