from __future__ import annotations

from gatecache.structured_exact.schema import Constraint


def score_structured_confidence(
    anchors: tuple[str, ...],
    constraints: tuple[Constraint, ...],
    ambiguity_flags: tuple[str, ...],
) -> float:
    score = 0.0

    if anchors:
        score += 0.15

    high_conf = [constraint for constraint in constraints if constraint.confidence >= 0.9]
    score += min(0.45, 0.1 * len(high_conf))

    if any(constraint.kind == "identifier" for constraint in constraints):
        score += 0.35

    if any(constraint.kind in {"dimension", "numeric_bound", "quantity", "date_window"} for constraint in constraints):
        score += 0.45

    if ambiguity_flags:
        score -= 0.3

    return round(max(0.0, min(1.0, score)), 4)
