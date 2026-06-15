from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Constraint:
    kind: str
    name: str
    value: Any
    op: str | None = None
    unit: str | None = None
    confidence: float = 1.0
    span_text: str | None = None
    critical: bool = True


@dataclass(frozen=True, slots=True)
class StructuredQuery:
    normalized_text: str
    anchors: tuple[str, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    ambiguity_flags: tuple[str, ...] = ()
    confidence: float = 0.0
    shape: str = "structured_exact"

    def has_critical_constraints(self) -> bool:
        return any(constraint.critical for constraint in self.constraints)

    def critical_constraints(self) -> tuple[Constraint, ...]:
        return tuple(constraint for constraint in self.constraints if constraint.critical)
