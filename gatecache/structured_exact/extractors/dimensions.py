from __future__ import annotations

import re

from gatecache.structured_exact.schema import Constraint


DIM_RE = re.compile(r"\b(\d{1,3})x(\d{1,3})\b", re.I)
SIZE_RE = re.compile(r"\bsize\s+([a-z0-9x.-]+)\b", re.I)
UNIT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(inch|inches|cm|mm|gb|tb)\b", re.I)
NAMED_MEASURE_RE = re.compile(r"\b(waist|length|inseam)\s+(\d+(?:\.\d+)?)\b", re.I)
CAPACITY_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(gb|tb)\b", re.I)


def extract_dimensions(text: str) -> list[Constraint]:
    constraints: list[Constraint] = []
    for match in DIM_RE.finditer(text):
        constraints.append(
            Constraint(
                kind="dimension",
                name="size",
                value=f"{match.group(1)}x{match.group(2)}",
                confidence=0.95,
                span_text=match.group(0),
            )
        )
    for match in SIZE_RE.finditer(text):
        constraints.append(
            Constraint(
                kind="dimension",
                name="size",
                value=match.group(1).lower(),
                confidence=0.85,
                span_text=match.group(0),
            )
        )
    for match in NAMED_MEASURE_RE.finditer(text):
        constraints.append(
            Constraint(
                kind="dimension",
                name=match.group(1).lower(),
                value=_normalize_number(match.group(2)),
                unit="inch",
                confidence=0.94,
                span_text=match.group(0),
            )
        )
    for match in UNIT_RE.finditer(text):
        unit = _normalize_unit(match.group(2))
        constraints.append(
            Constraint(
                kind="dimension",
                name=unit,
                value=_normalize_number(match.group(1)),
                unit=unit,
                confidence=0.9,
                span_text=match.group(0),
            )
        )
    for match in CAPACITY_RE.finditer(text):
        unit = _normalize_unit(match.group(2))
        constraints.append(
            Constraint(
                kind="dimension",
                name="capacity",
                value=_normalize_number(match.group(1)),
                unit=unit,
                confidence=0.94,
                span_text=match.group(0),
            )
        )
    return _dedupe(constraints)


def _normalize_unit(value: str) -> str:
    return {"inches": "inch"}.get(value.lower(), value.lower())


def _normalize_number(value: str) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return str(number)


def _dedupe(constraints: list[Constraint]) -> list[Constraint]:
    seen: dict[tuple[str, str, str], Constraint] = {}
    for constraint in constraints:
        key = (constraint.kind, constraint.name, str(constraint.value))
        prior = seen.get(key)
        if prior is None or constraint.confidence > prior.confidence:
            seen[key] = constraint
    return list(seen.values())
