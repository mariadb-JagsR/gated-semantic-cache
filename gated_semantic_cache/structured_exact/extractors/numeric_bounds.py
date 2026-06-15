from __future__ import annotations

import re

from gated_semantic_cache.structured_exact.schema import Constraint


UNDER_RE = re.compile(r"\b(?:under|less than|below|at most|no more than)\s+(?:usd\s+)?(\d+(?:\.\d+)?)(?:\s+usd)?\b", re.I)
OVER_RE = re.compile(r"\b(?:over|greater than|more than|at least|minimum)\s+(?:usd\s+)?(\d+(?:\.\d+)?)(?:\s+usd)?\b", re.I)
BETWEEN_RE = re.compile(r"\bbetween\s*(\d+(?:\.\d+)?)\s*and\s*(\d+(?:\.\d+)?)\b", re.I)


def extract_numeric_bounds(text: str) -> list[Constraint]:
    constraints: list[Constraint] = []
    for match in UNDER_RE.finditer(text):
        unit = "usd" if "usd" in match.group(0).lower() else None
        constraints.append(
            Constraint(
                kind="numeric_bound",
                name="price" if unit == "usd" else "upper_bound",
                op="<=",
                value=_normalize_number(match.group(1)),
                unit=unit,
                confidence=0.95,
                span_text=match.group(0),
            )
        )
    for match in OVER_RE.finditer(text):
        unit = "usd" if "usd" in match.group(0).lower() else None
        constraints.append(
            Constraint(
                kind="numeric_bound",
                name="price" if unit == "usd" else "lower_bound",
                op=">=",
                value=_normalize_number(match.group(1)),
                unit=unit,
                confidence=0.92,
                span_text=match.group(0),
            )
        )
    for match in BETWEEN_RE.finditer(text):
        constraints.append(
            Constraint(
                kind="numeric_bound",
                name="range",
                op="between",
                value=f"{_normalize_number(match.group(1))}..{_normalize_number(match.group(2))}",
                confidence=0.9,
                span_text=match.group(0),
            )
        )
    return constraints


def _normalize_number(value: str) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return str(number)
