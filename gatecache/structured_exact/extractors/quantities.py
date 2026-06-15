from __future__ import annotations

import re

from gatecache.structured_exact.schema import Constraint


QUANTITY_RE = re.compile(
    r"\b(\d+)\s+(checked bag|checked bags|guest|guests|item|items|passenger|passengers)\b",
    re.I,
)


def extract_quantities(text: str) -> list[Constraint]:
    constraints: list[Constraint] = []
    for match in QUANTITY_RE.finditer(text):
        label = _normalize_quantity_name(match.group(2))
        constraints.append(
            Constraint(
                kind="quantity",
                name=label,
                value=str(int(match.group(1))),
                confidence=0.93,
                span_text=match.group(0),
            )
        )
    return constraints


def _normalize_quantity_name(value: str) -> str:
    singular = value.lower().rstrip("s")
    mapping = {
        "checked bag": "checked_bags",
        "guest": "guests",
        "item": "items",
        "passenger": "passengers",
    }
    return mapping.get(singular, singular.replace(" ", "_"))
