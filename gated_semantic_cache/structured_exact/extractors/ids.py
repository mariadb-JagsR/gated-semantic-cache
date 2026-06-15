from __future__ import annotations

import re

from gated_semantic_cache.routing.tech_tokens import filter_plausible_hostnames
from gated_semantic_cache.structured_exact.schema import Constraint


UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
HOST_RE = re.compile(r"\b[a-z0-9.-]+\.[a-z]{2,}\b", re.I)
LONG_NUM_RE = re.compile(r"\b\d{6,}\b")
LABELED_ID_RE = re.compile(
    r"\b(order|ticket|incident|case|sku|id|account|customer|host)\s*[:#-]?\s*([a-z0-9.-]*\d[a-z0-9.-]*)\b",
    re.I,
)


def extract_ids(text: str) -> list[Constraint]:
    constraints: list[Constraint] = []
    for match in UUID_RE.finditer(text):
        constraints.append(
            Constraint(kind="identifier", name="uuid", value=match.group(0).lower(), confidence=0.99, span_text=match.group(0))
        )
    for match in EMAIL_RE.finditer(text):
        constraints.append(
            Constraint(kind="identifier", name="email", value=match.group(0).lower(), confidence=0.99, span_text=match.group(0))
        )
    for match in LABELED_ID_RE.finditer(text):
        constraints.append(
            Constraint(
                kind="identifier",
                name=match.group(1).lower(),
                value=match.group(2).lower(),
                confidence=0.95,
                span_text=match.group(0),
            )
        )
    for hostname in filter_plausible_hostnames([match.group(0) for match in HOST_RE.finditer(text)]):
        constraints.append(
            Constraint(kind="identifier", name="hostname", value=hostname.lower(), confidence=0.92, span_text=hostname)
        )
    for match in LONG_NUM_RE.finditer(text):
        constraints.append(
            Constraint(kind="identifier", name="numeric_id", value=match.group(0), confidence=0.75, span_text=match.group(0))
        )
    return _dedupe(constraints)


def _dedupe(constraints: list[Constraint]) -> list[Constraint]:
    seen: dict[tuple[str, str, str], Constraint] = {}
    for constraint in constraints:
        key = (constraint.kind, constraint.name, str(constraint.value))
        prior = seen.get(key)
        if prior is None or constraint.confidence > prior.confidence:
            seen[key] = constraint
    return list(seen.values())
