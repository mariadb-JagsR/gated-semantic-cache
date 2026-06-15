from __future__ import annotations

import re

from gatecache.models.cache_entry import SemanticCacheEntry
from gatecache.routing.features import normalize_query_text
from gatecache.structured_exact.schema import StructuredQuery

_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_TOKEN_RE = re.compile(r"[a-z0-9$.-]+")
_NEGATION_RE = re.compile(r"\b(no|not|without|exclude|excluding|doesn'?t|must not|never)\b")
_COMPARISON_RE = re.compile(r"\b(under|over|less than|more than|at least|at most|minimum|maximum|between)\b")
_UNIT_OR_MEASURE_RE = re.compile(
    r"\b(inch|inches|cm|mm|gb|tb|usd|dollars?|days?|weeks?|months?|waist|length|size|capacity)\b"
)
_FILTER_CONNECTOR_RE = re.compile(r"\b(with|without|must|and|or|plus|including|included)\b")


def constraint_risk_reason(query: str, entry: SemanticCacheEntry, sq: StructuredQuery) -> str | None:
    """Return a reason when high-similarity reuse should require a judge.

    This is intentionally domain-agnostic. It does not try to know what "pants" or
    "loans" mean; it only detects prompts whose answer correctness likely depends on
    small tokens such as numbers, units, negation, or filter lists. If structured
    extraction already captured a reliable signature on both sides, the structured
    reuse gate remains the primary safety check and this detector stays out of the way.
    """

    if sq.confidence >= 0.85 and entry.structured_critical_signature:
        return None

    current = _signals(query)
    cached = _signals(entry.query_text_original or entry.query_text_normalized)
    if not current["heavy"] and not cached["heavy"]:
        return None

    if current["numbers"] != cached["numbers"]:
        return "constraint_risk_numeric_delta"
    if current["units"] != cached["units"]:
        return "constraint_risk_unit_delta"
    if current["has_negation"] != cached["has_negation"]:
        return "constraint_risk_negation_delta"

    current_filters = current["filter_tokens"]
    cached_filters = cached["filter_tokens"]
    if current_filters and cached_filters:
        overlap = len(current_filters & cached_filters)
        smaller = max(1, min(len(current_filters), len(cached_filters)))
        if overlap / smaller < 0.75:
            return "constraint_risk_filter_delta"

    return "constraint_risk_unverified_heavy_pair"


def _signals(text: str) -> dict[str, object]:
    normalized = normalize_query_text(text)
    tokens = set(_TOKEN_RE.findall(normalized))
    numbers = frozenset(_NUMBER_RE.findall(normalized))
    units = frozenset(match.group(0).lower() for match in _UNIT_OR_MEASURE_RE.finditer(normalized))
    has_negation = bool(_NEGATION_RE.search(normalized))
    connector_count = len(_FILTER_CONNECTOR_RE.findall(normalized))
    comma_count = normalized.count(",")
    heavy = (
        len(numbers) >= 2
        or bool(units)
        or has_negation
        or bool(_COMPARISON_RE.search(normalized))
        or comma_count >= 2
        or connector_count >= 3
    )
    stop = {
        "a",
        "an",
        "the",
        "me",
        "show",
        "find",
        "list",
        "with",
        "without",
        "must",
        "be",
        "in",
        "to",
        "for",
        "and",
        "or",
    }
    filter_tokens = frozenset(token for token in tokens if token not in stop and len(token) > 1)
    return {
        "heavy": heavy,
        "numbers": numbers,
        "units": units,
        "has_negation": has_negation,
        "filter_tokens": filter_tokens,
    }
