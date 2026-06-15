from __future__ import annotations

from gatecache.structured_exact.canonical_key import build_structured_key
from gatecache.structured_exact.matching import critical_constraints_match
from gatecache.structured_exact.normalize import normalize_query as normalize_structured_query
from gatecache.structured_exact.schema import Constraint as StructuredConstraint
from gatecache.structured_exact.schema import StructuredQuery as StructuredExtraction
from gatecache.structured_exact.structured_query import extract_structured_query


def extract_structured_constraints(query: str) -> StructuredExtraction:
    return extract_structured_query(query)


def build_canonical_structured_key(extraction: StructuredExtraction, namespace: str = "default") -> str | None:
    return build_structured_key(extraction, namespace=namespace)
