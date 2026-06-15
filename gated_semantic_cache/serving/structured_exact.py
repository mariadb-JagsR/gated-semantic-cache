from __future__ import annotations

from gated_semantic_cache.structured_exact.canonical_key import build_structured_key
from gated_semantic_cache.structured_exact.matching import critical_constraints_match
from gated_semantic_cache.structured_exact.normalize import normalize_query as normalize_structured_query
from gated_semantic_cache.structured_exact.schema import Constraint as StructuredConstraint
from gated_semantic_cache.structured_exact.schema import StructuredQuery as StructuredExtraction
from gated_semantic_cache.structured_exact.structured_query import extract_structured_query


def extract_structured_constraints(query: str) -> StructuredExtraction:
    return extract_structured_query(query)


def build_canonical_structured_key(extraction: StructuredExtraction, namespace: str = "default") -> str | None:
    return build_structured_key(extraction, namespace=namespace)
