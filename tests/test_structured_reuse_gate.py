from datetime import UTC, datetime, timedelta

from gatecache.models.cache_entry import SemanticCacheEntry
from gatecache.serving.structured_reuse_gate import (
    compute_structured_critical_signature,
    structured_reuse_gate,
)
from gatecache.structured_exact.schema import Constraint, StructuredQuery
from gatecache.structured_exact.structured_query import extract_structured_query


def _entry_with_sig(sig: str) -> SemanticCacheEntry:
    return SemanticCacheEntry(
        cache_id="x",
        namespace="default",
        query_text_original="cached",
        query_text_normalized="cached",
        embedding_vector=[0.0],
        response_payload={},
        response_preview="",
        created_at=datetime.now(tz=UTC),
        expires_at=datetime.now(tz=UTC) + timedelta(hours=1),
        cache_policy_class="semantic_ok",
        agent_version="v1",
        corpus_version=None,
        tool_or_schema_version=None,
        thread_scope_key=None,
        exact_anchor_key=None,
        freshness_class="stable",
        structured_critical_signature=sig,
        structured_confidence_at_insert=0.9,
        confidence_metadata={"structured_identifier_pairs": {"id:=": "11022"}},
    )


def test_high_confidence_mismatch_rejects() -> None:
    c1 = Constraint(kind="identifier", name="id", value="88291", op="=", confidence=0.9)
    sq = StructuredQuery(
        normalized_text="q",
        anchors=(),
        constraints=(c1,),
        ambiguity_flags=(),
        confidence=0.9,
    )
    sig = compute_structured_critical_signature(sq)
    other = "deadbeef" * 8  # wrong hash
    assert structured_reuse_gate(sq, _entry_with_sig(other)) == "structured_critical_mismatch"
    assert structured_reuse_gate(sq, _entry_with_sig(sig or "")) is None


def test_low_extraction_confidence_skips_gate() -> None:
    c1 = Constraint(kind="identifier", name="id", value="999", op="=", confidence=0.9)
    sq = StructuredQuery(
        normalized_text="q",
        anchors=(),
        constraints=(c1,),
        ambiguity_flags=(),
        confidence=0.4,
    )
    assert structured_reuse_gate(sq, _entry_with_sig("any")) is None


def test_retail_filter_constraints_are_extracted() -> None:
    sq = extract_structured_query(
        "show me pants with waist 32, length 32, cotton, must be made in us, doesn't fade, black"
    )
    canonical = {
        f"{constraint.kind}:{constraint.name}:{constraint.value}:{constraint.unit}"
        for constraint in sq.critical_constraints()
    }

    assert "dimension:waist:32:inch" in canonical
    assert "dimension:length:32:inch" in canonical
    assert "categorical:material:cotton:None" in canonical
    assert "categorical:color:black:None" in canonical
    assert "categorical:attribute:made in us:None" in canonical
    assert "categorical:attribute:doesn't fade:None" in canonical
