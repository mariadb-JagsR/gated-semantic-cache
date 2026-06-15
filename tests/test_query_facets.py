from datetime import UTC, datetime

from gated_semantic_cache.models.cache_entry import SemanticCacheEntry
from gated_semantic_cache.serving.query_facets import extract_query_facets, facet_conflict_reason


def test_vs_maps_to_compare_mode_and_terms() -> None:
    facets = extract_query_facets("react vs hue")

    assert "compare_mode:compare" in facets["contrast_facets"]
    assert facets["comparison_terms"] == ["hue", "react"]


def test_verdict_vs_compare_conflict() -> None:
    entry = _entry("react vs hue")
    reason = facet_conflict_reason("is react better than hue ?", entry)

    assert reason == "query_facet_contrast_conflict"


def test_added_comparison_participant_conflict() -> None:
    entry = _entry("react vs hue")
    reason = facet_conflict_reason("how about react vs hue vs angular", entry)

    assert reason == "query_facet_comparison_term_conflict"


def test_same_comparison_terms_allow() -> None:
    entry = _entry("react vs hue")
    reason = facet_conflict_reason("react versus hue", entry)

    assert reason is None


def test_with_term_regex_does_not_strip_leading_article_from_token() -> None:
    facets = extract_query_facets("include angular comparison")

    assert "angular" in facets.get("with_terms", [])
    assert "ngular" not in facets.get("with_terms", [])


def test_substituted_comparison_participant_conflict() -> None:
    entry = _entry("react vs hue")
    reason = facet_conflict_reason("react vs angular", entry)

    assert reason == "query_facet_comparison_term_conflict"


def test_deploy_paraphrase_does_not_trigger_contrast_conflict() -> None:
    entry = _entry("easy to deploy node js app to aws ?")
    reason = facet_conflict_reason("super simple to deploy node.js app to aws ?", entry)

    assert reason is None


def test_route_paraphrase_does_not_trigger_named_entity_conflict() -> None:
    entry = _entry("instructions for getting from JFK to Manhattan")
    reason = facet_conflict_reason("how to get from JFK airport to manhattan", entry)

    assert reason is None


def test_route_endpoints_are_named_entities_even_when_lowercase() -> None:
    facets = extract_query_facets("how to get from jfk airport to manhattan")

    assert facets["named_entities"] == ["jfk", "manhattan"]
    assert facets["route_pairs"] == ["jfk>manhattan"]


def test_named_entity_substitution_still_conflicts() -> None:
    entry = _entry("What is the capital of France?")
    reason = facet_conflict_reason("What is the capital of Germany?", entry)

    assert reason == "query_facet_named_entity_conflict"


def test_contrast_conflict_requires_disjoint_alternatives_not_superset() -> None:
    # Simulates removed cloud_target-style superset false positive: extra mention on one side only.
    current = {"contrast_facets": ["deploy_intent:deploy"]}
    cached = {"contrast_facets": ["deploy_intent:deploy"]}
    assert _contrast_conflict(current, cached) is False

    current = {"contrast_facets": ["code_object:string"]}
    cached = {"contrast_facets": ["code_object:list"]}
    assert _contrast_conflict(current, cached) is True


def _contrast_conflict(current: dict, cached: dict) -> bool:
    from gated_semantic_cache.serving import query_facets as qf

    return qf._contrast_conflict(current, cached)


def _entry(query: str) -> SemanticCacheEntry:
    return SemanticCacheEntry(
        cache_id="id",
        namespace="default",
        query_text_original=query,
        query_text_normalized=query.lower(),
        embedding_vector=[1.0, 0.0],
        response_payload={"answer": f"live:{query}"},
        response_preview=query,
        created_at=datetime.now(tz=UTC),
        expires_at=None,
        cache_policy_class="semantic",
        agent_version="v1",
        corpus_version=None,
        tool_or_schema_version=None,
        thread_scope_key=None,
        exact_anchor_key=None,
        freshness_class="stable",
        confidence_metadata={"query_facets": extract_query_facets(query)},
    )
