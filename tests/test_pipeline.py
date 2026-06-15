from gated_semantic_cache.cache.exact_cache import ExactCache
from gated_semantic_cache.cache.semantic_store import SemanticStore
from gated_semantic_cache.embeddings.backends import embedding_dim_for_openai_model, make_constant_unit_embedder
from gated_semantic_cache.models.context import RequestContext
from gated_semantic_cache.routing.classifier import train_default_classifier
from gated_semantic_cache.routing.classifier import RoutingPrediction
from gated_semantic_cache.routing.labels import RoutingLabel
from gated_semantic_cache.serving.pipeline import SemanticCachePipeline, _neighbor_judge_preflight
from gated_semantic_cache.eval.datasets import build_routing_dataset


def _build_pipeline() -> SemanticCachePipeline:
    dim = embedding_dim_for_openai_model("text-embedding-3-small")
    return SemanticCachePipeline(
        router=train_default_classifier(build_routing_dataset()),
        exact_cache=ExactCache(),
        semantic_store=SemanticStore(dimension=dim),
        embedder=make_constant_unit_embedder(dimension=dim),
    )


def _build_semantic_ok_pipeline() -> SemanticCachePipeline:
    class SemanticOkRouter:
        def predict(self, _query: str) -> RoutingPrediction:
            return RoutingPrediction(
                label=RoutingLabel.SEMANTIC_OK,
                confidence=1.0,
                probabilities={RoutingLabel.SEMANTIC_OK: 1.0},
            )

    dim = embedding_dim_for_openai_model("text-embedding-3-small")
    return SemanticCachePipeline(
        router=SemanticOkRouter(),  # type: ignore[arg-type]
        exact_cache=ExactCache(),
        semantic_store=SemanticStore(dimension=dim),
        embedder=make_constant_unit_embedder(dimension=dim),
    )


def _build_semantic_ok_pipeline_with_embedder(embedder) -> SemanticCachePipeline:
    class SemanticOkRouter:
        def predict(self, _query: str) -> RoutingPrediction:
            return RoutingPrediction(
                label=RoutingLabel.SEMANTIC_OK,
                confidence=1.0,
                probabilities={RoutingLabel.SEMANTIC_OK: 1.0},
            )

    return SemanticCachePipeline(
        router=SemanticOkRouter(),  # type: ignore[arg-type]
        exact_cache=ExactCache(),
        semantic_store=SemanticStore(dimension=2),
        embedder=embedder,
    )


def test_exact_cache_short_circuits_routing() -> None:
    pipeline = _build_pipeline()
    context = RequestContext()
    live_calls = {"count": 0}

    def live_answer(query: str, _: RequestContext) -> dict[str, object]:
        live_calls["count"] += 1
        return {"answer": f"live:{query}", "success": True}

    first = pipeline.answer_query("Explain what semantic caching is", context, live_answer)
    second = pipeline.answer_query("Explain what semantic caching is", context, live_answer)

    assert first.source == "live"
    assert second.source == "exact_cache"
    assert live_calls["count"] == 1
    assert first.trace.exact_cache_key_sha256 is not None
    assert first.trace.structured_extraction_attempted is True
    assert first.trace.routing_label is not None


def test_thread_scoped_queries_only_reuse_with_scope() -> None:
    pipeline = _build_pipeline()
    live_calls = {"count": 0}

    def live_answer(query: str, context: RequestContext) -> dict[str, object]:
        live_calls["count"] += 1
        return {"answer": f"{context.thread_scope_key}:{query}", "success": True}

    no_scope = RequestContext(thread_scope_key=None)
    with_scope = RequestContext(thread_scope_key="thread-1")

    first = pipeline.answer_query("Same but in december", no_scope, live_answer)
    second = pipeline.answer_query("Same but in december", with_scope, live_answer)
    third = pipeline.answer_query("Same but in december", with_scope, live_answer)

    assert first.source == "live"
    assert second.source == "live"
    assert third.source in {"exact_cache", "semantic_cache"}
    assert live_calls["count"] == 2


def test_exact_only_queries_do_not_use_semantic_retrieval() -> None:
    pipeline = _build_pipeline()
    context = RequestContext()

    def live_answer(query: str, _: RequestContext) -> dict[str, object]:
        return {"answer": f"live:{query}", "success": True}

    first = pipeline.answer_query("Lookup order #A123 status", context, live_answer)
    second = pipeline.answer_query("Lookup order #A123 status", context, live_answer)

    assert first.trace.semantic_lookup_attempted is False
    assert second.trace.semantic_lookup_attempted is False
    assert first.trace.structured_extraction_attempted is True
    assert first.trace.structured_canonical_key_sha256 is not None
    assert first.trace.anchor_lookup_key_sha256 is not None


def test_exact_only_without_anchor_low_confidence_falls_back_to_semantic() -> None:
    class ExactOnlyRouter:
        def predict(self, _query: str) -> RoutingPrediction:
            return RoutingPrediction(
                label=RoutingLabel.EXACT_ONLY,
                confidence=0.35,
                probabilities={RoutingLabel.EXACT_ONLY: 0.35},
            )

    dim = embedding_dim_for_openai_model("text-embedding-3-small")
    pipeline = SemanticCachePipeline(
        router=ExactOnlyRouter(),  # type: ignore[arg-type]
        exact_cache=ExactCache(),
        semantic_store=SemanticStore(dimension=dim),
        embedder=make_constant_unit_embedder(dimension=dim),
    )
    context = RequestContext(semantic_threshold=0.55, exact_only_min_route_confidence=0.55)
    live_calls = {"count": 0}

    def live_answer(query: str, _: RequestContext) -> dict[str, object]:
        live_calls["count"] += 1
        return {"answer": f"fee policy:{query}", "success": True}

    first = pipeline.answer_query("international wire transfer fees for a business account", context, live_answer)
    second = pipeline.answer_query("business account international wire transfer fees", context, live_answer)

    assert first.source == "live"
    assert first.trace.routing_label == RoutingLabel.EXACT_ONLY.value
    assert first.trace.routing_exact_only_downgraded is True
    assert first.trace.anchor_lookup_key_sha256 is None
    assert first.trace.insert_performed is True
    assert second.trace.semantic_lookup_attempted is True
    assert second.source == "semantic_cache"
    assert live_calls["count"] == 1


def test_exact_only_without_anchor_high_confidence_stays_live_only() -> None:
    class ExactOnlyRouter:
        def predict(self, _query: str) -> RoutingPrediction:
            return RoutingPrediction(
                label=RoutingLabel.EXACT_ONLY,
                confidence=0.9,
                probabilities={RoutingLabel.EXACT_ONLY: 0.9},
            )

    dim = embedding_dim_for_openai_model("text-embedding-3-small")
    pipeline = SemanticCachePipeline(
        router=ExactOnlyRouter(),  # type: ignore[arg-type]
        exact_cache=ExactCache(),
        semantic_store=SemanticStore(dimension=dim),
        embedder=make_constant_unit_embedder(dimension=dim),
    )
    context = RequestContext(semantic_threshold=0.55)
    live_calls = {"count": 0}

    def live_answer(query: str, _: RequestContext) -> dict[str, object]:
        live_calls["count"] += 1
        return {"answer": f"fee policy:{query}", "success": True}

    first = pipeline.answer_query("international wire transfer fees for a business account", context, live_answer)
    second = pipeline.answer_query("business account international wire transfer fees", context, live_answer)

    assert first.source == "live"
    assert first.trace.routing_exact_only_downgraded is False
    assert second.trace.semantic_lookup_attempted is False
    assert live_calls["count"] == 2


def test_node_js_deploy_paraphrase_uses_semantic_path() -> None:
    pipeline = _build_pipeline()
    context = RequestContext(semantic_threshold=0.55)

    def live_answer(query: str, _: RequestContext) -> dict[str, object]:
        return {"answer": f"live:{query}", "success": True}

    response = pipeline.answer_query("details on deploying a node.js app to gcp", context, live_answer)

    assert response.trace.routing_exact_only_downgraded in {True, False}
    assert response.trace.semantic_lookup_attempted is True
    assert response.trace.structured_critical_preview == []


def test_semantic_ok_queries_can_reuse_via_ann() -> None:
    pipeline = _build_pipeline()
    context = RequestContext(semantic_threshold=0.55)
    live_calls = {"count": 0}

    def live_answer(query: str, _: RequestContext) -> dict[str, object]:
        live_calls["count"] += 1
        return {"answer": "semantic caching is reuse of prior answers", "success": True}

    first = pipeline.answer_query("Explain what semantic caching is", context, live_answer)
    second = pipeline.answer_query("What is semantic caching?", context, live_answer)

    assert first.source == "live"
    assert second.source == "semantic_cache"
    assert live_calls["count"] == 1
    assert second.trace.rejected_reasons == []
    assert second.trace.final_result_source == "semantic_cache"
    assert isinstance(second.trace.semantic_neighbor_filter_counts, dict)


def test_gray_zone_candidate_requires_judge_without_hiding_similarity() -> None:
    def embedder(query: str) -> list[float]:
        if "gray" in query:
            return [0.75, 0.6614378277661477]
        return [1.0, 0.0]

    pipeline = _build_semantic_ok_pipeline_with_embedder(embedder)
    context = RequestContext(semantic_threshold=0.86, semantic_low_watermark=0.70)
    live_calls = {"count": 0}

    def live_answer(query: str, _: RequestContext) -> dict[str, object]:
        live_calls["count"] += 1
        return {"answer": f"ok:{query}", "success": True}

    pipeline.answer_query("seed query", context, live_answer)
    second = pipeline.answer_query("gray query", context, live_answer)

    assert second.source == "live"
    assert second.trace.top_candidate_similarity is not None
    assert second.trace.top_candidate_similarity < context.semantic_threshold
    assert second.trace.semantic_post_ann_reject_reason == "semantic_gray_zone_requires_judge"
    assert second.trace.rejected_reasons == ["semantic_gray_zone_requires_judge"]
    assert live_calls["count"] == 2


def test_gray_zone_candidate_can_reuse_when_judge_allows() -> None:
    def embedder(query: str) -> list[float]:
        if "gray" in query:
            return [0.75, 0.6614378277661477]
        return [1.0, 0.0]

    pipeline = _build_semantic_ok_pipeline_with_embedder(embedder)
    pipeline.neighbor_judge = lambda _query, _entry, _context: None
    context = RequestContext(semantic_threshold=0.86, semantic_low_watermark=0.70)

    def live_answer(query: str, _: RequestContext) -> dict[str, object]:
        return {"answer": f"ok:{query}", "success": True}

    pipeline.answer_query("seed query", context, live_answer)
    second = pipeline.answer_query("gray query", context, live_answer)

    assert second.source == "semantic_cache"
    assert second.trace.neighbor_judge_invoked is True
    assert second.trace.semantic_post_ann_reject_reason is None


def test_below_low_watermark_candidate_stays_filtered() -> None:
    def embedder(query: str) -> list[float]:
        if "far" in query:
            return [0.60, 0.80]
        return [1.0, 0.0]

    pipeline = _build_semantic_ok_pipeline_with_embedder(embedder)
    pipeline.neighbor_judge = lambda _query, _entry, _context: None
    context = RequestContext(semantic_threshold=0.86, semantic_low_watermark=0.70)

    def live_answer(query: str, _: RequestContext) -> dict[str, object]:
        return {"answer": f"ok:{query}", "success": True}

    pipeline.answer_query("seed query", context, live_answer)
    second = pipeline.answer_query("far query", context, live_answer)

    assert second.source == "live"
    assert second.trace.top_candidate_similarity is not None
    assert second.trace.top_candidate_similarity < context.semantic_low_watermark
    assert second.trace.rejected_reasons == ["below_threshold"]
    assert second.trace.neighbor_judge_invoked is False


def test_facet_gate_allows_same_protected_token_paraphrase() -> None:
    pipeline = _build_semantic_ok_pipeline()
    context = RequestContext(semantic_threshold=0.55)
    live_calls = {"count": 0}

    def live_answer(query: str, _: RequestContext) -> dict[str, object]:
        live_calls["count"] += 1
        return {"answer": f"ok:{query}", "success": True}

    first = pipeline.answer_query("docs for v2.7.1", context, live_answer)
    second = pipeline.answer_query("documentation for v2.7.1", context, live_answer)

    assert first.source == "live"
    assert second.source == "semantic_cache"
    assert second.trace.semantic_facet_conflict_reason is None
    assert live_calls["count"] == 1


def test_facet_gate_rejects_changed_protected_token_without_judge() -> None:
    pipeline = _build_semantic_ok_pipeline()
    context = RequestContext(semantic_threshold=0.55)
    live_calls = {"count": 0}

    def live_answer(query: str, _: RequestContext) -> dict[str, object]:
        live_calls["count"] += 1
        return {"answer": f"ok:{query}", "success": True}

    pipeline.answer_query("docs for v2.7.1", context, live_answer)
    second = pipeline.answer_query("docs for v2.8.0", context, live_answer)

    assert second.source == "live"
    assert second.trace.semantic_facet_conflict_reason == "query_facet_protected_token_conflict"
    assert second.trace.semantic_post_ann_reject_reason == "query_facet_protected_token_conflict"
    assert second.trace.rejected_reasons == ["query_facet_protected_token_conflict"]
    assert live_calls["count"] == 2


def test_facet_gate_rejects_named_entity_substitution_even_with_judge() -> None:
    pipeline = _build_semantic_ok_pipeline()
    pipeline.neighbor_judge = lambda _query, _entry, _context: None
    context = RequestContext(semantic_threshold=0.55)

    def live_answer(query: str, _: RequestContext) -> dict[str, object]:
        return {"answer": f"ok:{query}", "success": True}

    pipeline.answer_query("What is the capital of France?", context, live_answer)
    second = pipeline.answer_query("What is the capital of Germany?", context, live_answer)

    assert second.source == "live"
    assert second.trace.semantic_facet_conflict_reason == "query_facet_named_entity_conflict"
    assert second.trace.neighbor_judge_invoked is False


def test_facet_gate_rejects_contrast_shift_even_with_judge() -> None:
    pipeline = _build_semantic_ok_pipeline()
    pipeline.neighbor_judge = lambda _query, _entry, _context: None
    context = RequestContext(semantic_threshold=0.55)

    def live_answer(query: str, _: RequestContext) -> dict[str, object]:
        return {"answer": f"ok:{query}", "success": True}

    pipeline.answer_query("How do I reverse a string in Python?", context, live_answer)
    second = pipeline.answer_query("How do I reverse a list in Python?", context, live_answer)

    assert second.source == "live"
    assert second.trace.semantic_facet_conflict_reason == "query_facet_contrast_conflict"
    assert second.trace.neighbor_judge_invoked is False


def test_facet_gate_rejects_currency_direction_swap() -> None:
    pipeline = _build_semantic_ok_pipeline()
    context = RequestContext(semantic_threshold=0.55)

    def live_answer(query: str, _: RequestContext) -> dict[str, object]:
        return {"answer": f"ok:{query}", "success": True}

    pipeline.answer_query("Convert 100 USD to EUR", context, live_answer)
    second = pipeline.answer_query("Convert 100 EUR to USD", context, live_answer)

    assert second.source == "live"
    assert second.trace.semantic_facet_conflict_reason == "query_facet_quantity_conflict"


def test_facet_gate_rejects_verdict_after_neutral_comparison() -> None:
    pipeline = _build_semantic_ok_pipeline()
    pipeline.neighbor_judge = lambda _query, _entry, _context: None
    context = RequestContext(semantic_threshold=0.55)

    def live_answer(query: str, _: RequestContext) -> dict[str, object]:
        return {"answer": f"ok:{query}", "success": True}

    pipeline.answer_query("react vs hue", context, live_answer)
    second = pipeline.answer_query("is react better than hue ?", context, live_answer)

    assert second.source == "live"
    assert second.trace.semantic_facet_conflict_reason == "query_facet_contrast_conflict"
    assert second.trace.neighbor_judge_invoked is False


def test_facet_gate_rejects_added_comparison_participant() -> None:
    pipeline = _build_semantic_ok_pipeline()
    pipeline.neighbor_judge = lambda _query, _entry, _context: None
    context = RequestContext(semantic_threshold=0.55)

    def live_answer(query: str, _: RequestContext) -> dict[str, object]:
        return {"answer": f"ok:{query}", "success": True}

    pipeline.answer_query("react vs hue", context, live_answer)
    second = pipeline.answer_query("how about react vs hue vs angular", context, live_answer)

    assert second.source == "live"
    assert second.trace.semantic_facet_conflict_reason == "query_facet_comparison_term_conflict"
    assert second.trace.neighbor_judge_invoked is False


def test_facet_gate_allows_deploy_paraphrase_with_code_like_spelling_change() -> None:
    pipeline = _build_semantic_ok_pipeline()
    pipeline.neighbor_judge = lambda _query, _entry, _context: None
    context = RequestContext(semantic_threshold=0.55)

    def live_answer(query: str, _: RequestContext) -> dict[str, object]:
        return {"answer": f"ok:{query}", "success": True}

    pipeline.answer_query("easy to deploy node js app to aws ?", context, live_answer)
    second = pipeline.answer_query("super simple to deploy node.js app to aws ?", context, live_answer)

    assert second.source == "semantic_cache"
    assert second.trace.semantic_facet_conflict_reason is None


def test_facet_gate_allows_route_paraphrase_with_airport_and_case_changes() -> None:
    pipeline = _build_semantic_ok_pipeline()
    pipeline.neighbor_judge = lambda _query, _entry, _context: None
    context = RequestContext(semantic_threshold=0.55)

    def live_answer(query: str, _: RequestContext) -> dict[str, object]:
        return {"answer": f"ok:{query}", "success": True}

    pipeline.answer_query("instructions for getting from JFK to Manhattan", context, live_answer)
    second = pipeline.answer_query("how to get from JFK airport to manhattan", context, live_answer)

    assert second.source == "semantic_cache"
    assert second.trace.semantic_facet_conflict_reason is None


def test_semantic_hit_observability_no_misleading_reject_list() -> None:
    """``rejected_reasons`` must not list below_threshold for other neighbors when a hit is served."""
    pipeline = _build_pipeline()
    context = RequestContext(semantic_threshold=0.55)

    def live_answer(query: str, _: RequestContext) -> dict[str, object]:
        return {"answer": "ok", "success": True}

    pipeline.answer_query("Explain what semantic caching is", context, live_answer)
    second = pipeline.answer_query("What is semantic caching?", context, live_answer)
    assert second.source == "semantic_cache"
    assert second.trace.rejected_reasons == []


def test_semantic_lookup_rejects_tool_schema_mismatch() -> None:
    pipeline = _build_pipeline()
    writer_context = RequestContext(semantic_threshold=0.55, tool_or_schema_version="schema-v1")
    reader_context = RequestContext(semantic_threshold=0.55, tool_or_schema_version="schema-v2")
    live_calls = {"count": 0}

    def live_answer(query: str, _: RequestContext) -> dict[str, object]:
        live_calls["count"] += 1
        return {"answer": f"ok:{query}", "success": True}

    pipeline.answer_query("Explain what semantic caching is", writer_context, live_answer)
    second = pipeline.answer_query("What is semantic caching?", reader_context, live_answer)

    assert second.source == "live"
    assert second.trace.rejected_reasons == ["tool_or_schema_version_mismatch"]
    assert live_calls["count"] == 2


def test_neighbor_judge_runs_at_edge_and_can_veto() -> None:
    def veto(_q: str, _e, _c) -> str | None:
        return "edge_policy_veto"

    dim = embedding_dim_for_openai_model("text-embedding-3-small")
    pipeline = SemanticCachePipeline(
        router=train_default_classifier(build_routing_dataset()),
        exact_cache=ExactCache(),
        semantic_store=SemanticStore(dimension=dim),
        embedder=make_constant_unit_embedder(dimension=dim),
        neighbor_judge=veto,
    )
    context = RequestContext(semantic_threshold=0.55)

    def live_answer(query: str, _: RequestContext) -> dict[str, object]:
        return {"answer": "ok", "success": True}

    pipeline.answer_query("Explain what semantic caching is", context, live_answer)
    second = pipeline.answer_query("What is semantic caching?", context, live_answer)
    assert second.source == "live"
    assert second.trace.semantic_post_ann_reject_reason == "edge_policy_veto"
    assert second.trace.rejected_reasons == ["edge_policy_veto"]


def test_neighbor_judge_preflight_skips_for_strong_similarity() -> None:
    ctx = RequestContext(neighbor_judge_similarity_ceiling=0.9)
    skip, reason = _neighbor_judge_preflight(
        context=ctx, top_similarity=0.95, second_similarity=0.5, calls_used=0
    )
    assert skip and reason == "neighbor_judge_skipped_strong_similarity"


def test_neighbor_judge_preflight_skips_for_clear_margin() -> None:
    ctx = RequestContext(neighbor_judge_ambiguity_margin=0.05)
    skip, reason = _neighbor_judge_preflight(
        context=ctx, top_similarity=0.9, second_similarity=0.80, calls_used=0
    )
    assert skip and reason == "neighbor_judge_skipped_clear_margin"


def test_neighbor_judge_skipped_when_above_ceiling_even_if_judge_would_veto() -> None:
    def veto(_q: str, _e, _c) -> str | None:
        return "edge_policy_veto"

    dim = embedding_dim_for_openai_model("text-embedding-3-small")
    pipeline = SemanticCachePipeline(
        router=train_default_classifier(build_routing_dataset()),
        exact_cache=ExactCache(),
        semantic_store=SemanticStore(dimension=dim),
        embedder=make_constant_unit_embedder(dimension=dim),
        neighbor_judge=veto,
    )
    context = RequestContext(semantic_threshold=0.55, neighbor_judge_similarity_ceiling=0.0)

    def live_answer(query: str, _: RequestContext) -> dict[str, object]:
        return {"answer": "ok", "success": True}

    pipeline.answer_query("Explain what semantic caching is", context, live_answer)
    second = pipeline.answer_query("What is semantic caching?", context, live_answer)
    assert second.source == "semantic_cache"
    assert second.trace.neighbor_judge_skipped_reason == "neighbor_judge_skipped_strong_similarity"
    assert second.trace.semantic_post_ann_reject_reason is None


def test_negation_facet_rejects_unverified_hit_even_above_ceiling() -> None:
    dim = embedding_dim_for_openai_model("text-embedding-3-small")
    pipeline = SemanticCachePipeline(
        router=train_default_classifier(build_routing_dataset()),
        exact_cache=ExactCache(),
        semantic_store=SemanticStore(dimension=dim),
        embedder=make_constant_unit_embedder(dimension=dim),
    )
    context = RequestContext(semantic_threshold=0.55, neighbor_judge_similarity_ceiling=0.0)
    live_calls = {"count": 0}

    def live_answer(query: str, _: RequestContext) -> dict[str, object]:
        live_calls["count"] += 1
        return {"answer": f"ok:{query}", "success": True}

    pipeline.answer_query("find vegan, low sodium, snack box, no peanuts", context, live_answer)
    second = pipeline.answer_query("find vegan, low sodium, snack box, with peanuts", context, live_answer)

    assert second.source == "live"
    assert second.trace.semantic_facet_conflict_reason == "query_facet_negation_conflict"
    assert second.trace.semantic_post_ann_reject_reason == "query_facet_negation_conflict"
    assert second.trace.neighbor_judge_invoked is False
    assert live_calls["count"] == 2


def test_negation_facet_rejects_unverified_hit_even_when_judge_available() -> None:
    def allow(_q: str, _e, _c) -> str | None:
        return None

    dim = embedding_dim_for_openai_model("text-embedding-3-small")
    pipeline = SemanticCachePipeline(
        router=train_default_classifier(build_routing_dataset()),
        exact_cache=ExactCache(),
        semantic_store=SemanticStore(dimension=dim),
        embedder=make_constant_unit_embedder(dimension=dim),
        neighbor_judge=allow,
    )
    context = RequestContext(semantic_threshold=0.55, neighbor_judge_similarity_ceiling=0.0)

    def live_answer(query: str, _: RequestContext) -> dict[str, object]:
        return {"answer": f"ok:{query}", "success": True}

    pipeline.answer_query("find vegan, low sodium, snack box, no peanuts", context, live_answer)
    second = pipeline.answer_query("find vegan, low sodium, snack box, with peanuts", context, live_answer)

    assert second.source == "live"
    assert second.trace.semantic_facet_conflict_reason == "query_facet_negation_conflict"
    assert second.trace.semantic_post_ann_reject_reason == "query_facet_negation_conflict"
    assert second.trace.neighbor_judge_invoked is False


def test_second_candidate_similarity_on_trace() -> None:
    pipeline = _build_pipeline()
    context = RequestContext(semantic_threshold=0.55)

    def live_answer(query: str, _: RequestContext) -> dict[str, object]:
        return {"answer": "ok", "success": True}

    pipeline.answer_query("Explain what semantic caching is", context, live_answer)
    second = pipeline.answer_query("What is semantic caching?", context, live_answer)
    assert second.trace.second_candidate_similarity is None
