from gatecache import JudgePolicy, PutPolicy, SemanticCache
from gatecache.cache.exact_cache import ExactCache
from gatecache.cache.semantic_store import SemanticStore
from gatecache.embeddings.backends import embedding_dim_for_openai_model, make_constant_unit_embedder
from gatecache.eval.datasets import build_routing_dataset
from gatecache.routing.classifier import train_default_classifier
from gatecache.serving.neighbor_judge import noop_allow_neighbor_judge


def _cache(
    namespace: str = "product-support",
    *,
    judge=noop_allow_neighbor_judge,
    default_judge_policy: JudgePolicy | None = None,
) -> SemanticCache:
    dim = embedding_dim_for_openai_model("text-embedding-3-small")
    return SemanticCache.from_components(
        namespace=namespace,
        router=train_default_classifier(build_routing_dataset()),
        exact_cache=ExactCache(),
        semantic_store=SemanticStore(dimension=dim),
        embedder=make_constant_unit_embedder(dimension=dim),
        neighbor_judge=judge,
        default_judge_policy=default_judge_policy,
        use_default_llm_judge=False,
    )


def test_get_put_exact_with_namespace_only() -> None:
    cache = _cache()

    ref = cache.put(
        "Does the product support namespace isolation?",
        {"answer": "Yes", "success": True},
        policy=PutPolicy(semantic_mode="never"),
    )
    hit = cache.get(
        "Does the product support namespace isolation?",
        semantic_mode="never",
    )

    assert ref.namespace == "product-support"
    assert ref.semantic_indexed is False
    assert hit is not None
    assert hit.source == "exact_cache"
    assert hit.payload["answer"] == "Yes"


def test_scope_keys_isolate_exact_entries() -> None:
    cache = _cache(namespace="bank-online")

    cache.put(
        "find the last 5 transactions in Jack's account",
        {"answer": "jack transactions", "success": True},
        scope_keys={"principal_id": "jack", "account_id": "acct-jack"},
        policy=PutPolicy(semantic_mode="never"),
    )

    assert (
        cache.get(
            "find the last 5 transactions in Jack's account",
            scope_keys={"principal_id": "sam", "account_id": "acct-sam"},
            semantic_mode="never",
        )
        is None
    )
    hit = cache.get(
        "find the last 5 transactions in Jack's account",
        scope_keys={"principal_id": "jack", "account_id": "acct-jack"},
        semantic_mode="never",
    )
    assert hit is not None
    assert hit.payload["answer"] == "jack transactions"


def test_namespace_isolates_entries() -> None:
    cache = _cache(namespace="product-a")
    cache.put(
        "What is semantic caching?",
        {"answer": "product a answer", "success": True},
        policy=PutPolicy(semantic_mode="never"),
    )

    assert cache.get("What is semantic caching?", namespace="product-b", semantic_mode="never") is None


def test_semantic_get_uses_configured_judge_by_default() -> None:
    cache = _cache(judge=noop_allow_neighbor_judge)

    cache.put(
        "Explain what semantic caching is",
        {"answer": "reuse prior answers", "success": True},
    )
    hit = cache.get("What is semantic caching?")

    assert hit is not None
    assert hit.source == "semantic_cache"
    assert hit.trace["neighbor_judge_invoked"] is True


def test_default_judge_policy_fails_closed_without_configured_judge() -> None:
    cache = _cache(judge=None)

    cache.put(
        "Explain what semantic caching is",
        {"answer": "reuse prior answers", "success": True},
    )

    assert cache.get("What is semantic caching?") is None


def test_judge_can_be_disabled_for_low_risk_namespace() -> None:
    cache = _cache(judge=None)

    cache.put(
        "Explain what semantic caching is",
        {"answer": "reuse prior answers", "success": True},
    )
    hit = cache.get(
        "What is semantic caching?",
        judge_policy=JudgePolicy(enabled=False),
    )

    assert hit is not None
    assert hit.source == "semantic_cache"
    assert hit.trace["neighbor_judge_invoked"] is False


def test_put_defaults_to_semantic_indexing() -> None:
    cache = _cache(judge=noop_allow_neighbor_judge)

    ref = cache.put(
        "Explain what semantic caching is",
        {"answer": "reuse prior answers", "success": True},
    )
    hit = cache.get("What is semantic caching?")

    assert ref.semantic_indexed is True
    assert hit is not None
    assert hit.source == "semantic_cache"
