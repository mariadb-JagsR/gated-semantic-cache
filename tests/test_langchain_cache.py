import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, Generation

from gated_semantic_cache import SemanticCache
from gated_semantic_cache.cache.exact_cache import ExactCache
from gated_semantic_cache.cache.semantic_store import SemanticStore
from gated_semantic_cache.embeddings.backends import (
    embedding_dim_for_openai_model,
    make_constant_unit_embedder,
)
from gated_semantic_cache.eval.datasets import build_routing_dataset
from gated_semantic_cache.integrations.langchain import GatedLangChainCache
from gated_semantic_cache.routing.classifier import train_default_classifier
from gated_semantic_cache.serving.neighbor_judge import noop_allow_neighbor_judge


def _semantic_cache(namespace: str = "lc-test", *, embedder=None) -> SemanticCache:
    dim = embedding_dim_for_openai_model("text-embedding-3-small")
    return SemanticCache.from_components(
        namespace=namespace,
        router=train_default_classifier(build_routing_dataset()),
        exact_cache=ExactCache(),
        semantic_store=SemanticStore(dimension=dim),
        embedder=embedder or make_constant_unit_embedder(dimension=dim),
        neighbor_judge=noop_allow_neighbor_judge,
        use_default_llm_judge=False,
    )


def test_lookup_miss_returns_none() -> None:
    cache = GatedLangChainCache(cache=_semantic_cache())
    assert cache.lookup("anything", "llm-config") is None


def test_update_then_lookup_roundtrips_plain_generation() -> None:
    cache = GatedLangChainCache(cache=_semantic_cache())
    cache.update("What is the capital of France?", "llm-config", [Generation(text="Paris.")])

    hit = cache.lookup("What is the capital of France?", "llm-config")

    assert hit is not None
    assert [g.text for g in hit] == ["Paris."]


def test_chat_generation_serialized_losslessly() -> None:
    cache = GatedLangChainCache(cache=_semantic_cache())
    message = AIMessage(content="Paris.", response_metadata={"finish_reason": "stop"})
    cache.update("capital of France?", "llm-config", [ChatGeneration(message=message)])

    hit = cache.lookup("capital of France?", "llm-config")

    assert hit is not None
    (gen,) = hit
    assert isinstance(gen, ChatGeneration)
    assert isinstance(gen.message, AIMessage)
    assert gen.message.content == "Paris."
    assert gen.message.response_metadata == {"finish_reason": "stop"}


def test_clear_evicts_entries() -> None:
    cache = GatedLangChainCache(cache=_semantic_cache())
    cache.update("ping", "llm-config", [Generation(text="pong")])
    assert cache.lookup("ping", "llm-config") is not None

    cache.clear()

    assert cache.lookup("ping", "llm-config") is None


def test_isolate_by_llm_scopes_entries() -> None:
    cache = GatedLangChainCache(cache=_semantic_cache(), isolate_by_llm=True)
    cache.update("ping", "model-a", [Generation(text="from-a")])

    # Different llm_string => different scope => no reuse.
    assert cache.lookup("ping", "model-b") is None
    same = cache.lookup("ping", "model-a")
    assert same is not None and same[0].text == "from-a"


def test_from_sqlite_persists_and_clears(tmp_path) -> None:
    dim = embedding_dim_for_openai_model("text-embedding-3-small")
    db_path = tmp_path / "cache.sqlite3"
    cache = GatedLangChainCache.from_sqlite(
        db_path=db_path,
        namespace="lc-sqlite",
        enable_llm_judge=False,
        embedder=make_constant_unit_embedder(dimension=dim),
    )
    try:
        cache.update("durable?", "llm-config", [Generation(text="yes")])
        hit = cache.lookup("durable?", "llm-config")
        assert hit is not None and hit[0].text == "yes"

        cache.clear()
        assert cache.lookup("durable?", "llm-config") is None
    finally:
        cache.close()
