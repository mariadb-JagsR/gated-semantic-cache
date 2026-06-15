"""Embedding backends for semantic ANN (OpenAI API or offline fake for tests)."""

from gated_semantic_cache.embeddings.backends import (
    caching_embedder,
    embedding_dim_for_openai_model,
    make_constant_unit_embedder,
    make_offline_fake_embedder,
    make_openai_embedder,
)

__all__ = [
    "caching_embedder",
    "embedding_dim_for_openai_model",
    "make_constant_unit_embedder",
    "make_offline_fake_embedder",
    "make_openai_embedder",
]
