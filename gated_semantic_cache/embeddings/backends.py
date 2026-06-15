from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from collections.abc import Callable

import numpy as np

# Default OpenAI model dimensions (full-size vectors; faiss uses cosine via normalized IP).
_OPENAI_MODEL_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def embedding_dim_for_openai_model(model: str, *, dimensions: int | None = None) -> int:
    if dimensions is not None:
        return dimensions
    if model in _OPENAI_MODEL_DIMS:
        return _OPENAI_MODEL_DIMS[model]
    # Unknown model: caller should pass explicit dimension if needed; default to small.
    return _OPENAI_MODEL_DIMS["text-embedding-3-small"]


def make_constant_unit_embedder(*, dimension: int) -> Callable[[str], list[float]]:
    """Test-only: same L2-normalized vector for every input (cosine similarity 1.0 vs any stored row).

    Use in unit tests to exercise semantic-cache paths without an embedding API. Not semantically meaningful.
    """
    v = np.zeros(dimension, dtype=np.float64)
    v[0] = 1.0
    out = (v / (np.linalg.norm(v) + 1e-12)).astype(np.float64).tolist()

    def embed(_text: str) -> list[float]:
        return list(out)

    return embed


def make_offline_fake_embedder(*, dimension: int | None = None) -> Callable[[str], list[float]]:
    """Deterministic pseudo-embeddings for unit tests and CI without API calls.

    Each query maps to a L2-normalized vector derived from SHA-256 (not a semantic model).
    Use :func:`make_openai_embedder` in production.
    """
    dim = dimension if dimension is not None else _OPENAI_MODEL_DIMS["text-embedding-3-small"]

    def embed(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(dim).astype(np.float64)
        n = float(np.linalg.norm(v)) + 1e-12
        return (v / n).tolist()

    return embed


def caching_embedder(
    inner: Callable[[str], list[float]],
    *,
    max_entries: int = 2048,
) -> Callable[[str], list[float]]:
    """LRU-ish embedding cache keyed by exact input string (normalized query text upstream)."""
    cache: OrderedDict[str, list[float]] = OrderedDict()

    def embed(text: str) -> list[float]:
        if text in cache:
            cache.move_to_end(text)
            return cache[text]
        vec = inner(text)
        cache[text] = vec
        cache.move_to_end(text)
        while len(cache) > max_entries:
            cache.popitem(last=False)
        return vec

    return embed


def make_openai_embedder(
    *,
    model: str = "text-embedding-3-small",
    api_key: str | None = None,
    dimensions: int | None = None,
) -> Callable[[str], list[float]]:
    """Embeddings via OpenAI HTTP API (requires ``openai`` package and ``OPENAI_API_KEY``).

    ``dimensions`` (for ``text-embedding-3-*``) trades vector size vs fidelity; index dimension must match.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        msg = "OpenAI embeddings require OPENAI_API_KEY (or pass api_key= to make_openai_embedder)."
        raise RuntimeError(msg)

    from openai import OpenAI

    client = OpenAI(api_key=key)

    def embed(text: str) -> list[float]:
        kwargs: dict[str, object] = {"model": model, "input": text}
        if dimensions is not None:
            kwargs["dimensions"] = dimensions
        response = client.embeddings.create(**kwargs)
        vec = response.data[0].embedding
        return [float(x) for x in vec]

    return embed
