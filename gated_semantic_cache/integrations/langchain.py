"""LangChain ``BaseCache`` adapter for GatedSemanticCache.

Drop-in for ``langchain_core.globals.set_llm_cache`` — the app sets the cache once and
every LLM call is routed through the full gated pipeline (routing, structured match,
facet gates, optional gray-zone judge) with a live-answer fallback on every miss.

    from langchain_core.globals import set_llm_cache
    from gated_semantic_cache.integrations.langchain import GatedLangChainCache

    set_llm_cache(GatedLangChainCache.from_sqlite(
        db_path=".gated-semantic-cache/cache.sqlite3",
        namespace="product-support",
    ))

Requires the ``langchain`` extra::

    pip install "gated-semantic-cache[langchain]"
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gated_semantic_cache.api import JudgePolicy, PutPolicy, SemanticCache, SemanticMode

try:
    from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
    from langchain_core.load import dumps, loads
    from langchain_core.outputs import Generation
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "GatedLangChainCache requires langchain-core. "
        'Install it with: pip install "gated-semantic-cache[langchain]"'
    ) from exc

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

# Default cosine floor when neither the constructor nor SEMANTIC_THRESHOLD env var is set.
_DEFAULT_SIMILARITY_THRESHOLD = 0.86


class GatedLangChainCache(BaseCache):
    """Adapts :class:`~gated_semantic_cache.SemanticCache` to LangChain's ``BaseCache``.

    LangChain's ``lookup``/``update`` take no per-call tuning knobs, so the behavior an app
    would otherwise pass to ``SemanticCache.get``/``put`` is fixed here at construction:
    ``namespace``, ``similarity_threshold``, the LLM judge on/off, ``ttl_seconds`` and
    ``semantic_mode``. Judge internals (model, timeout, similarity ceiling, ambiguity margin,
    max calls) continue to come from the environment exactly as for ``SemanticCache``.

    ``llm_string`` (LangChain's serialized model + params) is ignored by default so the same
    question reuses across models — the cached answer is about the question, not the model that
    produced it. Set ``isolate_by_llm=True`` to scope entries per model config instead.
    """

    def __init__(
        self,
        *,
        cache: SemanticCache,
        similarity_threshold: float | None = None,
        ttl_seconds: int | None = 3600,
        semantic_mode: SemanticMode = "always",
        judge_policy: JudgePolicy | None = None,
        isolate_by_llm: bool = False,
    ) -> None:
        self._cache = cache
        self._judge_policy = judge_policy or cache.default_judge_policy
        self._similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self._judge_policy.similarity_floor
        )
        self._ttl_seconds = ttl_seconds
        self._semantic_mode = semantic_mode
        self._isolate_by_llm = isolate_by_llm

    @classmethod
    def from_sqlite(
        cls,
        *,
        db_path: Path | str,
        namespace: str,
        similarity_threshold: float | None = None,
        enable_llm_judge: bool = True,
        ttl_seconds: int | None = 3600,
        semantic_mode: SemanticMode = "always",
        isolate_by_llm: bool = False,
        embeddings: Embeddings | None = None,
        embedder: Callable[[str], list[float]] | None = None,
        openai_model: str = "text-embedding-3-small",
        openai_api_key: str | None = None,
        openai_dimensions: int | None = None,
    ) -> GatedLangChainCache:
        """Build a durable SQLite-backed gated cache wired for LangChain.

        Pass a LangChain ``embeddings`` object (Redis-style) to reuse the app's embedder, or a raw
        ``embedder`` callable; otherwise OpenAI embeddings are built from ``openai_*`` / the
        environment. ``similarity_threshold`` falls back to ``$SEMANTIC_THRESHOLD`` then 0.86.
        """
        threshold = _resolve_threshold(similarity_threshold)
        judge_policy = JudgePolicy(enabled=enable_llm_judge, similarity_floor=threshold)
        resolved_embedder = embedder
        if resolved_embedder is None and embeddings is not None:
            resolved_embedder = embeddings.embed_query
        cache = SemanticCache.from_sqlite(
            db_path=db_path,
            namespace=namespace,
            openai_model=openai_model,
            openai_api_key=openai_api_key,
            openai_dimensions=openai_dimensions,
            embedder=resolved_embedder,
            default_judge_policy=judge_policy,
            use_default_llm_judge=enable_llm_judge,
        )
        return cls(
            cache=cache,
            similarity_threshold=threshold,
            ttl_seconds=ttl_seconds,
            semantic_mode=semantic_mode,
            judge_policy=judge_policy,
            isolate_by_llm=isolate_by_llm,
        )

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        hit = self._cache.get(
            prompt,
            scope_keys=self._scope_keys(llm_string),
            semantic_mode=self._semantic_mode,
            judge_policy=self._judge_policy,
            semantic_threshold=self._similarity_threshold,
        )
        if hit is None:
            return None
        return _generations_from_payload(hit.payload)

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        self._cache.put(
            prompt,
            _payload_from_generations(return_val),
            scope_keys=self._scope_keys(llm_string),
            policy=PutPolicy(semantic_mode=self._semantic_mode, ttl_seconds=self._ttl_seconds),
        )

    def clear(self, **kwargs: Any) -> None:
        self._cache.clear()

    def close(self) -> None:
        self._cache.close()

    def _scope_keys(self, llm_string: str) -> dict[str, str] | None:
        return {"llm": llm_string} if self._isolate_by_llm else None


def _resolve_threshold(similarity_threshold: float | None) -> float:
    if similarity_threshold is not None:
        return similarity_threshold
    env = os.getenv("SEMANTIC_THRESHOLD")
    return float(env) if env else _DEFAULT_SIMILARITY_THRESHOLD


def _payload_from_generations(return_val: RETURN_VAL_TYPE) -> dict[str, Any]:
    """Serialize generations losslessly; keep an ``answer`` key for response previews."""
    return {
        "generations": [dumps(g) for g in return_val],
        "answer": return_val[0].text if return_val else "",
    }


def _generations_from_payload(payload: dict[str, Any]) -> list[Generation]:
    serialized = payload.get("generations")
    if serialized:
        return [loads(g) for g in serialized]
    # Payload written outside this adapter (e.g. a direct SemanticCache.put): synthesize a Generation.
    return [Generation(text=str(payload.get("answer", "")))]
