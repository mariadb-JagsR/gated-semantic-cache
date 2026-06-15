"""GatedSemanticCache — multi-gate semantic cache for LLM applications."""

from gated_semantic_cache.api import CacheEntryRef, CacheHit, JudgePolicy, PutPolicy, SemanticCache
from gated_semantic_cache.serving.llm_judge import make_openai_neighbor_judge

__all__ = [
    "CacheEntryRef",
    "CacheHit",
    "JudgePolicy",
    "PutPolicy",
    "SemanticCache",
    "make_openai_neighbor_judge",
    "__version__",
]

__version__ = "0.1.0"
