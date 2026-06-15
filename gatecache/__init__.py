"""Classifier-first semantic cache redesign package."""

from gatecache.api import CacheEntryRef, CacheHit, JudgePolicy, PutPolicy, SemanticCache
from gatecache.serving.llm_judge import make_openai_neighbor_judge

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
