from __future__ import annotations

import contextvars
from typing import Any, Callable, Protocol

from gatecache.models.cache_entry import SemanticCacheEntry
from gatecache.models.context import RequestContext

_last_judge_observation: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "neighbor_judge_last_observation",
    default=None,
)


def set_neighbor_judge_observation(observation: dict[str, Any] | None) -> None:
    _last_judge_observation.set(observation)


def take_neighbor_judge_observation() -> dict[str, Any] | None:
    observation = _last_judge_observation.get(None)
    _last_judge_observation.set(None)
    return observation

# Post-retrieval policy hook: return None to allow reuse, or a short machine-readable reason to reject.
NeighborJudge = Callable[[str, SemanticCacheEntry, RequestContext], str | None]


class NeighborJudgeProtocol(Protocol):
    def __call__(self, query: str, entry: SemanticCacheEntry, context: RequestContext) -> str | None: ...


def noop_allow_neighbor_judge(_query: str, _entry: SemanticCacheEntry, _context: RequestContext) -> str | None:
    """Always allow reuse. Useful to validate wiring or to combine with gray-zone gating only."""

    return None


def rejecting_neighbor_judge(reason: str) -> NeighborJudge:
    """Build a judge that always rejects with a fixed machine-readable reason (tests / policies)."""

    def _inner(_query: str, _entry: SemanticCacheEntry, _context: RequestContext) -> str | None:
        return reason

    return _inner


# --- Product notes (cost / latency / when it runs) ---
#
# The neighbor judge is an *optional* escalation after ANN + deterministic gates (namespace, TTL,
# structured reuse, etc.). It is off by default (pipeline receives ``neighbor_judge=None``).
#
# When enabled, prefer **gray-zone gating** via ``RequestContext.neighbor_judge_similarity_ceiling``:
# similarities at or above the ceiling skip the judge (treat as a strong embedding match); the judge
# only runs for ``semantic_threshold <= sim < ceiling`` unless skipped by ambiguity margin or call caps.
#
# Cost: each invocation is typically an extra LLM round-trip (latency + tokens), unless you inject a
# cheap heuristic or rule-based implementation. Tune ceilings and ``neighbor_judge_max_calls`` to cap
# worst-case spend per request.
