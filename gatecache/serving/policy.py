from __future__ import annotations

import hashlib
import re

from gatecache.models.context import RequestContext
from gatecache.routing.features import normalize_query_text
from gatecache.routing.labels import RoutingLabel


ANCHOR_RE = re.compile(
    r"\b(?:order|ticket|incident|customer|account|case|uuid|host)?\s*([a-z]*[-#]?\d+[a-z0-9-]*|[0-9a-f]{8}-[0-9a-f-]{27})\b",
    re.IGNORECASE,
)


def build_exact_key(normalized_query: str, context: RequestContext) -> str:
    parts = [
        normalized_query,
        context.effective_cache_namespace(),
        context.agent_version,
        context.corpus_version or "",
        context.tool_or_schema_version or "",
        repr(context.exact_key_parts()),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def build_anchor_key(query: str, context: RequestContext) -> str | None:
    normalized = normalize_query_text(query)
    match = ANCHOR_RE.search(normalized)
    if not match:
        return None
    anchor = match.group(1)
    return hashlib.sha256(f"{context.effective_cache_namespace()}|{anchor}".encode("utf-8")).hexdigest()


def cache_policy_for_route(route: RoutingLabel) -> str:
    if route is RoutingLabel.SEMANTIC_OK:
        return "semantic_ok"
    if route is RoutingLabel.EXACT_ONLY:
        return "exact_only"
    if route is RoutingLabel.THREAD_SCOPED_ONLY:
        return "thread_scoped"
    return "skip"


def is_policy_compatible(cache_policy_class: str, context: RequestContext) -> bool:
    if cache_policy_class == "thread_scoped":
        return context.thread_scope_key is not None
    return cache_policy_class in {"semantic_ok", "exact_only"}
