from __future__ import annotations

from gated_semantic_cache.models.context import RequestContext
from gated_semantic_cache.routing.labels import RoutingLabel


def should_insert_response(
    *,
    route_label: RoutingLabel,
    context: RequestContext,
    success: bool,
    is_private: bool = False,
    is_destructive: bool = False,
    is_freshness_sensitive: bool = False,
) -> bool:
    if not success or is_private or is_destructive or is_freshness_sensitive:
        return False
    if route_label is RoutingLabel.SKIP_CACHE:
        return False
    if route_label is RoutingLabel.THREAD_SCOPED_ONLY:
        return context.thread_scope_key is not None
    return True
