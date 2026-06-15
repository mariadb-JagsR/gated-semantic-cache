from __future__ import annotations

from gated_semantic_cache.models.context import RequestContext
from gated_semantic_cache.routing.labels import RoutingLabel
from gated_semantic_cache.serving.policy import build_anchor_key


def resolve_effective_route_label(
    *,
    query: str,
    context: RequestContext,
    predicted_label: RoutingLabel,
    routing_confidence: float,
) -> tuple[RoutingLabel, str | None, bool]:
    """Apply post-classifier routing adjustments.

    Returns ``(effective_label, anchor_key, exact_only_downgraded)``.
    """

    if predicted_label is not RoutingLabel.EXACT_ONLY:
        return predicted_label, None, False

    anchor_key = build_anchor_key(query, context)
    if anchor_key is not None:
        return RoutingLabel.EXACT_ONLY, anchor_key, False

    floor = context.exact_only_min_route_confidence
    if floor is not None and routing_confidence < floor:
        return RoutingLabel.SEMANTIC_OK, None, True

    return RoutingLabel.EXACT_ONLY, None, False
