from gatecache.models.context import RequestContext
from gatecache.routing.labels import RoutingLabel
from gatecache.serving.routing_resolve import resolve_effective_route_label


def test_low_confidence_exact_only_without_anchor_downgrades() -> None:
    label, anchor, downgraded = resolve_effective_route_label(
        query="details on deploying a node.js app to gcp",
        context=RequestContext(exact_only_min_route_confidence=0.55),
        predicted_label=RoutingLabel.EXACT_ONLY,
        routing_confidence=0.47,
    )

    assert label is RoutingLabel.SEMANTIC_OK
    assert anchor is None
    assert downgraded is True


def test_high_confidence_exact_only_without_anchor_stays_exact_only() -> None:
    label, anchor, downgraded = resolve_effective_route_label(
        query="international wire transfer fees for a business account",
        context=RequestContext(exact_only_min_route_confidence=0.55),
        predicted_label=RoutingLabel.EXACT_ONLY,
        routing_confidence=0.9,
    )

    assert label is RoutingLabel.EXACT_ONLY
    assert anchor is None
    assert downgraded is False


def test_exact_only_with_anchor_stays_exact_only() -> None:
    label, anchor, downgraded = resolve_effective_route_label(
        query="Lookup order #A123 status",
        context=RequestContext(exact_only_min_route_confidence=0.55),
        predicted_label=RoutingLabel.EXACT_ONLY,
        routing_confidence=0.35,
    )

    assert label is RoutingLabel.EXACT_ONLY
    assert anchor is not None
    assert downgraded is False
