from __future__ import annotations

from enum import StrEnum


class RoutingLabel(StrEnum):
    SEMANTIC_OK = "SEMANTIC_OK"
    SKIP_CACHE = "SKIP_CACHE"
    EXACT_ONLY = "EXACT_ONLY"
    THREAD_SCOPED_ONLY = "THREAD_SCOPED_ONLY"


ALL_ROUTING_LABELS = tuple(label.value for label in RoutingLabel)


def parse_routing_label(value: str | RoutingLabel) -> RoutingLabel:
    if isinstance(value, RoutingLabel):
        return value
    return RoutingLabel(value)
