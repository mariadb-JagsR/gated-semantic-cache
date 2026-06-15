from __future__ import annotations

import re

from gated_semantic_cache.structured_exact.schema import Constraint


COLORS = {"black", "brown", "blue", "red", "green", "white", "gray", "grey"}
MATERIALS = {"cotton", "linen", "wool", "leather", "polyester", "denim"}
ATTRIBUTES = {
    "breakfast included",
    "doesn't fade",
    "made in america",
    "made in us",
    "made in usa",
    "noise cancelling",
    "nonstop",
    "organic",
    "premium economy",
    "refundable",
    "usb-c",
    "waterproof",
    "wireless",
}


def extract_categoricals(text: str) -> list[Constraint]:
    tokens = set(re.findall(r"[a-z0-9-]+", text.lower()))
    constraints: list[Constraint] = []

    for color in COLORS & tokens:
        constraints.append(Constraint(kind="categorical", name="color", value=color, confidence=0.98, span_text=color))
    for material in MATERIALS & tokens:
        constraints.append(
            Constraint(kind="categorical", name="material", value=material, confidence=0.98, span_text=material)
        )
    for attribute in _present_attributes(text):
        constraints.append(
            Constraint(kind="categorical", name="attribute", value=attribute, confidence=0.85, span_text=attribute)
        )
    return constraints


def _present_attributes(text: str) -> set[str]:
    present: set[str] = set()
    for attribute in ATTRIBUTES:
        if attribute in text:
            present.add(attribute)
    return present
