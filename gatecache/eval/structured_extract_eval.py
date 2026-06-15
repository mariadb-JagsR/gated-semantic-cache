"""Structured-extraction evaluation: legacy corpora coverage + programmatic (non hand-tuned) pairs."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from gatecache.eval.legacy_bridge import unique_messages_for_structured_eval
from gatecache.serving.structured_exact import extract_structured_constraints
from gatecache.structured_exact.canonical_key import build_structured_key


@dataclass(frozen=True, slots=True)
class StructuredExactPair:
    category: str
    left_query: str
    right_query: str
    should_match: bool
    notes: str


@dataclass(slots=True)
class StructuredCoverageReport:
    """Aggregate stats over a message list (e.g. legacy IBM + ablation + novel)."""

    message_count: int
    with_canonical_key: int
    with_canonical_key_rate: float
    mean_confidence: float
    ambiguity_rate: float
    constraint_kind_histogram: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_structured_coverage_on_messages(
    messages: list[str],
    *,
    namespace: str = "default",
) -> StructuredCoverageReport:
    if not messages:
        return StructuredCoverageReport(
            message_count=0,
            with_canonical_key=0,
            with_canonical_key_rate=0.0,
            mean_confidence=0.0,
            ambiguity_rate=0.0,
            constraint_kind_histogram={},
        )
    keys = 0
    conf_sum = 0.0
    amb = 0
    kinds: Counter[str] = Counter()
    for msg in messages:
        sq = extract_structured_constraints(msg)
        conf_sum += sq.confidence
        if sq.ambiguity_flags:
            amb += 1
        for c in sq.constraints:
            kinds[c.kind] += 1
        if build_structured_key(sq, namespace=namespace) is not None:
            keys += 1
    return StructuredCoverageReport(
        message_count=len(messages),
        with_canonical_key=keys,
        with_canonical_key_rate=round(keys / len(messages), 4),
        mean_confidence=round(conf_sum / len(messages), 4),
        ambiguity_rate=round(amb / len(messages), 4),
        constraint_kind_histogram=dict(kinds.most_common(50)),
    )


def run_legacy_structured_coverage() -> StructuredCoverageReport:
    return run_structured_coverage_on_messages(unique_messages_for_structured_eval())


def build_novel_structured_exact_pairs() -> list[StructuredExactPair]:
    """Template-generated pairs (generic wording + numeric dims); not copied from product examples.

    Uses only patterns the shared extractors already support (e.g. WxH + `under N usd` / `at most N usd`).
    """
    pairs: list[StructuredExactPair] = []
    triples = ((12, 40, 80), (21, 33, 150), (8, 9, 25), (15, 15, 200), (30, 22, 95))
    for w, h, price in triples:
        left = f"show modular units {w}x{h} under {price} usd"
        right = f"modular units sized {w}x{h} costing at most {price} usd"
        pairs.append(
            StructuredExactPair(
                category="novel_template_paraphrase",
                left_query=left,
                right_query=right,
                should_match=True,
                notes="programmatic WxH + price paraphrase",
            )
        )
        shifted = f"show modular units {w}x{h + 1} under {price} usd"
        pairs.append(
            StructuredExactPair(
                category="novel_template_dimension_shift",
                left_query=left,
                right_query=shifted,
                should_match=False,
                notes="programmatic single-dimension change",
            )
        )
    return pairs


def structured_canonical_key_fingerprint(query: str, *, namespace: str = "default") -> str | None:
    """Stable short hash for logs/tests; None if structured key cannot be built."""
    sq = extract_structured_constraints(query)
    ckey = build_structured_key(sq, namespace=namespace)
    if ckey is None:
        return None
    return hashlib.sha256(ckey.encode("utf-8")).hexdigest()[:24]


__all__ = [
    "StructuredExactPair",
    "StructuredCoverageReport",
    "build_novel_structured_exact_pairs",
    "run_legacy_structured_coverage",
    "run_structured_coverage_on_messages",
    "structured_canonical_key_fingerprint",
]
