from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CacheHitMetrics:
    total_pairs: int
    duplicate_pairs: int
    non_duplicate_pairs: int
    total_hits: int
    correct_hits: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    routing_blocked: int

    @property
    def precision_hit(self) -> float:
        return (self.correct_hits / self.total_hits) if self.total_hits else 0.0

    @property
    def recall_hit(self) -> float:
        return (self.correct_hits / self.duplicate_pairs) if self.duplicate_pairs else 0.0

    @property
    def false_positive_rate(self) -> float:
        return (self.false_positives / self.non_duplicate_pairs) if self.non_duplicate_pairs else 0.0

    @property
    def wrong_cache_answer_rate(self) -> float:
        return ((self.total_hits - self.correct_hits) / self.total_hits) if self.total_hits else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total_pairs": self.total_pairs,
            "duplicate_pairs": self.duplicate_pairs,
            "non_duplicate_pairs": self.non_duplicate_pairs,
            "total_hits": self.total_hits,
            "correct_hits": self.correct_hits,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "routing_blocked": self.routing_blocked,
            "precision_hit": round(self.precision_hit, 4),
            "recall_hit": round(self.recall_hit, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "wrong_cache_answer_rate": round(self.wrong_cache_answer_rate, 4),
        }


def compute_cache_hit_metrics(
    *,
    expected_hits: list[bool],
    actual_hits: list[bool],
    routing_blocked: list[bool] | None = None,
) -> CacheHitMetrics:
    if len(expected_hits) != len(actual_hits):
        raise ValueError("expected_hits and actual_hits must have the same length")
    blocked = routing_blocked or [False] * len(expected_hits)
    if len(blocked) != len(expected_hits):
        raise ValueError("routing_blocked must match expected_hits length")

    duplicate_pairs = 0
    non_duplicate_pairs = 0
    total_hits = 0
    correct_hits = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    routing_blocked_count = 0

    for expected, actual, is_blocked in zip(expected_hits, actual_hits, blocked, strict=True):
        if expected:
            duplicate_pairs += 1
        else:
            non_duplicate_pairs += 1
        if is_blocked:
            routing_blocked_count += 1

        if actual:
            total_hits += 1
            if expected:
                correct_hits += 1
            else:
                false_positives += 1
        elif expected:
            false_negatives += 1
        else:
            true_negatives += 1

    return CacheHitMetrics(
        total_pairs=len(expected_hits),
        duplicate_pairs=duplicate_pairs,
        non_duplicate_pairs=non_duplicate_pairs,
        total_hits=total_hits,
        correct_hits=correct_hits,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_negatives=true_negatives,
        routing_blocked=routing_blocked_count,
    )
