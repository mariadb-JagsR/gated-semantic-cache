from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from gatecache.eval.structured_extract_eval import (
    StructuredExactPair,
    build_novel_structured_exact_pairs,
)
from gatecache.serving.structured_exact import (
    build_canonical_structured_key,
    critical_constraints_match,
    extract_structured_constraints,
)


@dataclass(slots=True)
class StructuredExactBenchmarkReport:
    total_pairs: int
    overall_accuracy: float
    positive_pair_match_rate: float
    negative_pair_rejection_rate: float
    extraction_coverage: float
    category_accuracy: dict[str, float]
    failures: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_structured_exact_pairs() -> list[StructuredExactPair]:
    curated: list[StructuredExactPair] = [
        StructuredExactPair(
            category="same_constraints_paraphrase",
            left_query="show me all pants size 32 x 32 cotton stretch under $40 brown",
            right_query="show me brown stretch cotton pants 32x32 for less than 40 dollars",
            should_match=True,
            notes="same constraints, different wording",
        ),
        StructuredExactPair(
            category="dimension_change",
            left_query="show me all pants size 32x32 cotton stretch under $40 brown",
            right_query="show me all pants size 32x34 cotton stretch under $40 brown",
            should_match=False,
            notes="dimension change should force miss",
        ),
        StructuredExactPair(
            category="numeric_bound_change",
            left_query="show me brown cotton pants under $40",
            right_query="show me brown cotton pants under $50",
            should_match=False,
            notes="numeric bound change should force miss",
        ),
        StructuredExactPair(
            category="quantity_change",
            left_query="find me nonstop flights from sfo to tokyo in november with premium economy and 1 checked bag",
            right_query="find me nonstop flights from sfo to tokyo in november with premium economy and 2 checked bags",
            should_match=False,
            notes="checked bag count should force miss",
        ),
        StructuredExactPair(
            category="date_window_change",
            left_query="show me top 5 widget sales in europe last seven days",
            right_query="show me top 5 widget sales in europe last 30 days",
            should_match=False,
            notes="date window should force miss",
        ),
        StructuredExactPair(
            category="binary_flag_change",
            left_query="find me nonstop flights from sfo to tokyo with refundable fares",
            right_query="find me flights from sfo to tokyo with refundable fares",
            should_match=False,
            notes="binary flag difference should force miss",
        ),
        StructuredExactPair(
            category="identifier_paraphrase",
            left_query="lookup order #A123 status",
            right_query="find order A123 status",
            should_match=True,
            notes="anchored exact paraphrase",
        ),
        StructuredExactPair(
            category="incomplete_extraction",
            left_query="show me the one with stretch",
            right_query="show me the one with stretch",
            should_match=False,
            notes="low-information query should not produce reusable exact key",
        ),
        StructuredExactPair(
            category="date_window_paraphrase",
            left_query="show me widget revenue for the last seven days",
            right_query="show me widget revenue for the past week",
            should_match=True,
            notes="date-window paraphrase should normalize to the same critical constraints",
        ),
        StructuredExactPair(
            category="policy_action_difference",
            left_query="what does the cancellation policy say for orders that already shipped?",
            right_query="change the shipping address on my order to tokyo",
            should_match=False,
            notes="policy question vs action request must not match",
        ),
    ]
    return curated + build_novel_structured_exact_pairs()


def run_structured_exact_benchmark(pairs: list[StructuredExactPair] | None = None) -> StructuredExactBenchmarkReport:
    dataset = pairs or build_structured_exact_pairs()
    correct = 0
    matched_positive = 0
    rejected_negative = 0
    positive_total = 0
    negative_total = 0
    extraction_covered = 0
    failures: list[dict[str, object]] = []
    category_success: Counter[str] = Counter()
    category_total: Counter[str] = Counter()

    for pair in dataset:
        left = extract_structured_constraints(pair.left_query)
        right = extract_structured_constraints(pair.right_query)
        left_key = build_canonical_structured_key(left)
        right_key = build_canonical_structured_key(right)
        comparable = left_key is not None and right_key is not None
        if comparable:
            extraction_covered += 1
        matched = comparable and left_key == right_key and critical_constraints_match(left, right)
        expected = pair.should_match
        category_total[pair.category] += 1

        if expected:
            positive_total += 1
            matched_positive += int(matched)
        else:
            negative_total += 1
            rejected_negative += int(not matched)

        if matched == expected:
            correct += 1
            category_success[pair.category] += 1
        else:
            failures.append(
                {
                    "category": pair.category,
                    "left_query": pair.left_query,
                    "right_query": pair.right_query,
                    "should_match": pair.should_match,
                    "matched": matched,
                    "left_key": left_key,
                    "right_key": right_key,
                    "left_constraints": [asdict(item) for item in left.constraints],
                    "right_constraints": [asdict(item) for item in right.constraints],
                    "notes": pair.notes,
                }
            )

    category_accuracy = {
        category: round(category_success[category] / total, 4)
        for category, total in category_total.items()
    }
    return StructuredExactBenchmarkReport(
        total_pairs=len(dataset),
        overall_accuracy=round(correct / len(dataset), 4) if dataset else 0.0,
        positive_pair_match_rate=round(matched_positive / positive_total, 4) if positive_total else 0.0,
        negative_pair_rejection_rate=round(rejected_negative / negative_total, 4) if negative_total else 0.0,
        extraction_coverage=round(extraction_covered / len(dataset), 4) if dataset else 0.0,
        category_accuracy=category_accuracy,
        failures=failures,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the structured exact POC benchmark.")
    parser.add_argument("--report-json", help="Optional path to write benchmark metrics")
    args = parser.parse_args()

    report = run_structured_exact_benchmark()
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report.to_dict(), indent=2))
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
