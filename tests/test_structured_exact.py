from gated_semantic_cache.eval.structured_exact_benchmark import run_structured_exact_benchmark
from gated_semantic_cache.serving.structured_exact import (
    build_canonical_structured_key,
    critical_constraints_match,
    extract_structured_constraints,
)


def test_structured_exact_matches_paraphrased_same_constraints() -> None:
    left = extract_structured_constraints("show me all pants size 32 x 32 cotton stretch under $40 brown")
    right = extract_structured_constraints("show me brown stretch cotton pants 32x32 for less than 40 dollars")

    assert build_canonical_structured_key(left) == build_canonical_structured_key(right)
    assert critical_constraints_match(left, right) is True


def test_structured_exact_rejects_changed_dimension() -> None:
    left = extract_structured_constraints("show me all pants size 32x32 cotton stretch under $40 brown")
    right = extract_structured_constraints("show me all pants size 32x34 cotton stretch under $40 brown")

    assert build_canonical_structured_key(left) != build_canonical_structured_key(right)
    assert critical_constraints_match(left, right) is False


def test_structured_exact_rejects_changed_quantity() -> None:
    left = extract_structured_constraints(
        "find me nonstop flights from sfo to tokyo in november with premium economy and 1 checked bag"
    )
    right = extract_structured_constraints(
        "find me nonstop flights from sfo to tokyo in november with premium economy and 2 checked bags"
    )

    assert critical_constraints_match(left, right) is False


def test_structured_exact_benchmark_meets_basic_poc_targets() -> None:
    report = run_structured_exact_benchmark()

    assert report.total_pairs >= 8
    assert report.positive_pair_match_rate >= 0.6
    assert report.negative_pair_rejection_rate >= 0.8
