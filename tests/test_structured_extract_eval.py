from gatecache.eval.structured_extract_eval import (
    build_novel_structured_exact_pairs,
    run_legacy_structured_coverage,
)
from gatecache.eval.structured_exact_benchmark import (
    build_structured_exact_pairs,
    run_structured_exact_benchmark,
)


def test_legacy_structured_coverage_includes_ibm_and_ablation_corpus() -> None:
    report = run_legacy_structured_coverage()
    assert report.message_count >= 150
    assert 0.0 <= report.with_canonical_key_rate <= 1.0
    assert report.mean_confidence >= 0.0


def test_novel_template_pairs_pass_structured_exact_benchmark() -> None:
    novel = build_novel_structured_exact_pairs()
    report = run_structured_exact_benchmark(novel)
    assert report.overall_accuracy == 1.0
    assert report.failures == []


def test_full_structured_exact_benchmark_includes_novel_pairs() -> None:
    pairs = build_structured_exact_pairs()
    assert any(p.category.startswith("novel_template_") for p in pairs)
    report = run_structured_exact_benchmark(pairs)
    assert report.total_pairs == len(pairs)
    assert report.overall_accuracy >= 0.95
