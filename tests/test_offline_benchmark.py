from gatecache.eval.datasets import build_routing_dataset
from gatecache.eval.offline_benchmark import analyze_routing_errors, benchmark_routing_classifier


def test_error_analysis_report_has_expected_shape() -> None:
    report = analyze_routing_errors(build_routing_dataset())

    assert report.total_examples >= 100
    assert report.total_misclassified >= 1
    assert report.overall_error_rate > 0.0
    assert report.top_confusions
    assert report.misclassified_examples


def test_benchmark_report_tracks_all_labels() -> None:
    report = benchmark_routing_classifier(build_routing_dataset())

    assert set(report.class_distribution) == {
        "SEMANTIC_OK",
        "SKIP_CACHE",
        "EXACT_ONLY",
        "THREAD_SCOPED_ONLY",
    }
