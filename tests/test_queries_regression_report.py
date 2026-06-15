from pathlib import Path

from gated_semantic_cache.eval.queries_regression import (
    _default_queries_path,
    run_queries_regression_report,
)


def test_queries_regression_routing_report_smoke() -> None:
    path = _default_queries_path()
    assert path.is_file(), f"missing {path}"
    report = run_queries_regression_report(mode="routing", queries_file=path)
    assert report.mode == "routing"
    assert len(report.rows) >= 10
    assert all(r.routing_label for r in report.rows)


def test_queries_regression_accepts_classifier_pkl(tmp_path: Path) -> None:
    from gated_semantic_cache.eval.datasets import build_routing_dataset
    from gated_semantic_cache.routing.classifier import train_default_classifier

    pkl = tmp_path / "router.pkl"
    train_default_classifier(build_routing_dataset()).save(pkl)
    report = run_queries_regression_report(
        mode="routing",
        queries=["What is semantic caching?"],
        classifier_path=pkl,
    )
    assert report.rows[0].routing_label == "SEMANTIC_OK"
