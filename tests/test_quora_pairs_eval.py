from pathlib import Path

import pytest

from gatecache.eval.cache_hit_metrics import compute_cache_hit_metrics
from gatecache.eval.quora_pairs_eval import (
    _vector_only_context,
    _vector_only_lookup,
    default_quora_report_path,
    load_quora_pairs,
    run_quora_pairs_eval,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "quora_sample.tsv"


def test_load_quora_pairs_from_fixture() -> None:
    pairs = load_quora_pairs(FIXTURE_PATH)
    assert len(pairs) == 4
    assert sum(1 for pair in pairs if pair.is_duplicate) == 2
    assert pairs[0].question1.startswith("What is the step")


def test_compute_cache_hit_metrics() -> None:
    metrics = compute_cache_hit_metrics(
        expected_hits=[True, True, False, False],
        actual_hits=[True, False, False, True],
    )
    assert metrics.total_pairs == 4
    assert metrics.correct_hits == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.true_negatives == 1
    assert metrics.precision_hit == 0.5
    assert metrics.recall_hit == 0.5
    assert metrics.false_positive_rate == 0.5


def test_default_quora_report_path_is_under_docs() -> None:
    path = default_quora_report_path(
        limit=200,
        seed=42,
        judge_enabled=True,
        route_policy="semantic_ok",
        semantic_threshold=0.86,
        run_at=__import__("datetime").datetime(2026, 5, 24, 12, 0, 0, tzinfo=__import__("datetime").timezone.utc),
    )
    assert path.name.startswith("quora_pairs_limit200_seed42_judge-on_semantic_ok_thresh0p86_")
    assert "quora_pairs_eval" in str(path)


def test_default_quora_report_path_vector_only() -> None:
    path = default_quora_report_path(
        limit=200,
        seed=42,
        judge_enabled=False,
        route_policy="vector_only",
        semantic_threshold=0.86,
        run_at=__import__("datetime").datetime(2026, 5, 24, 12, 0, 0, tzinfo=__import__("datetime").timezone.utc),
    )
    assert "vector-only_vector_only" in path.name


def test_vector_only_lookup_hits_at_threshold() -> None:
    from gatecache.cli import build_default_pipeline
    from gatecache.models.context import RequestContext

    pipeline = build_default_pipeline()
    pipeline.router.predict = lambda _: __import__(
        "gatecache.routing.classifier", fromlist=["RoutingPrediction"]
    ).RoutingPrediction(
        label=__import__("gatecache.routing.labels", fromlist=["RoutingLabel"]).RoutingLabel.SEMANTIC_OK,
        confidence=1.0,
        probabilities={},
    )
    context = RequestContext(semantic_threshold=0.0, semantic_low_watermark=0.0)
    pipeline.answer_query("What is Python?", context, lambda q, _: {"answer": f"seed:{q}", "success": True})
    response = _vector_only_lookup(pipeline, "What is Python?", context)
    assert response.source == "semantic_cache"

    lookup_context = _vector_only_context(RequestContext(semantic_threshold=0.86))
    assert lookup_context.semantic_low_watermark == 0.86


def test_write_quora_pair_report_roundtrip(tmp_path: Path) -> None:
    from gatecache.eval.cache_hit_metrics import CacheHitMetrics
    from gatecache.eval.quora_pairs_eval import QuoraPairEvalReport, load_quora_pair_report, write_quora_pair_report

    metrics = CacheHitMetrics(
        total_pairs=2,
        duplicate_pairs=1,
        non_duplicate_pairs=1,
        total_hits=1,
        correct_hits=1,
        false_positives=0,
        false_negatives=0,
        true_negatives=1,
        routing_blocked=0,
    )
    report = QuoraPairEvalReport(
        dataset_path="/tmp/quora.tsv",
        model="text-embedding-3-small",
        semantic_threshold=0.86,
        semantic_low_watermark=0.7,
        route_policy="semantic_ok",
        limit=2,
        seed=42,
        metrics=metrics,
        rows=[],
    )
    out = tmp_path / "report.json"
    write_quora_pair_report(report, out, judge_enabled=False, balanced=True)
    loaded = load_quora_pair_report(out)
    assert loaded["eval_metadata"]["judge_enabled"] is False
    assert loaded["metrics"]["precision_hit"] == 1.0


@pytest.mark.skipif(
    __import__("os").environ.get("RUN_QUORA_PAIRS_EVAL") != "1",
    reason="OpenAI-backed Quora eval; set RUN_QUORA_PAIRS_EVAL=1 to run intentionally.",
)
def test_quora_pairs_eval_smoke() -> None:
    from gatecache.cli import _load_dotenv_files

    _load_dotenv_files()
    if not __import__("os").environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required for the Quora pairs eval")

    report = run_quora_pairs_eval(
        dataset_path=FIXTURE_PATH,
        limit=4,
        balanced=False,
        route_policy="semantic_ok",
        neighbor_judge=None,
        progress_every=0,
    )
    assert report.metrics.total_pairs == 4
    assert report.metrics.duplicate_pairs == 2
    assert report.metrics.non_duplicate_pairs == 2
