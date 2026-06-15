import os

from openai import APIConnectionError
import pytest

from gated_semantic_cache.cli import _load_dotenv_files
from gated_semantic_cache.eval.adversarial_cache_eval import (
    build_adversarial_cache_scenarios,
    run_adversarial_cache_eval,
)


def test_adversarial_cache_eval_dataset_shape() -> None:
    scenarios = build_adversarial_cache_scenarios()

    assert len(scenarios) == 11
    assert sum(len(s.tests) for s in scenarios) == 53
    assert any(test.cache_hit for scenario in scenarios for test in scenario.tests)
    assert any(not test.cache_hit for scenario in scenarios for test in scenario.tests)
    for scenario in scenarios:
        assert scenario.scenario.strip()
        assert scenario.cached_query.strip()
        for test in scenario.tests:
            assert test.query.strip()
            assert test.note.strip()


@pytest.mark.skipif(
    os.environ.get("RUN_ADVERSARIAL_CACHE_EVAL") != "1",
    reason="OpenAI-backed holdout eval; set RUN_ADVERSARIAL_CACHE_EVAL=1 to run intentionally.",
)
def test_adversarial_cache_eval_expected_behavior() -> None:
    _load_dotenv_files()
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required for the adversarial cache eval")

    try:
        report = run_adversarial_cache_eval()
    except APIConnectionError as exc:
        pytest.skip(f"OpenAI API is not reachable from this test environment: {exc}")

    assert report.failed == 0, {
        "passed": report.passed,
        "total": report.total,
        "failures": [
            {
                "scenario": row.scenario,
                "query": row.query,
                "expected": row.expected_cache_hit,
                "actual": row.actual_cache_hit,
                "source": row.source,
                "routing_label": row.routing_label,
                "similarity": row.top_candidate_similarity,
                "rejected_reasons": row.rejected_reasons,
                "post_ann_reject": row.semantic_post_ann_reject_reason,
                "facet_conflict": row.semantic_facet_conflict_reason,
                "constraint_risk": row.semantic_constraint_risk_reason,
            }
            for row in report.rows
            if not row.passed
        ],
    }
