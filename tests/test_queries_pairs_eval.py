from pathlib import Path

import pytest

from gated_semantic_cache.eval.queries_pairs_eval import (
    QueryPairScenario,
    load_query_pair_scenarios,
    run_queries_pairs_eval,
)


def test_load_query_pair_scenarios_matches_queries_txt() -> None:
    scenarios = load_query_pair_scenarios()
    assert len(scenarios) == 12
    assert sum(1 for s in scenarios if s.expected_cache_hit) == 2
    assert scenarios[0].pair_id == "lisinopril_paraphrase"
    assert scenarios[-1].pair_id == "trial_polarity"


def test_load_query_pair_scenarios_custom_tuple() -> None:
    custom = (
        QueryPairScenario("a", "seed one", "probe one", True, "safe_paraphrase", ""),
        QueryPairScenario("b", "seed two", "probe two", False, "negation", ""),
    )
    scenarios = load_query_pair_scenarios(
        queries_file=Path(__file__).resolve().parent / "queries.txt",
        scenarios=custom,
    )
    assert len(scenarios) == 2


def test_load_finance_adversarial_fixture() -> None:
    from gated_semantic_cache.eval.queries_pairs_eval import default_finance_pairs_path, load_query_pair_scenarios_from_json

    scenarios = load_query_pair_scenarios_from_json(default_finance_pairs_path())
    assert len(scenarios) == 32
    assert all(not s.expected_cache_hit for s in scenarios)
    assert scenarios[0].pair_id == "buy_vs_sell_action_polarity_0"


@pytest.mark.skipif(
    __import__("os").environ.get("RUN_QUERIES_PAIRS_EVAL") != "1",
    reason="OpenAI-backed queries-pairs eval; set RUN_QUERIES_PAIRS_EVAL=1 to run intentionally.",
)
def test_queries_pairs_eval_smoke_vector_only() -> None:
    from gated_semantic_cache.cli import _load_dotenv_files

    _load_dotenv_files()
    if not __import__("os").environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required")

    report = run_queries_pairs_eval(route_policy="vector_only", neighbor_judge=None)
    assert report.metrics.total_pairs == 12
    assert report.metrics.duplicate_pairs == 2
    assert report.metrics.non_duplicate_pairs == 10
