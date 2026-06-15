import pytest

from gated_semantic_cache.eval.datasets import _challenge_dispute_examples, build_routing_dataset
from gated_semantic_cache.routing.classifier import train_default_classifier
from gated_semantic_cache.routing.labels import RoutingLabel


@pytest.fixture(scope="module")
def routing_classifier():
    return train_default_classifier(build_routing_dataset())


@pytest.mark.parametrize(
    "query",
    [example.query for example in _challenge_dispute_examples()],
)
def test_challenge_dispute_training_rows_are_skip_cache(routing_classifier, query: str) -> None:
    assert routing_classifier.predict(query).label is RoutingLabel.SKIP_CACHE, query


def test_challenge_dispute_dataset_has_broad_coverage() -> None:
    queries = {example.query.lower() for example in _challenge_dispute_examples()}

    assert len(queries) >= 40
    assert "what? that is wrong" in queries
    assert "duh, try again" in queries
    assert "makes no sense" in queries
