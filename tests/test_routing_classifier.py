from pathlib import Path

from gated_semantic_cache.eval.datasets import build_routing_dataset
from gated_semantic_cache.routing.classifier import RoutingClassifier, train_default_classifier
from gated_semantic_cache.routing.labels import RoutingLabel


def test_classifier_predicts_expected_obvious_routes() -> None:
    classifier = train_default_classifier(build_routing_dataset())

    assert classifier.predict("What is semantic caching?").label is RoutingLabel.SEMANTIC_OK
    assert (
        classifier.predict("international wire transfer fees for a business account sending USD").label
        is RoutingLabel.SEMANTIC_OK
    )
    assert (
        classifier.predict("What is the pricing for storage overage this month?").label
        is RoutingLabel.SEMANTIC_OK
    )
    assert (
        classifier.predict(
            "Show me pants with waist 32, length 32, cotton, must be made in us, doesn't fade, black"
        ).label
        is RoutingLabel.SEMANTIC_OK
    )
    assert classifier.predict("Delete ticket 12345").label is RoutingLabel.SKIP_CACHE
    assert classifier.predict("Show transactions posted this week for account 998877").label is RoutingLabel.SKIP_CACHE
    assert classifier.predict("Lookup incident 884221").label is RoutingLabel.EXACT_ONLY
    assert classifier.predict("Same but in december").label is RoutingLabel.THREAD_SCOPED_ONLY


def test_classifier_keeps_short_concrete_queries_standalone() -> None:
    classifier = train_default_classifier(build_routing_dataset())

    assert classifier.predict("dimensions for Samsung washer D1234").label is RoutingLabel.SEMANTIC_OK
    assert classifier.predict("limits for db.r6g.large").label is RoutingLabel.SEMANTIC_OK
    assert classifier.predict("docs for max_connections").label is RoutingLabel.SEMANTIC_OK
    assert classifier.predict("capital France").label is RoutingLabel.SEMANTIC_OK
    assert classifier.predict("Tom Hanks movies").label is RoutingLabel.SEMANTIC_OK
    assert classifier.predict("reverse string python").label is RoutingLabel.SEMANTIC_OK
    assert classifier.predict("common cold symptoms").label is RoutingLabel.SEMANTIC_OK
    assert classifier.predict("Top restaurants in NYC").label is RoutingLabel.SEMANTIC_OK


def test_classifier_routes_standalone_adversarial_neighbors_to_semantic_gates() -> None:
    classifier = train_default_classifier(build_routing_dataset())

    assert classifier.predict("What movies has Tom Cruise been in?").label is RoutingLabel.SEMANTIC_OK
    assert classifier.predict("How do I reverse a list in Python?").label is RoutingLabel.SEMANTIC_OK
    assert classifier.predict("How do I treat a common cold?").label is RoutingLabel.SEMANTIC_OK
    assert classifier.predict("Who founded OpenAI?").label is RoutingLabel.SEMANTIC_OK
    assert classifier.predict("Best restaurants in Los Angeles").label is RoutingLabel.SEMANTIC_OK


def test_classifier_keeps_reusable_personal_advice_queries_semantic() -> None:
    classifier = train_default_classifier(build_routing_dataset())

    assert (
        classifier.predict("My throat is sore. I was travelling this week. Maybe a viral infection ?").label
        is RoutingLabel.SEMANTIC_OK
    )
    assert (
        classifier.predict("if throat gets sore after travel is this often due to a viral infection").label
        is RoutingLabel.SEMANTIC_OK
    )


def test_classifier_still_skips_private_or_fresh_personal_state() -> None:
    classifier = train_default_classifier(build_routing_dataset())

    assert classifier.predict("Check my latest blood glucose reading from my Dexcom.").label is RoutingLabel.SKIP_CACHE
    assert classifier.predict("Show failed login attempts for my account this week").label is RoutingLabel.SKIP_CACHE


def test_classifier_routes_challenge_disputes_to_skip_cache() -> None:
    classifier = train_default_classifier(build_routing_dataset())

    for query in (
        "what? that is wrong",
        "Duh, try again",
        "makes no sense",
        "that's wrong",
        "try again",
    ):
        assert classifier.predict(query).label is RoutingLabel.SKIP_CACHE, query


def test_classifier_keeps_true_followups_thread_scoped() -> None:
    classifier = train_default_classifier(build_routing_dataset())

    assert classifier.predict("same but in december").label is RoutingLabel.THREAD_SCOPED_ONLY
    assert classifier.predict("what about that one").label is RoutingLabel.THREAD_SCOPED_ONLY
    assert classifier.predict("instead use tokyo").label is RoutingLabel.THREAD_SCOPED_ONLY


def test_classifier_round_trips_serialization(tmp_path: Path) -> None:
    classifier = train_default_classifier(build_routing_dataset())
    model_path = tmp_path / "routing.pkl"
    classifier.save(model_path)

    loaded = RoutingClassifier.load(model_path)
    assert loaded.predict("Show today's revenue in apac").label is RoutingLabel.SKIP_CACHE
