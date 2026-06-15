from gated_semantic_cache.eval.datasets import build_routing_dataset
from gated_semantic_cache.routing.classifier import train_default_classifier
from gated_semantic_cache.routing.features import identifier_like_token_count
from gated_semantic_cache.routing.labels import RoutingLabel
from gated_semantic_cache.routing.tech_tokens import is_dotted_tech_token, is_plausible_hostname
from gated_semantic_cache.structured_exact.structured_query import extract_structured_query


def test_node_js_is_not_a_hostname() -> None:
    assert is_dotted_tech_token("node.js") is True
    assert is_plausible_hostname("node.js") is False
    assert is_plausible_hostname("app.example.com") is True


def test_node_js_does_not_count_as_identifier_like_token() -> None:
    query = "details on deploying a node.js app to gcp"
    assert identifier_like_token_count(query) == 0


def test_node_js_is_not_extracted_as_hostname_identifier() -> None:
    sq = extract_structured_query("details on deploying a node.js app to gcp")
    assert all(not (c.kind == "identifier" and c.name == "hostname") for c in sq.constraints)


def test_node_js_deploy_query_routes_semantic_ok() -> None:
    classifier = train_default_classifier(build_routing_dataset())
    prediction = classifier.predict("details on deploying a node.js app to gcp")

    assert prediction.label is RoutingLabel.SEMANTIC_OK
