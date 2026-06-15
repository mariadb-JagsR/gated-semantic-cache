from gatecache.eval.datasets import build_routing_dataset
from gatecache.routing.labels import RoutingLabel


def test_routing_dataset_covers_all_labels_and_slices() -> None:
    examples = build_routing_dataset()
    labels = {example.label for example in examples}
    slices = {example.slice_id for example in examples}
    namespace_policies = {example.namespace_policy for example in examples}

    assert labels == {
        RoutingLabel.SEMANTIC_OK,
        RoutingLabel.SKIP_CACHE,
        RoutingLabel.EXACT_ONLY,
        RoutingLabel.THREAD_SCOPED_ONLY,
    }
    assert slices >= {"A", "B", "C", "D", "E", "F", "G", "H", "M"}
    assert len(examples) >= 100
    assert namespace_policies >= {"default", "ttl_ok", "freshness_strict"}
