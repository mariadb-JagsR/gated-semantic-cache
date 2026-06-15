"""Small labeled set for quick generalization checks (not part of ``build_routing_dataset()``)."""

from __future__ import annotations

from gatecache.eval.datasets import RoutingExample
from gatecache.routing.labels import RoutingLabel


def build_novel_eval_examples() -> list[RoutingExample]:
    """Expectations for routes / reuse safety on prompts outside the main training mass."""

    return [
        RoutingExample(
            query="In two sentences, what is semantic caching for LLM apps?",
            label=RoutingLabel.SEMANTIC_OK,
            slice_id="N1",
            source="novel_eval",
            notes="short constraint; should stay reusable",
        ),
        RoutingExample(
            query="Revoke API key key_9f3a for workspace ws-77 immediately",
            label=RoutingLabel.SKIP_CACHE,
            slice_id="N2",
            source="novel_eval",
            notes="scoped mutation with identifiers",
        ),
        RoutingExample(
            query="Compare checkpointing vs KV cache in transformers",
            label=RoutingLabel.SEMANTIC_OK,
            slice_id="N3",
            source="novel_eval",
            notes="conceptual comparison",
        ),
        RoutingExample(
            query="What is the account balance for IBAN DE89370400440532013000?",
            label=RoutingLabel.SKIP_CACHE,
            slice_id="N4",
            source="novel_eval",
            notes="anchored personal finance",
        ),
        RoutingExample(
            query="Ticket INC-22118 root cause summary",
            label=RoutingLabel.EXACT_ONLY,
            slice_id="N5",
            source="novel_eval",
            notes="incident anchor",
        ),
        RoutingExample(
            query="Following up: use the other region from your last answer",
            label=RoutingLabel.THREAD_SCOPED_ONLY,
            slice_id="N6",
            source="novel_eval",
            notes="thread referent",
            thread_scope_present=True,
        ),
    ]
