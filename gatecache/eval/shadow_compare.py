from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from gatecache.cache.exact_cache import ExactCache
from gatecache.cache.semantic_store import SemanticStore
from gatecache.embeddings.backends import embedding_dim_for_openai_model, make_offline_fake_embedder
from gatecache.models.context import RequestContext
from gatecache.routing.classifier import train_default_classifier
from gatecache.serving.pipeline import SemanticCachePipeline

from .datasets import build_routing_dataset


@dataclass(frozen=True, slots=True)
class ShadowTurn:
    query: str
    context: RequestContext
    expected_source: str
    source: str = "legacy_eval"


def run_shadow_compare() -> dict[str, object]:
    classifier = train_default_classifier(build_routing_dataset())
    dim = embedding_dim_for_openai_model("text-embedding-3-small")
    pipeline = SemanticCachePipeline(
        router=classifier,
        exact_cache=ExactCache(),
        semantic_store=SemanticStore(dimension=dim),
        embedder=make_offline_fake_embedder(dimension=dim),
    )

    def live_answer(query: str, _: RequestContext) -> dict[str, object]:
        return {"answer": f"live:{query}", "success": True}

    results = []
    matches = 0
    for turn in build_shadow_turns():
        response = pipeline.answer_query(turn.query, turn.context, live_answer)
        matched = response.source == turn.expected_source
        matches += int(matched)
        results.append(
            {
                "query": turn.query,
                "expected_source": turn.expected_source,
                "actual_source": response.source,
                "matched": matched,
                "routing_label": response.trace.routing_label,
            }
        )

    return {
        "total_turns": len(results),
        "matched_turns": matches,
        "agreement_rate": round(matches / len(results), 4) if results else 0.0,
        "results": results,
    }


def build_shadow_turns() -> list[ShadowTurn]:
    return [
        ShadowTurn(
            query="Explain what semantic caching is",
            context=RequestContext(semantic_threshold=0.55),
            expected_source="live",
        ),
        ShadowTurn(
            query="What is semantic caching?",
            context=RequestContext(semantic_threshold=0.55),
            expected_source="semantic_cache",
        ),
        ShadowTurn(
            query="Lookup order #A123 status",
            context=RequestContext(),
            expected_source="live",
        ),
        ShadowTurn(
            query="Lookup order #A123 status",
            context=RequestContext(),
            expected_source="exact_cache",
        ),
        ShadowTurn(
            query="Same but in december",
            context=RequestContext(thread_scope_key=None),
            expected_source="live",
        ),
        ShadowTurn(
            query="Same but in december",
            context=RequestContext(thread_scope_key="thread-1", semantic_threshold=0.55),
            expected_source="live",
        ),
        ShadowTurn(
            query="Same but in december",
            context=RequestContext(thread_scope_key="thread-1", semantic_threshold=0.55),
            expected_source="exact_cache",
        ),
        ShadowTurn(
            query="Show today's revenue in apac",
            context=RequestContext(),
            expected_source="live",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a compact shadow comparison for the redesign pipeline.")
    parser.add_argument("--output-json", help="Optional path to write the shadow comparison report")
    args = parser.parse_args()

    report = run_shadow_compare()
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
