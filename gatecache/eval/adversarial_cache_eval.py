"""Adversarial cache-hit expectations for end-to-end semantic reuse behavior.

This dataset is intentionally outside ``build_routing_dataset()``. It should be used as a
holdout/eval set so routing and gate changes prove generalization instead of memorizing
these exact examples.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

from gatecache.models.context import DEFAULT_SEMANTIC_LOW_WATERMARK, RequestContext


@dataclass(frozen=True, slots=True)
class AdversarialCacheTest:
    query: str
    cache_hit: bool
    note: str


@dataclass(frozen=True, slots=True)
class AdversarialCacheScenario:
    scenario: str
    cached_query: str
    tests: tuple[AdversarialCacheTest, ...]


@dataclass(frozen=True, slots=True)
class AdversarialCacheEvalRow:
    scenario: str
    cached_query: str
    query: str
    expected_cache_hit: bool
    actual_cache_hit: bool
    passed: bool
    source: str
    routing_label: str | None
    routing_confidence: float | None
    top_candidate_similarity: float | None
    rejected_reasons: list[str]
    semantic_post_ann_reject_reason: str | None
    semantic_facet_conflict_reason: str | None
    semantic_constraint_risk_reason: str | None
    neighbor_judge_invoked: bool
    note: str


@dataclass(frozen=True, slots=True)
class AdversarialCacheEvalReport:
    model: str
    semantic_threshold: float
    semantic_low_watermark: float
    total: int
    passed: int
    rows: list[AdversarialCacheEvalRow]

    @property
    def failed(self) -> int:
        return self.total - self.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "semantic_threshold": self.semantic_threshold,
            "semantic_low_watermark": self.semantic_low_watermark,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "rows": [asdict(row) for row in self.rows],
        }


def build_adversarial_cache_scenarios() -> list[AdversarialCacheScenario]:
    return [
        AdversarialCacheScenario(
            scenario="Paraphrase - same intent, different words",
            cached_query="What is the capital of France?",
            tests=(
                _t("Tell me France's capital city", True, "direct paraphrase"),
                _t("Which city is the capital of France?", True, "reordered"),
                _t("whats the captial of france", True, "typos + casing"),
                _t("capital France", True, "terse keyword-style query"),
                _t("What is the capital of Germany?", False, "entity substitution"),
                _t("What is the capital of Spain?", False, "entity substitution"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Negation - embedding weak spot",
            cached_query="Is Python a compiled language?",
            tests=(
                _t("Is Python an interpreted language?", False, "opposite classification"),
                _t("Python isn't compiled, right?", False, "negation"),
                _t("Is Python NOT a compiled language?", False, "explicit negation"),
                _t("Does Python need to be compiled?", True, "same practical question"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Entity substitution",
            cached_query="What movies has Tom Hanks been in?",
            tests=(
                _t("List Tom Hanks filmography", True, "same entity and intent"),
                _t("Tom Hanks movies", True, "terse form"),
                _t("What movies has Tom Cruise been in?", False, "person swap"),
                _t("What movies has Meryl Streep been in?", False, "person swap"),
                _t("What TV shows has Tom Hanks been in?", False, "media type swap"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Number/quantity swaps",
            cached_query="Convert 100 USD to EUR",
            tests=(
                _t("How much is 100 dollars in euros?", True, "natural-language equivalent"),
                _t("Convert 200 USD to EUR", False, "amount swap"),
                _t("Convert 5000 USD to EUR", False, "amount swap"),
                _t("Convert 100 EUR to USD", False, "direction swap"),
                _t("Convert 100 USD to GBP", False, "target currency swap"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Code tasks - swap object/language",
            cached_query="How do I reverse a string in Python?",
            tests=(
                _t("Python code to reverse a string", True, "same task"),
                _t("reverse string python", True, "keyword form"),
                _t("How do I reverse a list in Python?", False, "object swap"),
                _t("How do I reverse a string in JavaScript?", False, "language swap"),
                _t("How do I sort a string in Python?", False, "operation swap"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Specificity - subset vs superset",
            cached_query="Tell me about World War 2",
            tests=(
                _t("Give me an overview of WWII", True, "abbreviation"),
                _t("Summarize World War II", True, "roman numeral"),
                _t("Tell me about the Battle of Stalingrad", False, "subset topic"),
                _t("What caused World War 2?", False, "narrower focus"),
                _t("Tell me about World War 1", False, "digit swap"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Definition vs application",
            cached_query="What is recursion?",
            tests=(
                _t("Explain recursion to me", True, "same ask"),
                _t("Define recursion", True, "same intent"),
                _t("Show me an example of recursion in code", False, "example request"),
                _t("When should I use recursion?", False, "use-case request"),
                _t("What is iteration?", False, "related concept"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Same words, different intent",
            cached_query="How do I make a cake?",
            tests=(
                _t("Recipe for a basic cake", True, "same intent"),
                _t("How do I bake a cake?", True, "synonym"),
                _t("How do I make a cake stand?", False, "different object"),
                _t("How do I make a cake in Minecraft?", False, "game context"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Casual vs formal register",
            cached_query="What are the symptoms of the common cold?",
            tests=(
                _t("i think i caught a cold what do i look out for", True, "casual phrasing"),
                _t("common cold symptoms", True, "keyword form"),
                _t("How do I treat a common cold?", False, "treatment vs symptoms"),
                _t("What are flu symptoms?", False, "different illness"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Time-sensitive - cache cautiously",
            cached_query="Who is the current CEO of OpenAI?",
            tests=(
                _t("Who runs OpenAI?", True, "paraphrase"),
                _t("Current OpenAI CEO", True, "noun phrase"),
                _t("Who was the CEO of OpenAI in 2020?", False, "temporal shift"),
                _t("Who founded OpenAI?", False, "founder vs CEO"),
                _t("Who is the current CEO of Anthropic?", False, "organization swap"),
            ),
        ),
        AdversarialCacheScenario(
            scenario="Threshold calibration cases",
            cached_query="Best restaurants in New York",
            tests=(
                _t("Top restaurants in NYC", True, "abbreviation"),
                _t("Where should I eat in New York?", True, "rephrase"),
                _t("Best restaurants in Brooklyn", False, "location subset"),
                _t("Best restaurants in Los Angeles", False, "city swap"),
                _t("Best bars in New York", False, "venue type swap"),
            ),
        ),
    ]


def run_adversarial_cache_eval(
    *,
    semantic_threshold: float | None = None,
    semantic_low_watermark: float | None = None,
    openai_model: str | None = None,
    scenarios: list[AdversarialCacheScenario] | None = None,
) -> AdversarialCacheEvalReport:
    """Run the holdout set through isolated one-seed caches.

    Each test gets a fresh pipeline seeded with only its scenario's ``cached_query`` so misses
    cannot be affected by prior test prompts. Pass ``scenarios`` to run a domain-specific set
    (e.g. the banking trap suite) instead of the default generic holdout.
    """

    from gatecache.cli import build_pipeline

    threshold = semantic_threshold if semantic_threshold is not None else float(os.environ.get("SEMANTIC_THRESHOLD", "0.86"))
    low_watermark = (
        semantic_low_watermark
        if semantic_low_watermark is not None
        else float(os.environ.get("SEMANTIC_LOW_WATERMARK", str(DEFAULT_SEMANTIC_LOW_WATERMARK)))
    )
    model = openai_model or os.environ.get("OPENAI_MODEL", "text-embedding-3-small")

    rows: list[AdversarialCacheEvalRow] = []
    for scenario in (scenarios if scenarios is not None else build_adversarial_cache_scenarios()):
        for test in scenario.tests:
            pipeline = build_pipeline(openai_model=model, embed_cache=True)
            context = RequestContext(semantic_threshold=threshold, semantic_low_watermark=low_watermark)
            pipeline.answer_query(scenario.cached_query, context, _seed_answer)
            response = pipeline.answer_query(test.query, context, _seed_answer)
            actual_hit = response.source in {"exact_cache", "semantic_cache", "exact_anchor"}
            trace = response.trace
            rows.append(
                AdversarialCacheEvalRow(
                    scenario=scenario.scenario,
                    cached_query=scenario.cached_query,
                    query=test.query,
                    expected_cache_hit=test.cache_hit,
                    actual_cache_hit=actual_hit,
                    passed=actual_hit is test.cache_hit,
                    source=response.source,
                    routing_label=trace.routing_label,
                    routing_confidence=trace.routing_confidence,
                    top_candidate_similarity=trace.top_candidate_similarity,
                    rejected_reasons=list(trace.rejected_reasons),
                    semantic_post_ann_reject_reason=trace.semantic_post_ann_reject_reason,
                    semantic_facet_conflict_reason=trace.semantic_facet_conflict_reason,
                    semantic_constraint_risk_reason=trace.semantic_constraint_risk_reason,
                    neighbor_judge_invoked=trace.neighbor_judge_invoked,
                    note=test.note,
                )
            )

    passed = sum(1 for row in rows if row.passed)
    return AdversarialCacheEvalReport(
        model=model,
        semantic_threshold=threshold,
        semantic_low_watermark=low_watermark,
        total=len(rows),
        passed=passed,
        rows=rows,
    )


def _t(query: str, cache_hit: bool, note: str) -> AdversarialCacheTest:
    return AdversarialCacheTest(query=query, cache_hit=cache_hit, note=note)


def _seed_answer(query: str, _context: RequestContext) -> dict[str, Any]:
    return {"answer": f"seed:{query}", "success": True}
