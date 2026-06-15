# GatedSemanticCache

**Similarity is not safety.** Most semantic caches decide reuse with one number: cosine similarity above a threshold. That works for FAQ paraphrases. On real traffic — identifiers, negations, freshness, actions — it fails badly.

GatedSemanticCache keeps embeddings as the **retrieval engine** and adds a **control plane** on top: routing, structured matching, facet gates, and an optional bounded gray-zone judge. Each layer can independently veto a reuse. Every veto falls back to a live answer.

> Full write-up with charts and per-trap breakdown: [`docs/blog_draft.md`](docs/blog_draft.md)

---

## The problem

A user asks your assistant: *"Show the dispute details for case #D-7781."* You cache the answer. A minute later another user asks: *"Show the dispute details for case #D-7782."*

The two queries embed to **0.997 cosine similarity**. A standard similarity cache serves the first user's answer to the second. Two different disputes. One wrong answer, zero latency, full confidence.

The usual pattern is simple and cheap:

```
query → embed → nearest neighbor → cosine ≥ 0.85 ? reuse : call the model
```

Off-the-shelf semantic caches (LiteLLM, Redis semantic cache, GPTCache) differ in backends and TTLs, but the **reuse decision** is usually the same: one embedding distance against one cutoff.

That assumes **more similar means safer to reuse**. On many domains, it doesn't.

| Trap type | Example | Typical cosine |
|-----------|---------|----------------|
| Identifier swap | `#D-7781` vs `#D-7782` | 0.997 |
| Negation | "fees waived" vs "fees **NOT** waived" | 0.965 |
| Freshness | "today's mortgage rate" asked tomorrow | ~1.0 (stale) |
| Action | "Cancel my pending Zelle payment" | N/A — not a reusable lookup |

These sit at or above common thresholds. Embeddings can't reliably separate them from genuine paraphrases.

![Cosine similarity of safe paraphrases versus traps on 94 banking queries. The distributions overlap; there is no cutoff that catches paraphrases without serving traps.](docs/blog_assets/cosine_overlap.svg)

We measured this on a **94-query banking adversarial suite** (27 should-reuse paraphrases, 67 must-not-reuse traps). On high-risk queries, a cosine-only baseline at 0.85 served a **wrong cached answer 51% of the time**. The median trap (0.851) is about as similar as the median paraphrase (0.869); 30 traps were *more* similar to their cached query than the median true paraphrase.

There is no magic threshold. Raise it to block identifier and negation traps and recall collapses. Lower it to catch paraphrases and you serve the traps.

---

## Our approach

**Decide reuse on routing, structure, and intent — not on distance alone.**

![Control-plane flow: routing → exact/structured match → vector retrieval → facet gates → optional gray-zone judge. Any failed check falls back to a live model call.](docs/blog_assets/architecture.svg)

Each layer answers a different question:

1. **Routing classifier** — Should this query be cached at all? Labels every query `SEMANTIC_OK`, `EXACT_ONLY`, `THREAD_SCOPED_ONLY`, or `SKIP_CACHE`. Actions, freshness-sensitive lookups, and personal-data requests hit `SKIP_CACHE` *before* any vector lookup.

2. **Exact / structured match** — Do load-bearing identifiers match? `#D-7781` and `#D-7782` produce different structured keys and never collide, regardless of embedding similarity.

3. **Facet gates** — Do entities, quantities, and polarity agree? Catches negation flips, account-type swaps, and named-entity conflicts that embeddings smooth over.

4. **Bounded gray-zone judge** — For borderline cases, one cheap, timeout-bounded LLM call makes the yes/no reuse decision. Post-retrieval, optional, budget-capped — not an LLM on every request.

**Design bias:** a missed cache hit costs latency and money; a wrong cache hit costs trust. We bias hard toward the first.

---

## Results (banking adversarial suite)

Same `text-embedding-3-small` vectors. Baseline: cosine-only at 0.85. GatedSemanticCache: routing + structured match + facet gates + judge (threshold 0.86).

| Metric | GatedSemanticCache | Cosine-only baseline |
|--------|-------------------|----------------------|
| **False reuse** (wrong answers served) | **0%** (0/67) | **51%** (34/67) |
| **Recall** (paraphrases reused) | **67%** (18/27) | 56% (15/27) |

![Scoreboard: 0% false reuse vs 51% baseline; 67% recall vs 56%.](docs/blog_assets/scoreboard.svg)

The baseline is worse on **both** axes — not a safety/recall tradeoff. Identifier swaps and negations: **100% false reuse** on the baseline, **0%** on our stack.

Reproduce the eval:

```bash
python3 -m gated_semantic_cache.eval.banking_adversarial_eval --suite full100 \
  --report-json docs/banking_adversarial_report_full100.json
```

See also [`docs/eval_summary_report.md`](docs/eval_summary_report.md) for healthcare, finance, and Quora comparisons.

---

## Quick start

```bash
python -m pip install -e '.[dev]'
```

**CLI** (`gated-semantic-cache` or `python -m gated_semantic_cache`):

```bash
gated-semantic-cache query -q "Explain what semantic caching is"
gated-semantic-cache route "question one" "question two"
gated-semantic-cache eval routing
gated-semantic-cache cache put -q "What is semantic caching?" \
  --response-json '{"answer":"Reuse of prior answers","success":true}'
gated-semantic-cache cache get -q "What is semantic caching?" --no-judge
```

Persistent cache: `$GATED_SEMANTIC_CACHE_DB` or `./.gated-semantic-cache/cache.sqlite3`. Full CLI reference: [`docs/cli_user_guide.md`](docs/cli_user_guide.md).

**Python API:**

```python
from gated_semantic_cache import JudgePolicy, SemanticCache

cache = SemanticCache.from_components(
    namespace="product-support",
    router=router,
    exact_cache=exact_cache,
    semantic_store=semantic_store,
    embedder=embedder,
)

hit = cache.get("Does the product support namespace isolation?")
if hit is None:
    response = app_fetches_answer()
    cache.put("Does the product support namespace isolation?", response)
```

**Tests:**

```bash
pytest -q
```

Tests use deterministic offline embedding stubs. CLI `query` / `cache get` / `cache put` call the OpenAI embedding API when configured.

---

## Repository layout

- `gated_semantic_cache/` — package (`routing/`, `cache/`, `serving/`, `structured_exact/`, `eval/`, …)
- `tests/` — unit and regression tests
- `docs/` — eval reports, CLI guide, blog draft and figures
- `semantic_cache_redesign_for_cursor.md` — source architecture document

---

## When to use what

- **FAQ-shaped traffic** where wrong reuse is a minor annoyance → a plain cosine cache may be fine. Run your own pairs first.
- **Identifiers, polarity, freshness, or actions** → similarity alone will be confidently wrong exactly when it is most expensive to be. That is what GatedSemanticCache is for.

Bring your own near-duplicate pairs: the eval harness accepts `(cached_query, [candidate, should_reuse])` scenarios. See `gated_semantic_cache/eval/adversarial_cache_eval.py` and the banking suite as a template.
