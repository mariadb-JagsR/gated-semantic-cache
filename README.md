# GatedSemanticCache

**GatedSemanticCache** is a production-oriented semantic cache for LLM applications. Reuse decisions are made by a **multi-gate control plane** — not by cosine similarity alone.

```
exact cache → routing classifier → scoped ANN retrieval → structured/metadata gates → optional gray-zone judge
```

Similarity-only caches treat distance as safety. GatedSemanticCache adds routing (`SEMANTIC_OK`, `EXACT_ONLY`, `THREAD_SCOPED_ONLY`, `SKIP_CACHE`), exact/structured matching, facet and scope gates, and a bounded post-retrieval judge for uncertain matches.

See `docs/eval_summary_report.md` and `docs/blog_draft.md` for benchmark results (including a banking adversarial suite where similarity-only reuse fails ~51% of the time on high-risk traps).

## Layout

- `gated_semantic_cache/` — Python package (`routing/`, `cache/`, `serving/`, `structured_exact/`, `eval/`, …)
- `tests/` — unit and regression tests
- `docs/` — design notes, eval reports, CLI guide
- `semantic_cache_redesign_for_cursor.md` — source architecture document

## Setup

```bash
python -m pip install -e '.[dev]'
```

## CLI

After install: `gated-semantic-cache` (or `python -m gated_semantic_cache`).

```bash
gated-semantic-cache query -q "Explain what semantic caching is"
gated-semantic-cache route "question one" "question two"
gated-semantic-cache eval routing
gated-semantic-cache cache put -q "What is semantic caching?" \
  --response-json '{"answer":"Reuse of prior answers","success":true}'
gated-semantic-cache cache get -q "What is semantic caching?" --no-judge
```

Persistent cache default path: `$GATED_SEMANTIC_CACHE_DB` or `./.gated-semantic-cache/cache.sqlite3`.

Full CLI reference: `docs/cli_user_guide.md`.

## Python API

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

## Tests

```bash
pytest -q
```

Tests use deterministic offline embedding stubs. CLI `query` / `cache get` / `cache put` call the OpenAI embedding API when configured.

## Prior prototype

The earlier `semantic-cache-gateway` prototype was moved to `../semantic_cache_legacy_archive/` (sibling directory, not in this repo). Eval prompts from that harness are vendored under `tests/fixtures/legacy_eval/` for regression coverage only.
