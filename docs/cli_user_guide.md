# `gatecache` CLI user guide

Install and invoke:

```bash
cd next && python -m pip install -e '.[dev]'
gatecache --version
gatecache --help
python -m gatecache --help   # equivalent
```

Optional env files are loaded automatically (if `python-dotenv` is installed): `next/.env` then `./.env` (cwd wins).

---

## Commands that require `OPENAI_API_KEY`

These embed text with OpenAI (`text-embedding-3-*` by default). Set `OPENAI_API_KEY` or pass `--openai-api-key` once:

| Command | Purpose |
|--------|---------|
| `query` | Full pipeline: routing → exact → semantic ANN (optional judge); JSON trace + latency |
| `repl` | Interactive loop in **one process** (repeat same query to see exact/semantic hits) |
| `cache get` | Lookup against **SQLite + hydrated FAISS** (`SemanticCache.get`) |
| `cache put` | Insert/update durable rows + refresh FAISS snapshot (`SemanticCache.put`) |
| `cache repl` | Interactive lookups against the persistent cache |

Commands **without** this requirement include `route`, `eval …`, `cache stats`, and `cache clear`.

---

## `query` — full pipeline (one-shot or batch)

Runs the production embedding path and returns JSON with `source`, `trace`, `payload_preview`, `total_latency_ms`.

```bash
gatecache query -q "Explain semantic caching"
gatecache query -f queries.txt -o out.json
gatecache query -q "same text twice"   # second query can hit exact cache (same process)
```

Notable flags:

| Flag | Meaning |
|------|---------|
| `-q` / `--query` | Single message |
| `-f` / `--queries-file` | One query per line; `#` starts a comment |
| `--stateless` | New empty pipeline **per query** (no cross-query cache hits) |
| `--namespace` | Logical tenant (default `default`) |
| `--cache-namespace` | Override partition for cache keys (defaults to `--namespace`) |
| `--reuse-scope`, `--thread-scope` | Reuse / thread isolation metadata |
| `--semantic-threshold` | Cosine floor for semantic reuse (default env `SEMANTIC_THRESHOLD` or `0.86`) |
| `--neighbor-judge-*` | Gray-zone judge bands (see env fallbacks below) |
| `--semantic-ok-min-route-confidence` | Downgrade weak `SEMANTIC_OK` routes |
| `--openai-model` | Default `$OPENAI_MODEL` or `text-embedding-3-small` |
| `--openai-dimensions` | Reduced width for `text-embedding-3-*` (must match index dimension) |
| `--embed-cache` | LRU cache duplicate embedding strings in-process |
| `-o` / `--output-json` | Write JSON to file |
| `--no-pretty` | Single-line JSON |

---

## `repl` — interactive (in-memory pipeline)

One pipeline per process so you can type the **same** query twice and observe `exact_cache_hit` / semantic behavior.

```bash
gatecache repl
```

Supports the same embedding and threshold knobs as `query` (see `--help`). Empty line, `quit`, `exit`, or Ctrl-D exits.

---

## `route` — classifier only (no cache, no embeddings)

```bash
gatecache route "question one" "question two"
gatecache route -m /path/to/router.pkl "hello"
```

Optional `-m` loads a saved classifier pickle. No `OPENAI_API_KEY` required.

---

## `eval` — offline benchmarks

```bash
gatecache eval routing --folds 4
gatecache eval shadow
gatecache eval structured
gatecache eval coverage
gatecache eval queries-regression --mode routing
gatecache eval queries-regression --mode pipeline   # uses OpenAI embeddings; needs key
```

`queries-regression` defaults `queries-file` to `next/tests/queries.txt` in the dev tree when omitted.

---

## `cache` — durable SQLite + FAISS snapshot

Data persists across processes. Default database:

1. `GATECACHE_DB` if set  
2. Otherwise `./.gatecache/cache.sqlite3` under the **current working directory**

Override with `--db /path/to/cache.sqlite3`.

Namespace (`--namespace`) partitions rows in SQLite and picks the FAISS sidecar next to the DB.

### `cache put`

Writes exact row + optional semantic row + anchors, then saves the FAISS snapshot.

```bash
gatecache cache put \
  -q "What is semantic caching?" \
  --response-json '{"answer":"Reuse of stored responses","success":true}'
```

| Flag | Meaning |
|------|---------|
| `--semantic-mode` | `auto` \| `always` \| `never` — how inserts follow routing for ANN indexing (default `always`) |
| `--scope KEY=VALUE` | Repeatable; isolation keys for `put` |
| `--ttl-seconds` | Default `3600`; semantic row expiry metadata |
| `--no-ttl` | Store semantic row with no `expires_at` |
| `--metadata-json` | JSON object merged into entry metadata |
| `--force-rebuild-index` | On open: ignore FAISS snapshot, rebuild from DB |

### `cache get`

Lookup through `SemanticCache.get` (exact + semantic per `--semantic-mode`).

```bash
gatecache cache get -q "What is semantic caching?" --no-judge
```

| Flag | Meaning |
|------|---------|
| `--semantic-mode` | `auto` \| `always` \| `never` — retrieval behavior (default `always`) |
| `--scope KEY=VALUE` | Repeatable |
| `--no-judge` | Disable neighbor-judge gray-zone policy for this run |
| `--semantic-threshold` | Similarity floor (default env or `0.86`) |

JSON output: `hit` boolean; when hit, `source`, `payload`, `similarity`, `trace`.

### `cache repl`

Interactive `cache get` loop; scope is fixed at startup from `--scope` pairs.

### `cache stats`

Global row counts for the SQLite file. **No OpenAI calls.**

```bash
gatecache cache stats
```

### `cache clear`

Deletes all rows and the FAISS snapshot for **one** `--namespace`. Requires **`--yes`**.

```bash
gatecache cache clear --namespace default --yes
```

---

## Shared environment variables

| Variable | Used for |
|----------|-----------|
| `OPENAI_API_KEY` | Embeddings (and optional eval pipeline modes) |
| `OPENAI_MODEL` | Default embedding model (`text-embedding-3-small`) |
| `SEMANTIC_THRESHOLD` | Default cosine floor (`0.86`) |
| `GATECACHE_DB` | Default SQLite path for `cache` |
| `NEIGHBOR_JUDGE_ENABLED` | When `1`/`true`/`yes`, `query`/`repl` attach the noop neighbor judge from env wiring |
| `NEIGHBOR_JUDGE_SIMILARITY_CEILING` | Fallback if `--neighbor-judge-ceiling` omitted |
| `NEIGHBOR_JUDGE_AMBIGUITY_MARGIN` | Fallback if `--neighbor-judge-ambiguity-margin` omitted |
| `NEIGHBOR_JUDGE_MAX_CALLS` | Fallback if `--neighbor-judge-max-calls` omitted |
| `SEMANTIC_OK_MIN_ROUTE_CONFIDENCE` | Fallback for `--semantic-ok-min-route-confidence` (`query` / `repl`) |

For optional LLM neighbor judging via OpenAI (separate from embeddings), see `serving/llm_judge.py` and env such as `SEMANTIC_CACHE_DEFAULT_JUDGE`, `SEMANTIC_CACHE_JUDGE_MODEL`. The **`cache`** CLI wires **`use_default_llm_judge=False`** and uses the same noop neighbor-judge hook as `build_pipeline` (`NEIGHBOR_JUDGE_ENABLED`), not automatic LLM judge attachment.

---

## Tips

- **`query` vs `cache`**: `query` exercises an **ephemeral** pipeline (unless you only care about one-shot behavior). Use **`cache put` / `cache get`** to verify **restart-safe** behavior or multi-session workflows.
- **Embedding dimension**: If you change `--openai-dimensions` or model, existing FAISS snapshots may be incompatible; use **`--force-rebuild-index`** or delete the namespace sidecar files after clearing data.
- **Help**: `gatecache <command> --help` (e.g. `gatecache cache put --help`).
