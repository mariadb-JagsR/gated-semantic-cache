# TODO (`next/` semantic cache)

Backlog for the `next/` tree. Broader historical items live in `legacy/TODO.md`.
Domain-specific rules stay out of the shared engine (see workspace direction).

---

## Persistent cache storage and pluggable backends (partially done)

**Done (dev defaults):** Protocols in `gated_semantic_cache/cache/ports.py`:
`CacheEntryPersistence`, `VectorIndexPersistence`, `NamespaceHydrationBundle`.
`SqliteCachePersistence` + `FaissVectorPersistence` implement SQLite rows and a
namespace-scoped FAISS snapshot beside the DB file. `SemanticCache.from_sqlite`
hydrates exact + semantic stores; `put` write-through updates SQLite and refreshes
the snapshot. CLI: `gated-semantic-cache cache get|put|repl|stats|clear` (default DB:
`$GATED_SEMANTIC_CACHE_DB` or `.gated-semantic-cache/cache.sqlite3`). Swap these
adapters for GridGain / HTTP / JDBC-backed implementations without changing core
cache logic.

**Still open**

- **TTL enforcement:** `expires_at` is stored on semantic rows; eviction / pruning
  APIs are not wired yet (TTL filtering deferred per product direction).
- **Idle expiry / exact-row TTL:** exact cache payloads do not carry expiry metadata
  yet; no `last_accessed_at` touch-on-hit.
- **Production-grade backends:** MariaDB or GridGain-backed adapters implementing the
  same ports; optional MariaDB vector search instead of local FAISS.
- **HTTP server:** long-lived service wrapping `SemanticCache` (auth, quotas,
  multi-tenant routing) — CLI + library first.

---

## Embedding model choice and retrieval quality (open)

**Goal:** Pick embedders (and optional second-stage scoring) that balance latency, cost,
and precision on near-duplicate queries where a single token or attribute differs
(e.g. color words), long agent context, and offline eval in
`gated_semantic_cache/eval/`.

**Today:** `embeddings/backends.py` implements OpenAI (`make_openai_embedder`) plus test
fakes. Index dimension must match the chosen model; production wiring is integration-owned.

**Candidates to benchmark (not endorsements; validate on our datasets and traces):**

| Track | Notes |
|--------|--------|
| **API — retrieval-tuned** | e.g. Voyage `voyage-3-large` (long context, retrieval benchmarks — verify cost/latency). |
| **API — general large** | OpenAI `text-embedding-3-large` (high dims; good default to compare). |
| **Local / self-hosted** | Ollama or similar for fast iteration without cloud round-trips; quality varies by model. |
| **Open weights — versatile** | BGE-M3: dense + sparse + multi-vector in one family — useful if we add hybrid lexical+dense later. |
| **Open weights — long context** | e.g. Qwen3-Embedding family — evaluate only if our embedding inputs regularly exceed small models’ sweet spot. |
| **Efficient local** | Nomic-class models — strong “run on a laptop” option when API latency or cost dominates. |

**Second-stage disambiguation (separate from “pick a better embedder”):**

- A **reranker** (cross-encoder or dedicated rerank API) over top-*k* ANN results can
  outperform embedding cosine alone on subtle attribute differences. Treat as an
  optional stage with its own latency budget.
- The pipeline already has a **neighbor judge** hook (`serving/neighbor_judge.py`,
  `SemanticCachePipeline`) for post-retrieval LLM disambiguation. Deciding between
  “embedding + judge”, “embedding + small reranker + judge”, or “stronger
  embedding + tighter threshold” is an empirical trade-off, not a one-size label.

**Action items**

- [ ] Add pluggable embedder backends (env or constructor) for at least one
      non-OpenAI API and one local path, with explicit dimension configuration for FAISS.
- [ ] Run `offline_benchmark` / novel eval with the same gates and multiple
      embedders; log embedding + ANN latency split.
- [ ] Build a small **attribute-swap** probe set (same template, one constrained
      word changed) to stress “black vs brown” style errors; measure unsafe reuse, not
      just similarity scores.
- [ ] Document chosen defaults in `next/README.md` once numbers exist (keep engine
      domain-agnostic; configuration stays caller-owned).

**References to fold into the eval plan:** public retrieval benchmarks (e.g. MTEB) are
a starting point; product decisions should use our traces and the probes above.

---

## Scoped reuse for app-level chatbots (deferred / risky)

**Idea:** App-level chatbot caches should be isolated by application surface
(`bank-online`, `bank-trading`, support bot, DBA bot, etc.), never by a global
cross-domain cache. Some user-flavored questions could technically reuse cached
answers if the lookup is constrained by caller-supplied scope such as tenant,
principal, account, product, or session identifiers.

**Risk:** Do not let the text router broaden scope. If an account-scoped or
user-specific question is misclassified as shared-safe, cached answers could leak
across users. Apps may pass user/account scope on every request, including simple
product questions, so the cache engine must not infer shared eligibility just
because scope metadata is present.

**Current decision:** Do not introduce this architecture yet. Treat it as a
future design topic only. If revisited, shared reuse must be opt-in by
app-declared policy or response provenance, while router decisions may only narrow
reuse or force live/no-cache behavior.
