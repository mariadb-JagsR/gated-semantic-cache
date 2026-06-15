# Semantic Cache Redesign for Cursor

## Objective

Completely redesign the semantic cache architecture.

This is a proposal I created using GPT 5.4 and my tacit knowledge .. seems to have promise. 

The current implementation is too influenced by:

- deep semantic analysis on the hot path
- LLM gatekeepers and LLM neighbor judges
- large structured extraction objects
- complex semantic gating logic
- fail-closed behavior that blocks obvious reusable cases
- a research architecture optimized for correctness experiments rather than production latency/cost

This redesign must **explicitly reject** that architecture as the default.

The new system should be built as a **production semantic cache**, not as a second inference pipeline.

---

## Mandatory design stance

### The new system must optimize for:

1. low hot-path latency
2. low request cost
3. simple, explainable routing
4. reasonable protection against obviously wrong semantic reuse
5. easy tuning and evaluation
6. no always-on LLM dependency for normal cache lookup
7. bounded LLM use only for uncertain post-retrieval cases
8. explicit separation between exact cache, routing, semantic retrieval, gray-zone judgment, and offline verification

### The new system must NOT optimize for:

- rich semantic understanding of every prompt
- full conversational state reconstruction
- extracting domain/action/timeframe/predicates/filters/logic for every request
- always-on LLM-based reuse decisions on the critical path
- research-grade semantic analysis in the serving path

If the new implementation drifts back toward deep hot-path semantic reasoning, it has failed the redesign.
The permitted exception is a bounded post-ANN **gray-zone judge** for candidates that are similar enough to consider but not strong enough to auto-return.

---

## Non-goals

The redesign is **not** trying to:

- deeply interpret every user request
- build a general-purpose natural language understanding engine
- reconstruct hidden prior intent state for all follow-ups
- compare large semantic state objects to decide reuse
- guarantee perfect semantic equivalence for all reuse decisions
- keep compatibility with the existing analyzer/judge architecture

---

## Explicit rejection of the current architecture

Do **not** preserve or re-center the redesign around any of the following hot-path concepts:

- provider_split analysis
- LLM gatekeeper
- always-on LLM neighbor judge
- unified_llm analyzer for live routing
- rich semantic extraction before semantic lookup
- canonical query synthesis from deep extracted semantic structure
- exact-field semantic gating via large extracted objects
- fail-closed lookup blocking caused by analyzer/extractor errors
- broad conversational state reconstruction
- fallback behavior that silently routes back through older semantic logic

The redesign should be built from scratch around a much simpler serving model.
A small LLM judge may be used only after cheap routing, scoped ANN retrieval, and deterministic gates have already narrowed the decision to an uncertainty band.

---

## Core architectural principle

### Principle

**Similarity search is candidate generation only.**

Reuse decisions should be driven by:

1. exact cache first
2. cheap routing classifier
3. scoped semantic retrieval
4. lightweight metadata compatibility checks
5. two-threshold semantic confidence
6. bounded gray-zone LLM judgment only when needed
7. optional offline verification or promotion

### Principle we are rejecting

We are **not** using a multi-step hot-path semantic analysis system to decide whether a cache entry is safe.
We are also not asking an LLM to analyze every request before retrieval.

---

## High-level request flow

```text
incoming query
  -> normalize query
  -> exact cache lookup
  -> routing classifier
  -> if classifier says SKIP_CACHE:
         go live
     else if classifier says EXACT_ONLY:
         try deterministic scoped/exact lookup
         else go live
     else if classifier says THREAD_SCOPED_ONLY:
         if thread scope exists, try scoped reuse
         else go live
     else if classifier says SEMANTIC_OK:
         compute embedding
         search FAISS HNSW within metadata scope
         apply lightweight filters
         if best similarity >= high threshold and safety gates pass:
             return cached answer
         else if best similarity >= low watermark:
             ask bounded LLM judge to accept one candidate or skip cache
         else:
             miss
  -> on miss: go live
  -> on successful live answer: insert according to route-specific insert policy
  -> optional offline verification / promotion pipeline
```

This is the only serving-path model we want.

---

## Required serving layers

## 1) Exact cache

This must be the first lookup.

### Exact cache key should be built from:

- normalized query text
- namespace / tenant / application
- agent or prompt version
- corpus / knowledge version if relevant
- tool version or schema version if relevant
- other deterministic execution context that materially affects correctness

Conceptually:

```text
exact_key = sha256(
  normalized_query
  + effective_cache_namespace
  + agent_version
  + corpus_version
  + tool_or_schema_version
  + exact_context
  + reuse_scope_key
)
```

This key is for exact same-query reuse under the same material execution context.

### Anchor key should be built from:

Some requests are not good semantic-reuse candidates but are still cacheable when strongly anchored.
Examples: order IDs, ticket IDs, incident IDs, account IDs, hostnames, UUIDs.

Conceptually:

```text
anchor_key = sha256(effective_cache_namespace + extracted_anchor)
```

The anchor key is used only for `EXACT_ONLY` / deterministic anchored lookup.
If no stable anchor can be extracted, the request goes live and should not fall back to broad semantic ANN.

### Exact cache requirements

- must be very cheap
- must not require embeddings
- must not require classifier inference
- must not require semantic analysis

---

## 2) Routing classifier

This is the most important redesign element.

Before any semantic ANN lookup, the system must run a **small, deterministic routing classifier**.

### Classifier purpose

The classifier is **not** trying to understand the whole query.
It is only deciding how the cache should behave.

### Required output labels

Start with exactly these four labels:

- `SEMANTIC_OK`
- `SKIP_CACHE`
- `EXACT_ONLY`
- `THREAD_SCOPED_ONLY`

Do not create a bigger ontology in v1.

### Label meanings

#### `SEMANTIC_OK`

Safe enough to attempt semantic retrieval.

Typical examples:

- generic factual questions
- stable documentation/FAQ queries
- reusable product capability questions
- “how does X work” or “does X support Y” style prompts

#### `SKIP_CACHE`

Do not attempt semantic cache.

Typical examples:

- destructive or mutating requests
- highly personalized/session-specific actional requests
- freshness-sensitive requests where reuse is risky
- extremely underspecified queries with no safe reuse value

#### `EXACT_ONLY`

Allow only exact or strongly anchored deterministic lookup.

Typical examples:

- order IDs
- ticket IDs
- UUID-driven lookups
- incident IDs
- customer/account specific anchored requests

#### `THREAD_SCOPED_ONLY`

Only attempt reuse if there is explicit thread/session scope.

Typical examples:

- “what about that one?”
- “same but in december”
- “instead use tokyo”
- “do it again”
- underspecified follow-up mutations

---

## Important classifier scope rule

The classifier is **cache-policy routing**, not full NLU.

Do **not** ask it to predict:

- domain
- action semantics
- timeframe normalization
- predicates
- filters
- business ontology
- semantic subject trees
- relation-to-previous beyond routing relevance

Only ask it:

- should semantic lookup be attempted?
- should cache be skipped?
- should only exact/scoped lookup be allowed?
- is the request thread-dependent?

---

## Classifier implementation requirements

### Phase 1 implementation

Implement:

- `TF-IDF + Logistic Regression`

Reasons:

- deterministic
- fast
- easy to debug
- easy to retrain
- no GPU required
- likely good enough for routing

### Phase 2 optional upgrade

Optionally add:

- `SetFit` on a small sentence-transformer backbone

Only do this if the v1 classifier clearly underperforms.

### Phase 3 optional upgrade

Only consider a small fine-tuned transformer if:

- TF-IDF + Logistic Regression clearly fails
- SetFit also fails
- routing quality is a proven bottleneck

Do **not** start with a transformer.

---

## Classifier training data

### Labeling target

Label **queries**, not semantic state objects.

Each example gets exactly one routing label:

- `SEMANTIC_OK`
- `SKIP_CACHE`
- `EXACT_ONLY`
- `THREAD_SCOPED_ONLY`

### Dataset size

Initial target:

- 1000 to 4000 labeled queries total

Good initial distribution:

- 250 to 1000 examples per class

### Data sources

Use:

1. real production/logged queries
2. existing traces
3. manually authored edge cases
4. synthetic paraphrases and synthetic small mutations

### Required dataset slices

The training/eval set must include:

- generic reusable factual questions
- destructive/actional queries
- personalized/account/session-specific queries
- anchored identifier queries
- ambiguous follow-ups
- freshness-sensitive queries
- short underspecified queries
- paraphrases that should remain reusable
- near-neighbor traps that should not reuse

---

## Classifier features

Use a combination of:

### Text features

- normalized query text
- word n-grams
- character n-grams
- TF-IDF weights

### Optional cheap engineered features

- token count
- presence of first-person markers: `my`, `me`, `our`
- presence of ambiguous reference markers: `this`, `that`, `it`, `same`, `again`, `instead`
- presence of mutation verbs: `delete`, `update`, `change`, `switch`, `cancel`, `remove`
- presence of freshness markers: `latest`, `today`, `current`, `now`, `recent`
- presence of identifier-like tokens:
  - UUIDs
  - emails
  - hostnames
  - long numeric strings
  - order/ticket/incident style tokens
- generic question markers:
  - `what is`
  - `does`
  - `how do`
  - `can`
  - `is X supported`

Do not overbuild feature extraction.

---

## 3) Semantic retrieval layer

### ANN engine

Use:

- `FAISS` with `HNSW`

Do not add an external vector DB dependency in this redesign.

### Embedding policy

Compute embeddings only when:

- exact cache misses
- routing classifier returns `SEMANTIC_OK`

Do **not** compute embeddings for:

- `SKIP_CACHE`
- `EXACT_ONLY`
- `THREAD_SCOPED_ONLY` unless explicitly scoped and allowed

### Retrieval policy

FAISS HNSW returns top-K candidates.
Start with:

- `K = 5`

Only increase if evaluation proves it is necessary.

Semantic retrieval is candidate generation only.

---

## Semantic cache entry schema

Each semantic cache entry should store:

- `cache_id`
- `namespace`
- `query_text_original`
- `query_text_normalized`
- `embedding_vector`
- `response_payload`
- `response_preview`
- `created_at`
- `expires_at`
- `cache_policy_class`
- `agent_version`
- `corpus_version`
- `tool_or_schema_version`
- `thread_scope_key` (optional)
- `exact_anchor_key` (optional)
- `freshness_class`
- `validation_status`
- `source_type`
- `confidence_metadata` (small operational metadata only)

### Important schema rule

Do **not** store or depend on large semantic extraction objects.

The old design over-relied on rich derived semantic state. Avoid this.

---

## Post-ANN candidate filtering

After FAISS returns candidates, apply only **lightweight compatibility checks**.

### Required checks

A candidate is eligible only if:

- same `namespace`
- same `agent_version` if relevant
- same `corpus_version` if relevant
- same `tool_or_schema_version` if relevant
- not expired
- same `thread_scope_key` if the current request is thread-scoped
- compatible `cache_policy_class`
- similarity above low watermark

### Forbidden hot-path checks

Do **not** do any of the following on the serving path:

- LLM gatekeeper
- LLM analyzer before retrieval
- always-on LLM judge
- semantic equivalence prompt before candidate retrieval
- large structured extraction comparison
- exact-field gating over large semantic objects
- multi-stage semantic analyzer/extractor flows

A bounded LLM judge is allowed only after ANN retrieval and deterministic compatibility checks, and only for the uncertainty band between the low watermark and the high-confidence direct-return threshold.

---

## Thresholding strategy

Do not hardcode one universal threshold and treat it as universally meaningful.

### v1 threshold policy

- semantic thresholding is only used for `SEMANTIC_OK`
- thresholds must be configurable
- start conservatively
- tune offline based on measured precision/recall and wrong-reuse rate

Use two semantic thresholds:

- **low watermark**: below this score, treat the request as a miss and go live
- **high-confidence threshold**: at or above this score, return the cached answer only if all scope, metadata, freshness, structured, and margin gates pass

Scores between the low watermark and high-confidence threshold are the **gray zone**.
In that zone, a bounded LLM judge may compare the current prompt to the eligible candidate original prompts and response previews, then either choose a winner or skip cache.

### Direct-hit safety rule

High similarity alone is not enough.
Legacy evals showed wrong hits from an overly permissive high-similarity bypass.
Direct semantic reuse must also pass compatibility gates such as:

- `cache_namespace` / effective namespace
- `reuse_scope_key`
- agent, corpus, tool, or schema version
- TTL / freshness class
- thread scope when required
- structured critical signature where available
- clear top-1 margin or another calibrated ambiguity check

Do not encode arbitrary threshold values as permanent architectural assumptions.

---

## Freshness-sensitive policy

Freshness-sensitive queries are a major source of bad semantic reuse.

### v1 rule

If the routing classifier identifies likely freshness-sensitive requests, route them to:

- `SKIP_CACHE`

Typical patterns:

- `latest`
- `today`
- `current`
- `recent`
- `this week`
- `now`

Later we may add TTL/freshness classes, but v1 should prefer safety and simplicity.

---

## Thread-scoped queries

We acknowledge thread dependence, but we do **not** want the old conversational reconstruction engine.

### v1 thread rule

If classifier returns `THREAD_SCOPED_ONLY`:

- only allow reuse if there is an explicit `thread_scope_key`
- only search within entries from the same thread/session scope
- otherwise skip semantic reuse and go live

Do not attempt hidden intent state reconstruction.

### What gets reused

Thread-scoped reuse does **not** combine the current prompt with prior user prompts inside the cache engine.
The cache engine does not inspect the last N messages, recover hidden state, or synthesize a resolved query.

Instead, it relies on the caller to provide a stable `thread_scope_key`.
Within that scope, a live answer to a follow-up can be cached with:

- the original follow-up prompt
- the response payload generated by the live system
- the same `thread_scope_key`
- `cache_policy_class = thread_scoped`

Later, a similar follow-up in the same thread may hit that scoped cache entry.

Example:

```text
thread_scope_key = support-thread-123

turn 4 user: "same but in december"
  -> classifier: THREAD_SCOPED_ONLY
  -> no scoped cache hit
  -> live answer
  -> insert as thread-scoped entry

turn 8 user: "do the december version again"
  -> classifier: THREAD_SCOPED_ONLY
  -> search only support-thread-123 entries
  -> eligible scoped semantic hit
```

The cache does not know how many previous messages matter.
If product behavior requires resolving "that one" into a standalone query, that resolution must happen outside this cache layer and be passed in as deterministic context or a resolved query representation.

---

## Exact-only queries

Exact-only queries must not go through broad semantic retrieval.

### v1 exact-only rule

If classifier returns `EXACT_ONLY`:

- try deterministic lookup using normalized anchor key
- if no exact/scoped hit, go live
- do not fall back to semantic ANN

This is intentionally strict.

---

## Insert policy

The insert policy must be simple.

### Insert only if

- live answer completed successfully
- query is cacheable by policy
- classifier returned `SEMANTIC_OK` or another explicitly allowed deterministic class
- response is not private/unsafe/non-reusable
- response is not an obvious destructive-action result
- response is not transient/freshness-sensitive unless explicitly allowed

### Route-specific insert behavior

- `SEMANTIC_OK`: insert into exact cache and semantic cache when the response is safe to reuse.
- `EXACT_ONLY`: insert into exact cache and, when a stable anchor exists, into the anchored deterministic lookup path. Do not index it for broad semantic ANN.
- `THREAD_SCOPED_ONLY`: insert only when `thread_scope_key` exists. The entry must remain scoped to that thread/session and must not become globally reusable.
- `SKIP_CACHE`: do not insert.

### Do not insert if

- classifier returned `SKIP_CACHE`
- query was unscoped and obviously session-specific
- response depends on mutable transient state
- response reflects action execution results
- response is freshness-sensitive without freshness-aware policy

Do not create a large insert-decision ontology.

---

## Optional offline verification / promotion

This is allowed, but it must remain **off the hot path**.
It is separate from the bounded gray-zone judge: the gray-zone judge may affect the current request, while offline verification improves future cache behavior only.

### Purpose

Improve future reuse quality without adding serving latency.

### Allowed offline tasks

- LLM-based semantic verification
- replay evaluation
- near-threshold analysis
- blocked-pair marking
- promoted-pair marking
- cluster-level validation

### Mandatory rule

No live request should depend on an offline verifier response.

---

## Observability requirements

For every request, emit a simple trace with:

- normalized query
- exact cache attempted? result?
- routing classifier label
- routing classifier confidence
- semantic lookup attempted?
- embedding latency
- ANN latency
- top candidate similarity
- second candidate similarity when available
- number of candidates returned
- reason candidate filtering rejected entries
- gray-zone judge invoked?
- gray-zone judge skipped reason?
- gray-zone judge rejection reason?
- final result source: exact cache / semantic cache / live
- insert performed?

Avoid giant trace blobs.
Avoid semantic analyzer-style trace trees.

---

## Required metrics

### Routing metrics

- classifier class distribution
- classifier confidence distribution
- routing latency
- percent of requests routed to semantic retrieval

### Exact cache metrics

- exact hit rate
- exact hit latency

### Semantic cache metrics

- semantic hit rate
- semantic hit latency
- semantic miss latency
- average similarity of semantic hits
- false positive rate on evaluation set
- percent of semantic candidates auto-returned above the high-confidence threshold
- percent of semantic candidates sent to gray-zone judge
- gray-zone judge accept / reject / timeout rate
- gray-zone judge latency and token cost

### End-to-end metrics

- p50 / p95 / p99 latency
- average request cost
- live-model call reduction rate

### Safety metrics

Using labeled offline evaluation:

- semantic hit precision
- semantic hit recall
- wrong reuse rate
- wrong reuse rate by routing class

---

## Evaluation plan

The redesign must ship with an explicit offline benchmark.

### Required benchmark slices

#### Slice A: generic reusable Q&A

Queries that should be good semantic-cache candidates.

#### Slice B: dangerous actional queries

Delete/change/switch/update/cancel style requests that should skip cache.

#### Slice C: anchored identifier queries

Order IDs, ticket IDs, incident IDs, hostnames, customer anchors.

#### Slice D: ambiguous follow-ups

Same but, what about that one, instead use X, do it again.

#### Slice E: freshness-sensitive queries

Latest/current/today/recent style requests.

#### Slice F: paraphrases

Different wording but same answer.

#### Slice G: near-neighbor traps

High lexical or embedding similarity but unsafe reuse.

### Evaluation requirements

Measure:

- exact hit precision/recall
- semantic hit precision/recall
- wrong reuse rate
- routing classifier confusion matrix
- direct-hit wrong reuse rate above the high-confidence threshold
- gray-zone judge precision/recall and timeout behavior
- p50/p95 latency impact
- percent of traffic sent to semantic ANN

---

## Acceptance criteria

The redesign is acceptable only if:

1. no LLM is required before exact lookup, routing, ANN retrieval, and deterministic compatibility gates
2. exact cache remains first and cheap
3. semantic lookup happens only after routing says it is worth trying
4. semantic lookup uses FAISS HNSW, not a new vector service
5. no large semantic extraction object is required for serving
6. no hot-path conversational state reconstruction is required
7. high-confidence semantic reuse relies on scope + lightweight metadata + thresholding, not LLM judgment
8. gray-zone LLM judgment is bounded, optional, observable, and fails safe to live answer
9. the system ships with offline evaluation for wrong-reuse measurement
10. the serving path is materially simpler than the current design
11. p95 latency of cache-hit paths is substantially lower than the current judge-heavy path

---

## Pseudocode: serving path

```python

def answer_query(query, context):
    normalized = normalize_query(query)

    exact_key = build_exact_key(normalized, context)
    exact_hit = exact_cache.get(exact_key)
    if exact_hit is not None:
        return response_from_exact_cache(exact_hit)

    route = routing_classifier.predict(query, context)

    if route.label == "SKIP_CACHE":
        live = live_answer(query, context)
        maybe_insert(live, route, query, context)
        return live

    if route.label == "EXACT_ONLY":
        anchor_key = build_anchor_key(query, context)
        if anchor_key is not None:
            scoped_hit = exact_cache.get(anchor_key)
            if scoped_hit is not None:
                return response_from_exact_cache(scoped_hit)

        live = live_answer(query, context)
        maybe_insert(live, route, query, context)
        return live

    if route.label == "THREAD_SCOPED_ONLY":
        if not context.thread_scope_key:
            live = live_answer(query, context)
            maybe_insert(live, route, query, context)
            return live

        semantic_hit = semantic_lookup(
            query=query,
            context=context,
            required_thread_scope=context.thread_scope_key,
        )
        if semantic_hit is not None:
            return response_from_semantic_cache(semantic_hit)

        live = live_answer(query, context)
        maybe_insert(live, route, query, context)
        return live

    if route.label == "SEMANTIC_OK":
        semantic_hit = semantic_lookup(
            query=query,
            context=context,
            required_thread_scope=None,
        )
        if semantic_hit is not None:
            return response_from_semantic_cache(semantic_hit)

        live = live_answer(query, context)
        maybe_insert(live, route, query, context)
        return live

    live = live_answer(query, context)
    maybe_insert(live, route, query, context)
    return live
```

---

## Pseudocode: semantic lookup

```python

def semantic_lookup(query, context, required_thread_scope=None):
    embedding = embed(query)

    candidates = faiss_hnsw_search(
        namespace=context.namespace,
        embedding=embedding,
        top_k=5,
    )

    filtered = []
    for c in candidates:
        if c.namespace != context.namespace:
            continue
        if c.is_expired():
            continue
        if c.agent_version != context.agent_version:
            continue
        if context.corpus_version and c.corpus_version != context.corpus_version:
            continue
        if context.tool_or_schema_version and c.tool_or_schema_version != context.tool_or_schema_version:
            continue
        if required_thread_scope is not None and c.thread_scope_key != required_thread_scope:
            continue
        if not is_policy_compatible(c.cache_policy_class, context):
            continue
        filtered.append(c)

    if not filtered:
        return None

    best = max(filtered, key=lambda x: x.similarity)
    if best.similarity < context.semantic_threshold:
        return None

    if not direct_reuse_gates_pass(best, filtered, context):
        return None

    high = context.neighbor_judge_similarity_ceiling
    if high is not None and best.similarity >= high:
        return best

    if has_clear_top1_margin(best, filtered, context):
        return best

    if context.neighbor_judge is None:
        return None

    judge_decision = context.neighbor_judge.decide(
        query=query,
        candidates=filtered,
        context=context,
    )
    if judge_decision.accepted_cache_id is None:
        return None

    return candidate_by_id(filtered, judge_decision.accepted_cache_id)
```

---

## Implementation order

Cursor should implement in this order:

### Step 1

Build exact cache cleanly.

### Step 2

Build normalization and deterministic keys.

### Step 3

Build the routing classifier:

- TF-IDF + Logistic Regression
- serialization/loading
- label mapping
- confidence output

### Step 4

Build FAISS HNSW semantic retrieval with metadata filtering.

### Step 5

Build simple insert policy.

### Step 6

Build observability and offline evaluation harness.

### Step 7

Add bounded gray-zone judge hooks after ANN and deterministic gates.
The judge must have timeouts, call caps, tracing, and fail-safe miss behavior.

### Step 8

Add optional offline verification hooks.

Do not reverse this order.

---

## Migration strategy

### Phase 1

Run the new system in shadow mode beside the existing one.

Compare:

- routing decisions
- semantic lookup attempts
- hit/miss patterns
- latency
- wrong reuse cases

### Phase 2

Enable:

- exact cache
- classifier-based routing
- FAISS/HNSW semantic lookup

with only conservative high-confidence direct reuse.

### Phase 3

Enable bounded gray-zone LLM judgment for near-threshold candidates behind configuration.
Compare judge accept/reject behavior, wrong reuse, latency, and cost before broad rollout.

### Phase 4

Retire old analyzer/judge hot-path logic.

### Phase 5

Optionally add offline verification and promotion.

---

## File/module structure suggestion

Suggested modules:

- `cache/exact_cache.py`
- `cache/semantic_store.py`
- `cache/faiss_index.py`
- `routing/classifier.py`
- `routing/features.py`
- `routing/labels.py`
- `serving/pipeline.py`
- `serving/policy.py`
- `serving/insert_policy.py`
- `serving/neighbor_judge.py`
- `eval/offline_benchmark.py`
- `eval/datasets.py`
- `observability/tracing.py`
- `models/context.py`
- `models/cache_entry.py`

This is only a suggested shape, but keep the separation strong.

---

## Final instruction to Cursor

Do not try to preserve compatibility with the current semantic-analysis-heavy architecture.

Do not infer intent from the old code structure.
Do not keep the old abstractions unless they are clearly useful.
Do not reintroduce LLM gatekeepers, `provider_split`, or always-on LLM judges on the serving path.
Do not rebuild rich extracted semantic state as a prerequisite for semantic lookup.

Build a new semantic cache that acts like:

- exact lookup first
- cheap routing second
- FAISS HNSW retrieval third
- lightweight metadata filtering fourth
- high-confidence direct reuse when gates pass
- bounded gray-zone judge only for uncertain candidates
- live answer fallback on miss, judge rejection, timeout, or budget exhaustion
- optional offline verification later

That is the target architecture.