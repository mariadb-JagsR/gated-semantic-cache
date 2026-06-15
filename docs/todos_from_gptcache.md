# TODOs: Ideas Worth Borrowing from GPTCache

Source: https://gptcache.readthedocs.io/en/latest/configure_it.html

This is a small backlog of concrete ideas from GPTCache that are compatible with
our `next/` architecture. It is **not** a redesign and it does not override
anything in `semantic_cache_redesign_for_cursor.md` or `implementation_contract.md`.

## Explicitly out of scope

The following GPTCache surfaces are **not** in our backlog, by decision:

- backend breadth across vector stores, SQL stores, and embedders
  (we will standardize on mature vector DBs + LangChain plugins on our own path)
- multi-modal (image/audio) caches
- `init_similar_cache_from_config(yaml)` style wiring
- `SearchDistanceEvaluation` and `OnnxModelEvaluation` style hot-path rerankers
  (we keep reuse decisions on scope + structured exact, not on a better scorer)

Everything below is additive and compatible with our "no LLM/no ML model on the
normal hot path" rule.

---

## 1. Provider-aware preprocessors (HIGH value, CHEAP)

GPTCache ships small, per-provider functions that convert a raw LLM call payload
into a clean string before anything else runs. The important ones:

- `last_content` - for chat payloads, take the last user message only
- `last_content_without_prompt` - strip a known system prompt prefix from the
  last user message
- `last_content_without_template` - strip a known prompt template (e.g. RAG
  preamble like "You are a helpful assistant... Context: {...}\nUser: {...}")
- `get_prompt` - for langchain/llama/stable-diffusion-style calls, extract the
  `prompt` field
- `get_messages_last_content` - chat variant for langchain
- `concat_all_queries` / `all_content` - full transcript concat (fallback)
- `context_process.summarization_context` / `selective_context` /
  `concat_context` - for long dialogs, compress the transcript before embedding

### Why this matters for us

The input to embedding is often wrapped in boilerplate the agent added
(system prompt, tool descriptions, retrieved context, role tags). If we embed
the whole envelope we will:

- under-cluster paraphrases (same user question, different system prompt = low
  cosine similarity),
- over-cluster unrelated requests that share templated boilerplate,
- leak prompt-template drift into "cache key" changes that should not affect
  reuse.

Stripping the templated part before embedding and before exact-key hashing is
the single cheapest accuracy improvement we can borrow. It is purely textual,
deterministic, and has zero impact on our serving path rules.

### Action

- add a `preprocessors/` module with:
  - `strip_chat_envelope(payload) -> (exact_key_text, embedding_input_text)`
  - `strip_prompt_template(text, known_templates) -> text`
  - optional `compress_long_transcript(text, max_tokens) -> text` (offline or
    on `SEMANTIC_OK` only)
- wire it in **before** normalization in `serving/pipeline.py`.
- keep the two-return shape: one string feeds the exact key, another feeds the
  embedding (GPTCache's pattern, and it's the right one for long chats).
- caller-configurable per agent/namespace, via our existing `agent_version` /
  `tool_or_schema_version` keying.

---

## 2. Multi-level cache (`next_cache` idea)

GPTCache lets you chain caches: L1 checked first, miss falls through to L2,
miss falls through to live; L2 hit populates L1.

### Why this matters for us

Natural fit for process-local FAISS (L1) + shared networked vector DB (L2)
once we bring in a mature vector backend. Also gives us a clean place to put a
per-pod hot cache without a new abstraction.

### Action

- keep our `cache/` package's public interface such that a `ChainedStore(primary, fallback)`
  wrapper can be added without changing `serving/pipeline.py`.
- do not implement until we bring in a networked vector store.

---

## 3. Post-selection policy for multiple acceptable hits

GPTCache has:

- `first` - highest similarity
- `random` - pick any candidate above threshold
- `temperature_softmax` - softmax-weighted pick, so similar-but-not-identical
  answers get rotated

### Why this matters for us

We currently return best-candidate-by-similarity after metadata filtering.
In workloads where multiple cached answers pass threshold and scope, we may
want either:

- deterministic `first` (current behavior, fine default), or
- `random` / softmax to avoid hammering the same cached response for every
  paraphrase in a burst (useful for evaluation diversity, A/B, and reducing
  staleness bias).

### Action

- add a `serving/post_select.py` with `first | random | softmax` strategies.
- config-only knob; default stays `first`.

---

## 4. Time-decay / TTL-aware scoring (inspired by `TimeEvaluation`)

GPTCache's `TimeEvaluation` penalizes old entries.

### Why this matters for us

We already have `expires_at` and `freshness_class` in the entry schema, but
scoring is currently binary (expired / not expired). For some namespaces a
softer "recent answers score higher" is useful - especially docs that change
over time without a hard expiry.

### Action

- in `SEMANTIC_OK` filtering, allow per-namespace configurable age decay:
  `score' = similarity * decay(age, halflife)`.
- opt-in per namespace; default off. Keep it deterministic.

---

## 5. Long-dialog compression for the embedding input (offline or SEMANTIC_OK only)

GPTCache's `context_process` family compresses long dialogs before embedding.

### Why this matters for us

If our agents ever cache with a long transcript as part of the intent, raw
embedding of the whole history is noisy and expensive. GPTCache uses summary
or selective-context models to produce a shorter embedding input.

### Action

- keep this **out of the hot path** by default.
- allow a namespace-level config to run a small summarizer offline, storing a
  `compressed_embedding_input` on the entry.
- on query side, use a cheap deterministic transcript compression (last-N
  turns + system prompt stripped) before embedding for `SEMANTIC_OK`.

---

## 6. Explicit "preprocessor decides two keys" pattern

GPTCache allows `pre_func` to return either one value or two:
`(exact_cache_key_text, embedding_input_text)`.

### Why this matters for us

We already have this conceptually (normalized query for exact key, possibly
different text for embedding input after stripping templates). Making it
explicit in the preprocessor contract keeps the two concerns from drifting:

- exact cache is cheap text hash on `exact_cache_key_text`
- embedding is computed on `embedding_input_text`

### Action

- enforce `Preprocessor -> (exact_key_text, embedding_input_text)` contract
  in `serving/pipeline.py`.
- default implementation returns the same string for both when no template
  stripping is needed.

---

## Priority

1. Provider-aware preprocessors (#1) + two-key contract (#6) - do first, cheap and high value.
2. Post-selection policy (#3) - trivial, drop in when someone asks.
3. Time-decay scoring (#4) - do when we have a namespace that needs it.
4. Multi-level cache (#2) - revisit when we adopt a networked vector DB.
5. Long-dialog compression (#5) - revisit only if agents start caching with
   long transcripts.

## Anti-regression reminders

Whatever we borrow from GPTCache must preserve:

- exact cache first, always
- routing classifier before embedding
- no ML model on the normal hot path
- scope + version + structured-exact as the basis for reuse, not a better scorer
- per-request trace stays compact
