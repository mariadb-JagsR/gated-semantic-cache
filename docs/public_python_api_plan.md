# Public Python API Plan

## Goal

Expose the cache as explicit cache primitives for app-level chatbots:

- `get`: lookup a cached answer
- `put`: explicitly store an answer the app already produced

The cache must not own live answer generation. On a miss, the application can do
whatever domain-specific work it needs, then call `put` if the response is safe to
cache.

## Boundaries

`namespace` is required and is the primary app/product boundary. A support bot for
product specs, a banking app, and a trading app should use different namespaces.

`scope_keys` are optional. Namespace-only caching is appropriate for app data that
has no personal/resource-specific content, such as product specifications or
public support documentation. Apps that cache user, account, session, tenant, or
permission-dependent answers must pass scope keys or avoid caching those answers.

The shared engine treats scope keys as opaque equality filters. It does not know
what a user, account, tenant, or product means.

## Classifier Role

The classifier remains useful as a conservative routing hint. It can narrow reuse
or skip semantic indexing/lookup for risky prompts, but it must not be treated as
an authority that broadens trust. App namespace, optional scope keys, metadata,
structured gates, and judge policy remain the binding controls.

The public API exposes semantic modes:

- `auto`: use classifier routing to decide semantic lookup/indexing
- `always`: force semantic lookup/indexing inside the provided namespace/scope
- `never`: exact lookup/storage only

The public API defaults to `always` for both `get` and `put`. This makes the
cache useful out of the box for app teams that intentionally create a namespace
and call `put` for cacheable responses. Use `auto` when you want the classifier
to narrow semantic behavior, and `never` for exact-only caches.

## Judge Default

The API defaults `JudgePolicy.enabled=True` with a `similarity_floor` of `0.70`.
This means semantic hits at or above the floor should be verified after retrieval
unless the caller disables the judge or configures skip conditions.

By default, the Python API attempts to create an OpenAI-backed judge from
`OPENAI_API_KEY`, using `SEMANTIC_CACHE_JUDGE_MODEL` when set and otherwise a
low-cost default judge model. Set `SEMANTIC_CACHE_DEFAULT_JUDGE=0` to disable
auto-wiring.

To avoid silently reusing risky answers, the implementation fails closed when no
judge callable can be configured: semantic candidates are rejected with
`neighbor_judge_not_configured`. Apps can:

- rely on the default LLM judge when credentials are available
- pass a cheaper judge/verifier callable
- disable judge policy for low-risk namespaces such as product specs
- set a high `similarity_ceiling` to skip the judge for very strong matches
- cap calls with `max_calls`

## Cheaper Verifier Options

The judge hook does not have to be a high-power LLM. Candidates to evaluate:

- small/local LLM with a constrained yes/no prompt
- cross-encoder or reranker over `(query, cached_query, cached_answer)`
- classifier-style verifier trained on hit/miss pairs
- deterministic structured/metadata gate for domains with extractable constraints
- hybrid lexical + embedding margin checks for simple FAQ/doc caches

The production default should be empirical: measure false reuse, latency, and cost
on app-specific traces before choosing a verifier.

## Server Mapping

The same request shape should map directly to a future service:

- `POST /v1/cache/get`
- `POST /v1/cache/put`
- `POST /v1/cache/delete`
- `POST /v1/cache/clear`

Server mode should preserve the same semantics: namespace required, scope keys
optional, live answer generation outside the cache.
