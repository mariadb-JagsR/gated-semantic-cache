# GatedSemanticCache Implementation Contract

This checklist translates `semantic_cache_redesign_for_cursor.md` into build-time constraints for the `next/` implementation.

## Non-negotiable serving path

The hot path must remain:

1. normalize query
2. exact cache lookup
3. routing classifier
4. semantic lookup only if routing allows it
5. lightweight compatibility checks
6. live answer fallback
7. optional insert

No live request may depend on an LLM verifier or an offline promotion job.

## Required routing labels

The v1 routing classifier must emit exactly one label per query:

- `SEMANTIC_OK`
- `SKIP_CACHE`
- `EXACT_ONLY`
- `THREAD_SCOPED_ONLY`

The classifier predicts cache behavior, not domain semantics.

## Forbidden hot-path behavior

The new implementation must not depend on any of the following while serving a request:

- provider-split analyzers
- LLM gatekeepers
- LLM neighbor judges
- rich semantic extraction objects
- exact-field gating over large semantic structures
- hidden conversational state reconstruction
- fallback paths that silently re-enter legacy logic

## Exact cache rules

- exact cache is always checked before routing or embeddings
- exact keys are built from normalized query text plus deterministic execution context
- exact lookup must not require embeddings, classifier inference, or semantic analysis
- `EXACT_ONLY` means deterministic normalized equality, not raw token-by-token equality
- exact-required requests may use generic structured normalization and compact typed constraints when answer-critical fields must match exactly
- structured exact logic must stay generic, cheap, and deterministic

## Semantic retrieval rules

- use FAISS HNSW for ANN search
- embeddings are computed only after exact miss and routing approval
- similarity search is candidate generation only
- start with `top_k = 5`
- post-ANN filtering must stay lightweight

## Allowed semantic candidate filters

- namespace match
- agent version match, if set
- corpus version match, if set
- tool or schema version match, if set
- expiry check
- thread scope match, when required
- cache policy compatibility
- threshold check

## Insert policy rules

Insert only when:

- the live answer completed successfully
- the route allows reusable storage
- the response is not private, destructive, or obviously freshness-sensitive

Do not create a large insert ontology.

## Evaluation and observability requirements

The redesign is not complete without:

- offline routing benchmark
- wrong-reuse oriented evaluation slices
- per-request traces with compact operational fields
- metrics for routing distribution, latency, hit quality, and wrong reuse

## Review checklist

Before accepting a change in `next/`, verify:

- exact cache is still first
- no LLM was added to normal lookup
- the classifier still routes only cache policy
- no large semantic extraction object became required
- semantic lookup still uses FAISS HNSW plus lightweight filters
- traces remain compact and operational
