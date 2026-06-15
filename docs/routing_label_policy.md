# Routing Label Policy

This document defines the v1 routing labels for the classifier-first proof of concept.

## `SEMANTIC_OK`

Use when the query is a stable, reusable information request where approximate candidate generation is acceptable.

Typical patterns:

- product capability questions
- stable documentation or FAQ questions
- reusable "how does X work" prompts
- paraphrases of the same informational request

## `SKIP_CACHE`

Use when semantic reuse is more dangerous than useful.

Typical patterns:

- destructive or mutating requests
- obviously personalized requests without a stable reusable answer
- freshness-sensitive prompts such as `today`, `latest`, `current`, `now`
- short, underspecified action requests

## `EXACT_ONLY`

Use when the request requires deterministic reuse only. This does not mean raw string equality. It means reuse must be authorized by normalized exactness over the answer-critical inputs.

Typical patterns:

- order IDs
- ticket IDs
- incident IDs
- customer or account identifiers
- UUIDs, hostnames, emails, long numeric identifiers
- structured filtered requests where critical constraints such as size, quantity, date window, or numeric bound must match exactly

Allowed exact strategies:

- normalized full-query exact lookup
- deterministic anchor-key lookup
- generic structured exact lookup over normalized typed constraints

## `THREAD_SCOPED_ONLY`

Use when the request only makes sense relative to explicit thread state, but we do not want hidden conversational reconstruction.

Typical patterns:

- `what about that one?`
- `same but in december`
- `instead use tokyo`
- `do it again`

## Guardrails

- Every query gets exactly one label.
- The classifier is not predicting domain or business ontology.
- `EXACT_ONLY` must not authorize reuse from ANN by similarity alone.
- `EXACT_ONLY` may use ANN only for candidate generation if final reuse still requires deterministic critical-constraint equality.
- `THREAD_SCOPED_ONLY` requires an explicit `thread_scope_key` to reuse.
- `SKIP_CACHE` is the safe fallback for freshness-sensitive or destructive requests.
