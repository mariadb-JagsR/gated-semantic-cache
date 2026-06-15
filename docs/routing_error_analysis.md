# Routing Error Analysis

This report summarizes the remaining cross-validated routing mistakes from `routing_error_analysis.json`.

## Headline numbers

- total examples: `128`
- total misclassified: `19`
- overall error rate: `0.1484`

## Most common confusion pairs

1. `SKIP_CACHE -> SEMANTIC_OK` (`5`)
2. `SKIP_CACHE -> THREAD_SCOPED_ONLY` (`5`)
3. `SKIP_CACHE -> EXACT_ONLY` (`3`)
4. `SEMANTIC_OK -> SKIP_CACHE` (`1`)
5. `SEMANTIC_OK -> EXACT_ONLY` (`1`)

## Main patterns

### 1. Some risky personalized or red-flag questions still look reusable

Examples:

- `I've had the same headache and congestion, but now I'm also having blurry vision. Could this still be sinusitis?`
- `What is the latest incident status?`
- `What is my current order status for A123?`

These still leak into `SEMANTIC_OK` because they resemble reusable informational questions on the surface, even though they are red-flag, personalized, or freshness-sensitive.

### 2. Corrections and disputes look like thread follow-ups

Examples:

- `that's wrong`
- `recheck that`

These currently lean toward `THREAD_SCOPED_ONLY`, which is understandable lexically, but the intended policy is safer as `SKIP_CACHE`.

### 3. Action requests with identifiers still look exact-match friendly

Examples:

- `Delete ticket 12345`
- `Update customer 55291 email to new@example.com`
- `are you sure`

These leak into `EXACT_ONLY` because anchored entities are present, even though the verb implies a live or mutating request.

### 4. Policy questions near action or freshness language can over-trigger `SKIP_CACHE`

Example:

- `Is it still possible to cancel an order I submitted earlier today?`

This is currently labeled `SEMANTIC_OK` in the dataset, but the model often pushes it to `SKIP_CACHE` because it contains action and freshness language. This boundary may need either better examples or a labeling review.

### 5. Namespace freshness policy is now explicit metadata

The dataset now carries a `namespace_policy` hint such as `ttl_ok` or `freshness_strict`.

That helps analysis, but the current classifier is still query-only. So prompts like:

- `What changed this week in the audit log?`
- `What is the latest incident status?`

are still being classified without access to namespace policy at inference time. If freshness behavior needs to vary by namespace in production, that policy may need to become an explicit classifier input later.

## Immediate follow-ups

1. Add more explicit correction/dispute examples labeled `SKIP_CACHE`.
2. Add more paired examples separating policy/explanation prompts from direct action-execution prompts.
3. Add more examples where identifiers appear together with destructive verbs, so the model learns that action verbs outrank anchor cues.
4. Review the labels for borderline policy questions involving `today`, `cancel`, or shipped orders to make sure the dataset matches the intended routing policy.
5. Decide whether `namespace_policy` should stay analysis-only metadata or become an explicit model input.
