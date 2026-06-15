# First Milestone Report

## What was built

The first milestone delivered the classifier-first proof of concept described in the redesign plan:

- a written implementation contract and routing label policy
- a reproducible routing dataset with four labels
- a TF-IDF + Logistic Regression routing classifier with serialization
- a compact exact cache + FAISS HNSW semantic retrieval slice
- focused tests for routing, exact cache, semantic reuse, and thread-scoped behavior
- a small shadow comparison harness for expected serving-path behavior

## Routing benchmark summary

Benchmark artifact: `routing_benchmark_report.json`

Current cross-validated results on the expanded routing dataset after targeted `SKIP_CACHE` tuning and policy-aligned relabeling:

- total examples: `128`
- class distribution:
  - `SEMANTIC_OK`: `35`
  - `SKIP_CACHE`: `37`
  - `EXACT_ONLY`: `28`
  - `THREAD_SCOPED_ONLY`: `28`
- semantic route rate: `0.3125`
- routing latency p50: `1.3377 ms`
- routing latency p95: `1.819 ms`

Per-label precision:

- `SEMANTIC_OK`: `0.8417`
- `SKIP_CACHE`: `0.9688`
- `EXACT_ONLY`: `0.8462`
- `THREAD_SCOPED_ONLY`: `0.8223`

Per-label recall:

- `SEMANTIC_OK`: `0.9444`
- `SKIP_CACHE`: `0.6472`
- `EXACT_ONLY`: `0.9286`
- `THREAD_SCOPED_ONLY`: `0.9286`

## Interpretation

The POC validates the basic architecture:

- the classifier is fast
- `EXACT_ONLY` and `THREAD_SCOPED_ONLY` are already strong
- the end-to-end serving slice can reuse via exact cache or semantic cache without any LLM on the hot path

The targeted dataset expansion, cheap feature work, and policy-aligned relabeling improved the eval signal. The total misclassification count dropped, `SKIP_CACHE` precision rose sharply to `0.9688`, and the remaining errors are more aligned with genuinely unresolved policy boundaries rather than mislabeled guidance prompts.

The current weak spot is still `SKIP_CACHE` recall. The model still lets some dangerous, freshness-sensitive, or near-neighbor trap prompts leak into safer labels, especially `SEMANTIC_OK` and `THREAD_SCOPED_ONLY`. We also now explicitly track namespace freshness assumptions in the dataset so TTL-friendly and freshness-strict namespaces can be distinguished in analysis, even though the current classifier is still query-only.

## Shadow comparison summary

Shadow artifact: `shadow_compare_report.json`

The compact shadow set currently matches expected source behavior on all `8/8` scenarios:

- semantic paraphrase reuse works
- anchored identifier queries stay out of ANN
- thread-scoped queries require explicit scope
- freshness-sensitive prompts stay live

This is a useful smoke test, not a release gate.

## Recommendation

Proceed, but only with the next routing-focused step:

1. continue expanding `SKIP_CACHE` coverage, especially freshness-sensitive, personalized, and action-execution prompts
2. add more near-neighbor trap examples that are lexically similar to safe reusable prompts
3. inspect the remaining `SKIP_CACHE -> SEMANTIC_OK` and `SKIP_CACHE -> THREAD_SCOPED_ONLY` confusions one by one
4. decide whether namespace freshness policy should become an explicit classifier input instead of metadata only
5. keep the serving path simple while the classifier data improves
6. do not add more model complexity until the current dataset and features are clearly the bottleneck

## Decision

`Proceed with cautious optimism.` The architecture is promising and materially simpler than legacy, and the latest policy-aligned eval is cleaner, but `SKIP_CACHE` still needs additional hard-negative coverage and a clearer freshness-policy story before treating this routing policy as production-ready.
