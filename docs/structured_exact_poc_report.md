# Structured Exact POC Report

## Goal

Validate the proposed `EXACT` interpretation where reuse is deterministic over normalized answer-critical constraints rather than raw string equality.

## What was implemented

- generic structured normalization
- modular typed constraint extraction for:
  - identifiers
  - dimensions
  - numeric bounds
  - quantities
  - date windows
  - simple categorical values
- ambiguity detection
- rule-based confidence scoring
- canonical structured key generation
- critical-constraint equality comparison

Implementation entry points:

- `gated_semantic_cache/serving/structured_exact.py`
- `gated_semantic_cache/structured_exact/normalize.py`
- `gated_semantic_cache/structured_exact/structured_query.py`
- `gated_semantic_cache/structured_exact/canonical_key.py`
- `gated_semantic_cache/structured_exact/matching.py`
- `gated_semantic_cache/structured_exact/extractors/`

Benchmark entry point:

- `gated_semantic_cache/eval/structured_exact_benchmark.py`

## Benchmark result

Artifact: `structured_exact_benchmark_report.json`

- total pairs: `10`
- overall accuracy: `1.0`
- positive pair match rate: `1.0`
- negative pair rejection rate: `1.0`
- extraction coverage: `0.7`

## Interpretation

The POC supports the design direction:

- paraphrases with the same critical constraints can collapse to the same deterministic key
- one-field critical changes cleanly force a miss
- incomplete low-information cases can fail closed by refusing to build a structured exact key
- the implementation can stay generic and modular without binding the core logic to one example domain

## Important caveat

This benchmark is intentionally small and hand-authored. It proves the path is technically reasonable, not that it is production-ready.

The next real step is to expand this benchmark with more diverse cases and then decide whether to:

1. keep `EXACT_ONLY` as the single routing label with multiple deterministic matching strategies, or
2. add an internal sub-mode for structured exact matching without expanding the public routing label set.
