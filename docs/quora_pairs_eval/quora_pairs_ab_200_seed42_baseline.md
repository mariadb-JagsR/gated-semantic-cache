# Quora Question Pairs — baseline A/B (200 pairs, seed 42)

Benchmark for semantic cache precision/recall against human-labeled duplicate pairs.
Each row seeds the cache with `question1`, then probes with `question2`.

## Run configuration

| Setting | Value |
|---------|-------|
| Dataset | `data/quora/quora_duplicate_questions.tsv` |
| Sample | 200 pairs, balanced (100 duplicate / 100 non-duplicate), seed `42` |
| Route policy | `semantic_ok` (forces `SEMANTIC_OK` to isolate retrieval + gates) |
| Embedding model | `text-embedding-3-small` |
| Semantic threshold | `0.86` |
| Semantic low watermark | `0.70` |
| Run date | 2026-05-24 |

## Artifacts

| Run | JSON report |
|-----|-------------|
| Judge ON | [quora_pairs_200_seed42_judge-on_baseline.json](./quora_pairs_200_seed42_judge-on_baseline.json) |
| Judge OFF (`--no-judge`) | [quora_pairs_200_seed42_no-judge_baseline.json](./quora_pairs_200_seed42_no-judge_baseline.json) |

Re-run with the same seed to reproduce the pair list:

```bash
cd next
gated-semantic-cache eval quora-pairs --limit 200 --seed 42
gated-semantic-cache eval quora-pairs --limit 200 --seed 42 --no-judge
```

Reports are saved automatically under `docs/quora_pairs_eval/` unless `--no-save-report` is passed.

## A/B results

| Metric | Judge ON | Judge OFF | Delta |
|--------|----------|-----------|-------|
| Precision | 0.909 | 0.857 | +5.2 pp |
| Recall | 0.40 | 0.36 | +4.0 pp |
| False positive rate | 0.04 (4/100) | 0.06 (6/100) | −2.0 pp |
| Wrong cache answer rate | 0.091 | 0.143 | −5.2 pp |
| Total hits | 44 | 42 | +2 |
| Correct hits | 40 | 36 | +4 |

Runtime (approx.): ~3.7 min with judge vs ~2.2 min without.

## Pair-level flips

| Outcome | Count |
|---------|-------|
| Both hit | 29 |
| Both miss | 143 |
| Judge-only hits | 15 (13 true duplicate recoveries, 2 false positives) |
| No-judge-only hits | 13 |

Judge invocation rate: **49%** (98/200 pairs).

## No-judge false-negative breakdown (top reasons)

| Reason | Count |
|--------|-------|
| `semantic_gray_zone_requires_judge` | 33 |
| `query_facet_named_entity_conflict` | 11 |
| `constraint_risk_requires_judge` | 8 |
| `below_threshold` | 8 |
| `query_facet_negation_conflict` | 4 |

## Interpretation

- **Safety:** Judge ON reduced false positives vs auto-hit above threshold (4 vs 6 on this slice).
- **Utility:** Judge recovered 13 duplicate pairs stuck in the gray zone (+4% recall).
- **Remaining gap:** Most misses are pre-judge (`below_threshold`, facet conflicts) or judge `intent_change` rejections — not fixable by judge alone.
- **Threshold tuning:** Use this report as a baseline when changing `SEMANTIC_THRESHOLD`, facet gates, or judge prompts.

## Notes for future comparisons

When re-running after design changes, keep `seed`, `limit`, and `route_policy` fixed so pair-level diffs are comparable. New runs get timestamped JSON filenames under this directory; add a row to the table below or link a new markdown summary.

| Date | Change under test | Judge | Precision | Recall | FPR | Report |
|------|-------------------|-------|-----------|--------|-----|--------|
| 2026-05-24 | Baseline (threshold 0.86, default gates) | ON | 0.909 | 0.40 | 0.04 | [judge-on baseline](./quora_pairs_200_seed42_judge-on_baseline.json) |
| 2026-05-24 | Baseline (threshold 0.86, default gates) | OFF | 0.857 | 0.36 | 0.06 | [no-judge baseline](./quora_pairs_200_seed42_no-judge_baseline.json) |
