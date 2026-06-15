# Quora Question Pairs — A/B seed 45 (400 pairs)

Final large held-out slice. **Zero overlap** with seeds 42, 43, or 44.

## Run configuration

| Setting | Value |
|---------|-------|
| Sample | 400 pairs, balanced (200 duplicate / 200 non-duplicate), seed **`45`** |
| Route policy | `semantic_ok` |
| Threshold / watermark | `0.86` / `0.70` |
| Model | `text-embedding-3-small` |
| Run date | 2026-05-24 |

## Artifacts

| Run | JSON report |
|-----|-------------|
| Judge ON | [quora_pairs_400_seed45_judge-on.json](./quora_pairs_400_seed45_judge-on.json) |
| Judge OFF | [quora_pairs_400_seed45_no-judge.json](./quora_pairs_400_seed45_no-judge.json) |

```bash
cd next
gatecache eval quora-pairs --limit 400 --seed 45 --report-json docs/quora_pairs_eval/quora_pairs_400_seed45_judge-on.json
gatecache eval quora-pairs --limit 400 --seed 45 --no-judge --report-json docs/quora_pairs_eval/quora_pairs_400_seed45_no-judge.json
```

Runtime (approx.): ~7 min judge ON, ~4.3 min judge OFF.

## A/B results (seed 45)

| Metric | Judge ON | Judge OFF | Delta |
|--------|----------|-----------|-------|
| **Precision** | **0.946** | 0.904 | +4.2 pp |
| **Recall** | **0.440** | 0.330 | **+11.0 pp** |
| **False positive rate** | **0.025** (5/200) | 0.035 (7/200) | **−1.0 pp** |
| Wrong cache answer rate | 0.054 | 0.096 | −4.2 pp |
| Correct hits | 88 | 66 | +22 |

Judge invoked: **46.2%** (185/400). Pair flips: 38 judge-only hits, 18 no-judge-only hits.

On this larger slice the judge is **clearly net positive**: more recall, fewer false positives, higher precision.

## Duplicate miss breakdown (judge ON, 112/200 misses)

| Reason | Count |
|--------|-------|
| `intent_change` (judge reject) | 42 |
| `query_facet_named_entity_conflict` | 41 |
| `below_threshold` | 11 |
| Other facet/judge/scope | 18 |

Same pattern as smaller slices: judge rejections + named-entity facet gate dominate misses.

## All seeds at a glance

| Seed | Pairs | Judge ON recall | Judge OFF recall | Judge ON FPR | Judge helps? |
|------|-------|-----------------|------------------|--------------|--------------|
| 42 | 200 | 0.40 | 0.36 | 0.04 | Yes |
| 43 | 200 | 0.44 | 0.44 | 0.07 | No (safety) |
| 44 | 200 | 0.41 | 0.34 | 0.05 | Yes |
| **45** | **400** | **0.44** | **0.33** | **0.025** | **Yes (strong)** |

See grand pooled summary: [1000 pairs](./quora_pairs_pooled_1000.md) (includes this run). Prior: [600 pairs](./quora_pairs_pooled_600_seeds42_43_44.md).
