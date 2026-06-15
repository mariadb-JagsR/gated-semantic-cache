# Quora Question Pairs — A/B seed 43 (200 pairs)

Second held-out sample from the Quora corpus. **Zero overlap** with the seed-42 baseline (different 200 row IDs).

## Run configuration

| Setting | Value |
|---------|-------|
| Dataset | `data/quora/quora_duplicate_questions.tsv` |
| Sample | 200 pairs, balanced (100 duplicate / 100 non-duplicate), seed **`43`** |
| Route policy | `semantic_ok` |
| Embedding model | `text-embedding-3-small` |
| Semantic threshold | `0.86` |
| Semantic low watermark | `0.70` |
| Run date | 2026-05-24 |

## Artifacts

| Run | JSON report |
|-----|-------------|
| Judge ON | [quora_pairs_200_seed43_judge-on.json](./quora_pairs_200_seed43_judge-on.json) |
| Judge OFF | [quora_pairs_200_seed43_no-judge.json](./quora_pairs_200_seed43_no-judge.json) |

```bash
cd next
gated-semantic-cache eval quora-pairs --limit 200 --seed 43 --report-json docs/quora_pairs_eval/quora_pairs_200_seed43_judge-on.json
gated-semantic-cache eval quora-pairs --limit 200 --seed 43 --no-judge --report-json docs/quora_pairs_eval/quora_pairs_200_seed43_no-judge.json
```

## A/B results (seed 43)

| Metric | Judge ON | Judge OFF | Delta |
|--------|----------|-----------|-------|
| Precision | 0.863 | 0.978 | **−11.5 pp** |
| Recall | 0.44 | 0.44 | 0.0 |
| False positive rate | 0.07 (7/100) | 0.01 (1/100) | **+6.0 pp** |
| Wrong cache answer rate | 0.137 | 0.022 | **+11.5 pp** |
| Total hits | 51 | 45 | +6 |
| Correct hits | 44 | 44 | 0 |

Judge invocation rate: **50.5%** (101/200 pairs).

## Pair-level flips

| Outcome | Count |
|---------|-------|
| Both hit | 32 |
| Both miss | 136 |
| Judge-only hits | 19 |
| No-judge-only hits | 13 |

On this slice the judge **adds 6 extra false positives** without improving recall. Several borderline non-duplicates were approved by the judge that auto-gates would have blocked.

## Comparison to seed-42 baseline

| Metric | Seed 42 judge Δ (ON−OFF) | Seed 43 judge Δ (ON−OFF) |
|--------|--------------------------|--------------------------|
| Precision | +5.2 pp | −11.5 pp |
| Recall | +4.0 pp | 0.0 |
| FPR | −2.0 pp | +6.0 pp |

Seed 42 favored the judge (better precision and recall). Seed 43 favors no-judge on safety (lower FPR) with identical recall. **Do not tune on a single 200-pair slice** — aggregate across seeds or use a larger sample before drawing design conclusions.

See also: [seed-42 baseline](./quora_pairs_ab_200_seed42_baseline.md).

## Run history

| Date | Seed | Judge | Precision | Recall | FPR | Report |
|------|------|-------|-----------|--------|-----|--------|
| 2026-05-24 | 42 | ON | 0.909 | 0.40 | 0.04 | [baseline judge-on](./quora_pairs_200_seed42_judge-on_baseline.json) |
| 2026-05-24 | 42 | OFF | 0.857 | 0.36 | 0.06 | [baseline no-judge](./quora_pairs_200_seed42_no-judge_baseline.json) |
| 2026-05-24 | 43 | ON | 0.863 | 0.44 | 0.07 | [seed43 judge-on](./quora_pairs_200_seed43_judge-on.json) |
| 2026-05-24 | 43 | OFF | 0.978 | 0.44 | 0.01 | [seed43 no-judge](./quora_pairs_200_seed43_no-judge.json) |
