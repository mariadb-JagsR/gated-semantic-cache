# Quora Question Pairs — A/B seed 44 (200 pairs)

Third held-out sample. **Zero overlap** with seed 42 or seed 43.

## Run configuration

| Setting | Value |
|---------|-------|
| Sample | 200 pairs, balanced, seed **`44`** |
| Route policy | `semantic_ok` |
| Threshold / watermark | `0.86` / `0.70` |
| Model | `text-embedding-3-small` |
| Run date | 2026-05-24 |

## Artifacts

| Run | JSON report |
|-----|-------------|
| Judge ON | [quora_pairs_200_seed44_judge-on.json](./quora_pairs_200_seed44_judge-on.json) |
| Judge OFF | [quora_pairs_200_seed44_no-judge.json](./quora_pairs_200_seed44_no-judge.json) |

## A/B results (seed 44)

| Metric | Judge ON | Judge OFF | Delta |
|--------|----------|-----------|-------|
| Precision | 0.891 | 0.872 | +2.0 pp |
| Recall | **0.41** | 0.34 | **+7.0 pp** |
| False positive rate | 0.05 (5/100) | 0.05 (5/100) | 0.0 |
| Wrong cache answer rate | 0.109 | 0.128 | −2.0 pp |
| Correct hits | 41 | 34 | +7 |

Judge invoked: **49.0%**. Pair flips: 20 judge-only hits, 13 no-judge-only hits.

## Interpretation

On this slice the judge **helps recall without increasing FPR** — same 5 false positives in count, but **different pairs**:

- **No-judge** auto-hit several very high-similarity non-duplicates (0.93–0.96 cosine) that the judge blocked.
- **Judge** approved 5 different borderline non-duplicates (0.71–0.82 sim) while recovering 7 true duplicates from the gray zone.

Net: judge is a **trade of error types**, not strictly safer or riskier on every slice.

## Cross-seed summary (judge ON minus OFF)

| Seed | Δ Precision | Δ Recall | Δ FPR | Verdict |
|------|-------------|----------|-------|---------|
| 42 | +5.2 pp | +4.0 pp | −2.0 pp | Judge helps both |
| 43 | −11.5 pp | 0.0 | **+6.0 pp** | Judge hurts safety |
| 44 | +2.0 pp | **+7.0 pp** | 0.0 | Judge helps recall, same FPR |

**600 pairs total across three seeds:** judge effect is **slice-dependent**. Aggregate before changing judge policy.

See also: [seed 42 baseline](./quora_pairs_ab_200_seed42_baseline.md), [seed 43](./quora_pairs_ab_200_seed43.md).

## Run history

| Date | Seed | Judge | Precision | Recall | FPR | Report |
|------|------|-------|-----------|--------|-----|--------|
| 2026-05-24 | 44 | ON | 0.891 | 0.41 | 0.05 | [seed44 judge-on](./quora_pairs_200_seed44_judge-on.json) |
| 2026-05-24 | 44 | OFF | 0.872 | 0.34 | 0.05 | [seed44 no-judge](./quora_pairs_200_seed44_no-judge.json) |
