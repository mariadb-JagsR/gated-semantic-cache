# Quora Question Pairs — grand pooled summary (1000 pairs)

All Quora eval runs to date, non-overlapping slices:

| Seed | Pairs | Reports |
|------|-------|---------|
| 42 | 200 | [A/B md](./quora_pairs_ab_200_seed42_baseline.md) |
| 43 | 200 | [A/B md](./quora_pairs_ab_200_seed43.md) |
| 44 | 200 | [A/B md](./quora_pairs_ab_200_seed44.md) |
| 45 | 400 | [A/B md](./quora_pairs_ab_400_seed45.md) |

**Total:** 1000 pairs (500 duplicate / 500 non-duplicate probes).

Config constant across runs: `semantic_ok`, threshold `0.86`, low watermark `0.70`, `text-embedding-3-small`.

## Grand pooled A/B (1000 pairs)

| Metric | Judge ON | Judge OFF | Δ (ON − OFF) |
|--------|----------|-----------|--------------|
| **Precision** | 0.910 | 0.905 | +0.6 pp |
| **Recall** | **0.426** | 0.360 | **+6.6 pp** |
| **False positive rate** | 0.042 (21/500) | 0.038 (19/500) | +0.4 pp |
| Wrong cache answer rate | 0.090 | 0.096 | −0.6 pp |
| Total hits | 234 | 199 | +35 |
| Correct hits | 213 | 180 | +33 |
| False positives | 21 | 19 | +2 |
| False negatives | 287 | 320 | −33 |

## Interpretation

- **Recall ~43% (judge ON)** across 1000 pairs — consistent with safety-first stack (threshold, facets, strict judge). See [pooled 600 analysis](./quora_pairs_pooled_600_seeds42_43_44.md) for miss-reason breakdown.
- **Judge net effect at scale:** +6.6 pp recall, +0.4 pp FPR — modest safety cost for 33 extra correct duplicate hits.
- **Slice variance matters:** seed 43 was an outlier where judge hurt; seed 45 (400 pairs) showed judge helping strongly (+11 pp recall, −1 pp FPR). Do not tune on any single seed.
- **Recommendation for baseline:** treat **seed 45 @ 400 pairs judge ON** as the strongest single-slice result, and **1000-pair pooled** as the headline number for design comparisons going forward.
- **GPTCache-style comparison:** see [vector-only baseline](./quora_pairs_gptcache_baseline_1000.md) — similarity-only gets **+5.6 pp recall** but **+2.0 pp FPR** vs judge-on at the same threshold.

## Reproduce full suite

```bash
cd next
for seed in 42 43 44; do
  gatecache eval quora-pairs --limit 200 --seed $seed --report-json docs/quora_pairs_eval/quora_pairs_200_seed${seed}_judge-on.json
  gatecache eval quora-pairs --limit 200 --seed $seed --no-judge --report-json docs/quora_pairs_eval/quora_pairs_200_seed${seed}_no-judge.json
done
gatecache eval quora-pairs --limit 400 --seed 45 --report-json docs/quora_pairs_eval/quora_pairs_400_seed45_judge-on.json
gatecache eval quora-pairs --limit 400 --seed 45 --no-judge --report-json docs/quora_pairs_eval/quora_pairs_400_seed45_no-judge.json
```

Run date: 2026-05-24.
