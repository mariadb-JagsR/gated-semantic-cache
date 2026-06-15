# Quora Question Pairs — pooled A/B (600 pairs, seeds 42 + 43 + 44)

Aggregate across three non-overlapping 200-pair balanced slices (300 duplicate / 300 non-duplicate probes total).

## Pooled results

| Metric | Judge ON | Judge OFF | Δ (ON − OFF) |
|--------|----------|-----------|--------------|
| **Precision** | 0.887 | 0.905 | −1.8 pp |
| **Recall** | **0.417** | 0.380 | +3.7 pp |
| **False positive rate** | 0.053 (16/300) | 0.040 (12/300) | +1.3 pp |
| **Wrong cache answer rate** | 0.113 | 0.095 | +1.8 pp |
| Total hits | 141 | 126 | +15 |
| Correct hits | 125 | 114 | +11 |
| False positives | 16 | 12 | +4 |
| False negatives | 175 | 186 | −11 |

Config held constant: `semantic_ok` routing, threshold `0.86`, low watermark `0.70`, `text-embedding-3-small`.

## Per-seed recall (duplicates only)

| Seed | Judge ON | Judge OFF |
|------|----------|-----------|
| 42 | 0.40 | 0.36 |
| 43 | 0.44 | 0.44 |
| 44 | 0.41 | 0.34 |
| **Pooled** | **0.417** | **0.380** |

## Why recall is ~40% (not a single bug)

Breakdown of **175 duplicate misses** (judge ON, pooled):

| Miss reason | Count | % of dup misses |
|-------------|-------|-----------------|
| `intent_change` (judge rejected) | 69 | 39.4% |
| `query_facet_named_entity_conflict` | 46 | 26.3% |
| `below_threshold` (embedding too weak) | 21 | 12.0% |
| Other facet / judge / scope reasons | 39 | 22.3% |

Similarity bands for duplicate misses (judge ON):

| Band | Approx. count (pooled) |
|------|------------------------|
| sim &lt; 0.70 (retrieval gap) | 21 |
| 0.70–0.86 gray zone | ~100+ |
| ≥ 0.86 but still rejected (gates/judge) | 52 |

Hitting duplicates averages **~0.90** cosine; missing them still averages **~0.80** — many labeled duplicates sit in a band the stack intentionally treats as ambiguous.

### Contributing factors

1. **Conservative by design** — Two-threshold policy + facet gates + strict judge prompt (`When uncertain … reject reuse`) prioritize safety over hit rate.
2. **Judge `intent_change`** — Largest bucket; gray-zone pairs that embed close but the judge treats as different intent/answer type.
3. **Facet named-entity gate** — Second largest; e.g. Quora labels duplicate but surface entities differ (`Americans` vs `The Americans`).
4. **Gray zone without judge** — No-judge runs stall at `semantic_gray_zone_requires_judge` (~33 misses per seed on seed 42 alone).
5. **Threshold 0.86** — 21 pooled misses never reach the candidate floor; lowering threshold would raise recall and FPR.
6. **Quora label ≠ cache-equivalence** — Human duplicate means “same Quora thread merge”; cache needs “same answer reusable.” Label noise inflates apparent recall ceiling.
7. **One-way probe** — Seed with `question1`, probe `question2` only; paraphrase asymmetry hurts embedding match.

## Judge net effect (600 pairs)

Judge **+3.7 pp recall**, **+1.3 pp FPR** — modest recall gain at a small safety cost on aggregate. Per-seed variance remains high (seed 43 judge hurt safety; seeds 42/44 judge helped recall).

## Artifacts

| Seed | Judge ON | Judge OFF | Summary |
|------|----------|-----------|---------|
| 42 | [baseline judge-on](./quora_pairs_200_seed42_judge-on_baseline.json) | [baseline no-judge](./quora_pairs_200_seed42_no-judge_baseline.json) | [md](./quora_pairs_ab_200_seed42_baseline.md) |
| 43 | [seed43 judge-on](./quora_pairs_200_seed43_judge-on.json) | [seed43 no-judge](./quora_pairs_200_seed43_no-judge.json) | [md](./quora_pairs_ab_200_seed43.md) |
| 44 | [seed44 judge-on](./quora_pairs_200_seed44_judge-on.json) | [seed44 no-judge](./quora_pairs_200_seed44_no-judge.json) | [md](./quora_pairs_ab_200_seed44.md) |

Run date: 2026-05-24.
