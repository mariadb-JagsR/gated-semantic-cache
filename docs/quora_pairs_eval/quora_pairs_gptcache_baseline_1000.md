# Quora pairs — GPTCache-style vector-only baseline (1000 pairs)

Comparison of **GPTCache-style similarity-only** lookup vs the full `next/` stack on the same 1000 labeled pairs used in [pooled summary](./quora_pairs_pooled_1000.md).

## What `vector_only` means

`--route-policy vector_only` disables everything except embedding + ANN + a single cosine threshold:

| Component | Full stack (`semantic_ok`) | GPTCache-style (`vector_only`) |
|-----------|---------------------------|--------------------------------|
| Routing classifier | Bypassed (`semantic_ok`) | Bypassed |
| Exact cache on probe | Yes | **No** (vector path only) |
| Query facets / structured gates | Yes | **No** |
| Constraint-risk path | Yes | **No** |
| Gray-zone LLM judge | Optional | **No** |
| Hit rule | Gates + threshold (+ judge in gray zone) | **Top-1 neighbor, sim ≥ 0.86** |

Same embedding model (`text-embedding-3-small`), threshold `0.86`, balanced duplicate/non-duplicate sampling per seed.

## Per-seed results

| Seed | Pairs | Mode | Precision | Recall | FPR | Hits | FP | FN |
|------|-------|------|-----------|--------|-----|------|----|----|
| 42 | 200 | judge-on | 0.909 | 0.400 | 0.040 | 44 | 4 | 60 |
| 42 | 200 | no-judge | 0.857 | 0.360 | 0.060 | 42 | 6 | 64 |
| 42 | 200 | **vector-only** | 0.778 | 0.420 | 0.120 | 54 | 12 | 58 |
| 43 | 200 | judge-on | 0.863 | 0.440 | 0.070 | 51 | 7 | 56 |
| 43 | 200 | no-judge | 0.978 | 0.440 | 0.010 | 45 | 1 | 56 |
| 43 | 200 | **vector-only** | 0.947 | 0.540 | 0.030 | 57 | 3 | 46 |
| 44 | 200 | judge-on | 0.891 | 0.410 | 0.050 | 46 | 5 | 59 |
| 44 | 200 | no-judge | 0.872 | 0.340 | 0.050 | 39 | 5 | 66 |
| 44 | 200 | **vector-only** | 0.889 | 0.480 | 0.060 | 54 | 6 | 52 |
| 45 | 400 | judge-on | 0.946 | 0.440 | 0.025 | 93 | 5 | 112 |
| 45 | 400 | no-judge | 0.904 | 0.330 | 0.035 | 73 | 7 | 134 |
| 45 | 400 | **vector-only** | 0.906 | 0.485 | 0.050 | 107 | 10 | 103 |

Reports: `quora_pairs_*_vector-only.json` in this directory.

## Grand pooled (1000 pairs)

| Metric | Judge ON | Judge OFF | **Vector-only** | Δ (vector − judge ON) |
|--------|----------|-----------|-----------------|------------------------|
| **Precision** | 0.910 | 0.905 | **0.886** | −2.4 pp |
| **Recall** | 0.426 | 0.360 | **0.482** | **+5.6 pp** |
| **False positive rate** | 0.042 (21/500) | 0.038 (19/500) | **0.062 (31/500)** | **+2.0 pp** |
| Total hits | 234 | 199 | **272** | +38 |
| Correct hits | 213 | 180 | **241** | +28 |
| False positives | 21 | 19 | **31** | +10 |
| False negatives | 287 | 320 | **259** | −28 |

## Interpretation

1. **Vector-only is not “dumber but safer.”** At threshold 0.86 it finds **more** true duplicates (+5.6 pp recall) but also **more** wrong hits (+2.0 pp FPR). The full stack’s facets, structured gates, and judge mostly **block legitimate duplicate hits** rather than dramatically improving precision on this benchmark.

2. **Precision cost is modest.** Vector-only loses ~2.4 pp precision vs judge-on (88.6% vs 91.0%) while gaining 28 extra correct duplicate hits across 1000 pairs — at the price of 10 extra false positives.

3. **Seed 42 is an outlier for vector-only FPR** (12% vs ~3–6% on other seeds). Aggregate across seeds before tuning; do not over-weight a single slice.

4. **Design takeaway:** If the product goal is maximum cache reuse on paraphrases with acceptable FPR, a pure similarity baseline is a strong lower bound on recall. The `next/` stack’s value is **controlling the tail** (facet/scope/metadata conflicts) and **bounding gray-zone risk with a judge** — not raw recall on clean paraphrase pairs.

## Reproduce

```bash
cd next
for seed in 42 43 44; do
  gatecache eval quora-pairs --limit 200 --seed $seed \
    --route-policy vector_only \
    --report-json docs/quora_pairs_eval/quora_pairs_200_seed${seed}_vector-only.json
done
gatecache eval quora-pairs --limit 400 --seed 45 \
  --route-policy vector_only \
  --report-json docs/quora_pairs_eval/quora_pairs_400_seed45_vector-only.json
```

Run date: 2026-05-24.
