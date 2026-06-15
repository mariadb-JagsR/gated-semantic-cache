# Curated queries.txt pair benchmark — full stack vs GPTCache-style

12 labeled seed/probe pairs from [tests/queries.txt](../../tests/queries.txt) (2 safe paraphrases, 10 traps). Each pair seeds the cache with query 1, then probes with query 2.

Threshold: **0.86**, model: **text-embedding-3-small**, judge **ON** (env default).

## Summary

| Mode | Precision | Recall | FPR | FP | FN | Pass (12 pairs) |
|------|-----------|--------|-----|----|----|-----------------|
| **Honest router + judge** | — | 0% | **0%** | **0** | 2 | **10/12** |
| Semantic_ok + judge | — | 0% | **0%** | **0** | 2 | **10/12** |
| **Vector-only (GPTCache)** | 0% | 0% | **40%** | **4** | 2 | **6/12** |

On this curated set, the full stack **blocks every trap** GPTCache-style similarity would have hit. Neither mode recalls the two safe paraphrases at threshold 0.86 (similarities 0.69 and 0.74).

Reports: [honest+judge](./queries_pairs_honest_judge-on.json), [semantic_ok+judge](./queries_pairs_semantic_ok_judge-on.json), [vector-only](./queries_pairs_vector-only.json).

## Pair-by-pair

| Pair | Expected | Honest + judge | Vector-only | Top sim | Notes |
|------|----------|----------------|-------------|---------|-------|
| Lisinopril paraphrase | **HIT** | miss | miss | 0.687 | Below threshold both modes |
| Metformin paraphrase | **HIT** | miss | miss | 0.740 | Judge `intent_change` (honest); below threshold (vector) |
| HIPAA regulatory | miss | miss | miss | 0.601 | Router SKIP_CACHE on probe |
| Pediatric vs adult Amoxicillin | miss | miss | **HIT** | 0.888 | **Vector FP** — population facet ignored |
| Type 1 vs Type 2 diabetes | miss | miss | **HIT** | 0.960 | **Vector FP** — facet gate blocks at 0.96 sim |
| Aspirin + Warfarin vs Vit C | miss | miss | miss | 0.792 | Below threshold |
| Patient 88291 vs 11022 | miss | miss | miss | 0.678 | Below threshold |
| John Doe vs Jane Smith | miss | miss | miss | 0.698 | Router THREAD_SCOPED / below threshold |
| ER wait (freshness) | miss | miss | **HIT** | 0.868 | **Vector FP** — router SKIP_CACHE blocks honest path |
| Personal Dexcom glucose | miss | miss | miss | 0.622 | Below threshold |
| PE symptoms vs exclude | miss | miss | miss | 0.748 | Facets/judge/threshold block honest; below threshold vector |
| Trial eligible vs ineligible | miss | miss | **HIT** | 0.869 | **Vector FP** — polarity flip; router SKIP_CACHE on honest |

## Interpretation

1. **GPTCache-style similarity is unsafe on these traps.** Four false positives (40% FPR on 10 negatives): pediatric/adult dosing, diabetes subtype, live ER wait, trial eligibility flip. Type 1/2 diabetes is especially dangerous — **0.96 cosine similarity** with a clinically different answer.

2. **Our stack earns its complexity here.** Honest routing (SKIP_CACHE, THREAD_SCOPED), query facets (`query_facet_named_entity_conflict`), and the gray-zone judge (`intent_change`) block every trap vector-only misses. This is the opposite of Quora pooled results (where vector-only had higher recall with modest FPR) — these examples were **designed** to catch facet/routing/freshness failures.

3. **Threshold 0.86 is conservative for healthcare paraphrases.** Both intended hits (Lisinopril, Metformin) sit in the 0.69–0.74 band — missed by all modes. Lowering threshold or gray-zone judge approval would improve recall on safe pairs but needs calibration against these traps.

4. **Quora vs queries.txt:** Quora measures generic paraphrase recall; `queries.txt` measures **safety under near-duplicate adversarial pairs**. Use both: Quora for recall/FPR baselines, queries.txt for regression on routing/facets/judge.

## Reproduce

```bash
cd next
gated-semantic-cache eval queries-pairs --route-policy honest \
  --report-json docs/queries_pairs_eval/queries_pairs_honest_judge-on.json
gated-semantic-cache eval queries-pairs --route-policy semantic_ok \
  --report-json docs/queries_pairs_eval/queries_pairs_semantic_ok_judge-on.json
gated-semantic-cache eval queries-pairs --route-policy vector_only \
  --report-json docs/queries_pairs_eval/queries_pairs_vector-only.json
```

Run date: 2026-05-24.
