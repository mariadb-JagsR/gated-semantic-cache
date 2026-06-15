# Semantic Cache Evaluation Summary

**Status:** Internal research report (May 2026; updated June 2026 with retail-banking benchmark)  
**Purpose:** Synthesize all benchmark results to date and articulate the product value of the `next/` semantic cache design versus similarity-only approaches (e.g. GPTCache-style vector lookup, LiteLLM/Redis semantic cache).

---

## Executive summary

Semantic caching promises cheaper, faster LLM responses by reusing prior answers for similar questions. The hard part is not finding *similar* questions — embeddings are good at that. The hard part is knowing when similarity is *safe*: when two phrasings deserve the same answer, and when they only *look* alike.

We evaluated our design across **four complementary benchmarks**:

1. **Quora Question Pairs (1,000 labeled pairs)** — a general-purpose paraphrase corpus with human duplicate labels. This measures recall and false-positive rate on everyday questions where some wrong reuse may be tolerable.
2. **Curated healthcare query pairs (12 labeled pairs from `tests/queries.txt`)** — adversarial near-duplicates where a wrong cache hit is clinically or operationally unacceptable.
3. **Personal finance adversarial pairs (32 labeled pairs)** — buy/sell, long/short, account-type, tax-polarity, and order-type traps where a wrong cached answer could cause the wrong trade or tax action.
4. **Retail banking adversarial suite (94 seed/probe candidates)** — identifier swaps, negation, destructive actions, freshness-sensitive balances/rates, tier swaps, product swaps, and account-holder swaps, scored head-to-head against a cosine-only baseline computed on the *same* embeddings.

**The headline finding:** On Quora, a GPTCache-style “embed + nearest neighbor + threshold” baseline achieves **higher recall** (+5.6 pp) but **more false positives** (+2.0 pp) than our full stack at the same threshold. The tradeoff looks modest on generic paraphrases. On healthcare, finance, and retail-banking traps, vector-only baselines produce **40%, 50%, and 51% false-positive rates** respectively while our stack produces **zero** in all three — including cases where cosine similarity exceeds **0.97** but the correct answer must differ. The banking benchmark adds the clearest evidence yet for *why* a single threshold cannot work: across 94 candidates on identical embeddings, the trap and paraphrase similarity distributions **interleave** — **30 of 67 traps are more cosine-similar to their seed than the median true paraphrase** — so no cutoff separates safe reuse from dangerous reuse. On that suite the cosine baseline is strictly dominated: lower recall (56% vs 67%) *and* far higher false reuse (51% vs 0%).

**Our value proposition:** Similarity-only caching is a reasonable starting point for low-risk FAQ reuse. It is not sufficient when near-duplicates encode different patients, populations, polarities, account types, or trade directions. Our design adds exactly the layers that similarity alone cannot provide: routing, structured scope, facet gates, and a bounded post-retrieval judge — without abandoning the embedding index as the primary retrieval path.

**What production users actually want:** Not maximum cache hit rate at any cost. They want **honest caching** — reuse when it is safe, a live answer when it is not, and a trace explaining why. That means accepting lower recall on generic benchmarks in exchange for near-zero false positives on the adversarial tails that dominate real-world risk in healthcare, finance, legal, and ops. Our evaluation program is built to measure both sides of that trade.

---

## What we built and measured

### Architecture under test

The `next/` cache follows a deliberate ordering:

1. **Exact cache first** — byte-normalized key hits before any embedding work.
2. **Cheap routing classifier** — decides whether a query may enter semantic reuse at all (`SEMANTIC_OK`, `SKIP_CACHE`, `EXACT_ONLY`, `THREAD_SCOPED_ONLY`).
3. **Scoped vector retrieval** — FAISS/HNSW over embeddings, filtered by namespace, version, and scope metadata.
4. **Deterministic post-retrieval gates** — structured reuse checks, query facet conflicts (entities, quantities, negation), constraint-risk signals.
5. **Two-threshold decision** — high-confidence auto-hit above `semantic_threshold` (0.86); gray zone between threshold and low watermark (0.70) optionally invokes a **bounded LLM neighbor judge**; below the watermark, miss and go live.

This is intentionally **not** “LLM gatekeeper on every request.” The judge is post-retrieval, optional, timeout-bounded, and fail-safe to live answers.

### Evaluation harnesses

| Harness | Pairs | Labels | What it tests |
|---------|-------|--------|---------------|
| `eval quora-pairs` | 1,000 (4 seeds) | Human duplicate / non-duplicate | Precision, recall, FPR on general paraphrases |
| `eval queries-pairs` (healthcare) | 12 | Expert safe-reuse intent | Safety on healthcare / operational traps |
| `eval queries-pairs --pairs-json` (finance) | 32 | All must-miss adversarial | Safety on personal-finance polarity traps |
| `eval.banking_adversarial_eval` (retail banking) | 94 (`full100`); 32 (`core32`) | 27 should-reuse / 67 must-miss | Recall + FPR on banking traps; built-in cosine-only baseline |
| GPTCache baseline (`--route-policy vector_only`) | Same sets | Same | Embed + ANN top-1 + threshold only; no router, gates, or judge |

**Common config:** `text-embedding-3-small`, cosine threshold **0.86**, low watermark **0.70**, balanced duplicate/non-duplicate sampling (Quora).

**Compared modes:**

- **Full stack (judge ON)** — gates + gray-zone judge (production-intended path).
- **Full stack (judge OFF)** — gates only; gray-zone pairs stall without a judge.
- **Vector-only** — GPTCache-style similarity baseline.
- **Honest router** — trained classifier on `queries-pairs` (production routing, not forced `SEMANTIC_OK`).

All raw JSON reports live under `docs/quora_pairs_eval/`, `docs/queries_pairs_eval/`, and `docs/finance_pairs_eval/`.

---

## Benchmark 1: Quora Question Pairs (general-purpose)

### Scale and methodology

- **1,000 pairs** across seeds 42, 43, 44 (200 each) and 45 (400).
- Each pair: seed cache with `question1`, probe with `question2`.
- **500 duplicate probes** (should hit) and **500 non-duplicate probes** (should miss).
- Routing forced to `semantic_ok` for Quora runs to isolate retrieval + gates (not router variance).

### Grand pooled results (1,000 pairs)

| Metric | Full stack (judge ON) | Full stack (judge OFF) | Vector-only (GPTCache) |
|--------|----------------------|------------------------|------------------------|
| **Precision** | **91.0%** | 90.5% | 88.6% |
| **Recall** | 42.6% | 36.0% | **48.2%** |
| **False positive rate** | **4.2%** (21/500) | 3.8% (19/500) | 6.2% (31/500) |
| Correct hits | 213 | 180 | 241 |
| False positives | 21 | 19 | **31** |
| False negatives | 287 | 320 | 259 |

**Reading the table:**

- Vector-only finds **28 more correct duplicate hits** but at the cost of **10 additional false positives** versus judge-on.
- Judge ON vs OFF: **+6.6 pp recall**, **+0.4 pp FPR** — the judge recovers gray-zone duplicates the deterministic gates alone would reject, with a small safety cost on this corpus.
- Precision remains high (~89–91%) across all modes; the main spread is recall vs FPR, not catastrophic precision collapse.

### Why recall is ~40%, not 90%+

Low recall is not a bug; it reflects **intentional conservatism** and label mismatch:

| Miss reason (judge ON, 600-pair pooled analysis) | Share of duplicate misses |
|----------------------------------------------------|---------------------------|
| Judge rejected (`intent_change`) | 39% |
| Facet named-entity conflict | 26% |
| Below embedding threshold | 12% |
| Other gates / scope / judge reasons | 23% |

Additional factors:

- **Quora “duplicate” ≠ “same reusable answer.”** Human labelers merge thread duplicates; a cache needs answer equivalence. Many labeled duplicates embed in the gray zone (~0.70–0.86) where we deliberately hesitate.
- **One-way probing** — we seed `question1` and probe `question2`, not the reverse; paraphrase asymmetry hurts embedding match.
- **Threshold 0.86** — conservative; duplicates we miss still average ~0.80 cosine similarity.

### Judge impact varies by slice — aggregate before tuning

| Slice | Judge ON recall | Judge OFF recall | Judge ON FPR | Judge OFF FPR |
|-------|-----------------|------------------|--------------|---------------|
| Pooled 1,000 | 42.6% | 36.0% | 4.2% | 3.8% |
| Seed 45 (400 pairs) | **44.0%** | 33.0% | **2.5%** | 3.5% |

On seed 45 (largest single slice), the judge is **net positive on every axis**: +11 pp recall, −1 pp FPR, +4.2 pp precision. Seed 43 was an outlier where judge hurt. **Design conclusions must use pooled metrics**, not a single random sample.

### Quora takeaway for general-purpose use

On everyday paraphrases, **similarity-only is competitive**. It wins raw recall. Our stack trades ~5–6 pp recall for ~2 pp lower FPR and higher precision, while keeping wrong-answer rate under ~9%. For a generic FAQ bot where occasional wrong reuse is annoying but not dangerous, either approach may be acceptable depending on cost/latency tolerance.

**Our stack’s Quora value is incremental on this corpus** — tighter FPR, gray-zone judge recovery, observability — not a night-and-day recall gap.

---

## Benchmark 2: Curated healthcare pairs (zero-FPR domain)

### Why this benchmark exists

Quora tells us how we behave on **average** questions. It does not tell us what happens when two queries differ by:

- Patient ID or name  
- Pediatric vs adult dosing  
- Disease subtype (Type 1 vs Type 2 diabetes)  
- Polarity flip (eligible vs ineligible; symptoms vs exclusion)  
- Freshness (“right now” ER wait)  
- Personal device readings (Dexcom glucose)

These are the cases where **no false positive rate can be tolerated** — wrong reuse is not a bad UX moment, it is a wrong patient, wrong dose, or stale operational data.

We encoded 12 seed/probe scenarios in `tests/queries.txt`: **2 safe paraphrases** (should hit), **10 traps** (must miss).

### Results (threshold 0.86, judge ON)

| Mode | False positives | FPR | Recall on safe paraphrases | Pair pass rate |
|------|-----------------|-----|----------------------------|----------------|
| **Honest router + judge** | **0** | **0%** | 0% (2 misses) | **10/12** |
| Vector-only (GPTCache) | **4** | **40%** | 0% (2 misses) | 6/12 |

### Traps vector-only hits — and our stack blocks

| Scenario | Vector-only | Full stack | Similarity | Why it matters |
|----------|-------------|------------|------------|----------------|
| Pediatric vs adult Amoxicillin | **HIT (wrong)** | miss | 0.89 | Wrong population → wrong dose |
| Type 1 vs Type 2 diabetes treatment | **HIT (wrong)** | miss | **0.96** | Embeddings nearly identical; clinically different |
| Live ER wait time paraphrase | **HIT (wrong)** | miss | 0.87 | Stale operational data |
| Trial eligible vs ineligible | **HIT (wrong)** | miss | 0.87 | Polarity flip |

**Type 1 vs Type 2 at 0.96 cosine similarity** is the clearest single datapoint in the entire evaluation program: **vector similarity is not a safety metric.** Our facet gate (`query_facet_named_entity_conflict`) rejects reuse even when embeddings scream “match.”

### How the stack blocked traps

| Mechanism | Examples blocked |
|-----------|-------------------|
| **Routing (`SKIP_CACHE`)** | HIPAA/PHI, live ER wait, trial polarity — queries never enter semantic reuse |
| **Routing (`THREAD_SCOPED_ONLY`)** | Patient-specific discharge summaries without thread scope |
| **Query facet gates** | Diabetes subtype entity conflict at 0.96 sim |
| **Gray-zone judge** | Metformin paraphrase rejected as `intent_change` at 0.74 sim |
| **Threshold** | Most traps and safe paraphrases below 0.86 |

### The recall gap on safe healthcare paraphrases

Both intended hits (Lisinopril class paraphrase, Metformin GI paraphrase) were **missed by all modes** at threshold 0.86 — similarities **0.69** and **0.74**. This is the central tension:

- **Lowering threshold** would improve recall on legitimate medical paraphrases.
- **Healthcare traps show vector-only FPR of 40%** at the same threshold — any threshold tuning must be calibrated on *both* benchmarks, not Quora alone.

The path forward is not “pick 0.86 forever.” It is **domain-specific calibration**: gray-zone judge approval for safe paraphrases, hard facet blocks for traps, and routing to skip cache entirely for freshness- and PHI-sensitive prompts.

### Healthcare takeaway

**This is where our design earns its complexity.** On traps engineered to fool embeddings, GPTCache-style similarity fails openly. Our stack achieves **zero false positives** with honest routing and gates. The cost is conservative recall on safe paraphrases at the current threshold — an acceptable trade in domains where FPR is unacceptable, and exactly the trade routing + judge + calibration are meant to manage.

---

## Benchmark 3: Personal finance adversarial pairs (zero-FPR domain)

### Why this benchmark exists

Finance is where embedding collapse is most economically dangerous. A one-token swap — buy vs sell, long vs short, call vs put, Roth vs Traditional — often leaves sentence structure identical and cosine similarity very high. Users asking these questions are frequently about to **take an action**; serving the wrong cached answer is not a minor UX glitch.

We encoded **32 seed/probe pairs** in `tests/fixtures/finance_adversarial_pairs.json`, covering:

- Action polarity (buy/sell, withdraw/deposit, owe/refund)
- Directional bets (long/short, call/put, hedge vs profit from crash)
- Account and tax distinctions (Roth vs Traditional IRA, pre-tax vs Roth 401(k), short- vs long-term gains)
- Order mechanics (market vs limit, stop-loss vs stop-limit)
- Timing and rule specifics (wash sale 1 day vs 31 days, refinance when rates rise vs fall)

Every pair is labeled **must miss**.

### Results (threshold 0.86, honest router + judge ON)

| Mode | False positives | FPR | Pair pass rate |
|------|-----------------|-----|----------------|
| **Honest router + judge** | **0** | **0%** | **32/32** |
| Vector-only (GPTCache) | **16** | **50%** | 16/32 |

**Half of finance traps fool pure similarity at 0.86.** Our stack blocks all of them.

### Highest-risk vector-only false positives

| Trap | Similarity | Why a hit is dangerous |
|------|------------|------------------------|
| Roth ↔ Traditional IRA conversion (reversed) | **0.975** | Second direction is often not allowed; opposite tax logic |
| 401(k) → IRA vs IRA → 401(k) rollover | **0.968** | Reverse rollovers have different rules and restrictions |
| Buy vs sell Tesla | 0.946 | Opposite trade action |
| Call vs put on Microsoft | 0.923 | Opposite directional bet |
| Unrealized vs realized gain on AAPL | 0.922 | Taxable event vs paper gain |
| Short-term vs long-term capital gains rate | 0.920 | Very different tax treatment |

The other 16 pairs vector-only missed only because similarity fell **below 0.86** — not because thresholding is inherently safe. On this set, **coin-flip FPR** is what similarity-only buys you.

### How our stack blocked the 16 vector false positives

| Mechanism | Pairs blocked |
|-----------|---------------|
| Gray-zone **judge** (`intent_change`, `intent_differs`, etc.) | 11 |
| **Query facet** gates (entity/token conflict) | 3 |
| **Routing** (`SKIP_CACHE`) | 2 |

Examples: reversed IRA rollover blocked by judge at **0.968 sim**; Roth vs Traditional withdrawal blocked by facet gate at 0.889 sim; buy vs sell Tesla blocked by router before semantic lookup at 0.946 sim.

### Finance takeaway

Finance adversarial pairs show **higher vector-only FPR (50%) than healthcare (40%)** on a larger trap set. This is the workload profile of robo-advisors, banking chatbots, tax assistants, and brokerage help desks — domains where production users will not accept “we hit 48% recall on Quora” as a safety argument. They want **zero wrong reuse on action-bearing near-duplicates**, with observability when the system chooses to go live.

---

## Benchmark 4: Retail banking adversarial (same-embedding head-to-head)

### Why this benchmark exists

The healthcare and finance suites compare against a *vector-only routing policy* inside our own harness. This benchmark closes the loop differently: it runs a **standalone cosine-only baseline** — embed seed, embed probe, reuse iff `cosine ≥ 0.85`, no router, no gates, no judge — on the **exact same `text-embedding-3-small` vectors** the full stack uses. Same embeddings, same probes; the only variable is the control plane. This isolates what the layers above retrieval are worth, and lets us inspect the raw similarity distribution directly.

The suite encodes **94 seed/probe candidates across 45 banking scenarios** (`gated_semantic_cache/eval/banking_adversarial_eval.py`, `--suite full100`), rebalanced from an initial 32-candidate pass so that **every trap type carries n ≥ 8** and per-type FPR is statistically meaningful. Tagged by trap type: **27 should-reuse paraphrases** (recall) and **67 must-miss traps** — identifier swaps (wire/dispute/loan/CD/branch IDs), negation, destructive actions (cancel payment, lock card, move funds), freshness-sensitive balances and live rates, tier swaps (premium vs basic), product swaps (checking vs savings, FDIC vs SIPC), and account-holder swaps (Jack vs Jill). Each probe runs against a fresh one-entry cache seeded only with its scenario's cached query, so misses cannot leak across scenarios.

### Results (94 candidates; threshold 0.86, judge ON; baseline 0.85)

| Metric | Full stack | Cosine-only baseline |
|--------|-----------|----------------------|
| **False reuse rate (FPR)** | **0%** (0/67) | **51%** (34/67) |
| Recall (TPR) on paraphrases | **67%** (18/27) | 56% (15/27) |
| Confusion | tp=18 fn=9 fp=0 tn=67 | tp=15 fn=12 fp=34 tn=33 |

On this suite the cosine-only cache is **worse on both axes at once**: it serves wrong answers on half the traps *and* recovers fewer genuine paraphrases. That is not a recall/safety tradeoff — it is strictly dominated. (The initial 32-candidate pass showed the same pattern: 0% vs 50% FPR, 58% vs 33% recall.)

### The clearest evidence that one threshold cannot work

Measured cosine similarity on the same embeddings (n=94):

```
PARAPHRASES (should-reuse):   0.707 ──── median 0.869 ──── 0.936
TRAPS       (must-NOT-reuse): 0.588 ──── median 0.851 ──── 0.997
```

The two distributions **interleave**: **30 of 67 traps are more cosine-similar to their seed than the median true paraphrase**, and **12 of 27 paraphrases are less similar than the median trap**. A threshold high enough to reject the negation and identifier traps (which run up to 0.997) also rejects most legitimate paraphrases (many at 0.71–0.86); a threshold low enough to admit the paraphrases admits the traps. There is no separating cutoff — the property a similarity-only cache depends on simply does not hold on this traffic.

### Per-trap-type false reuse (cosine-only baseline, n=94)

| Trap type | Baseline FPR | Representative trap | Similarity |
|-----------|-------------|---------------------|------------|
| Identifier swap | **100%** (9/9) | dispute case #D-7781 → #D-7782 | **0.997** |
| Negation | **100%** (10/10) | wire fees waived → **NOT** waived | **0.965** |
| Freshness | **50%** (5/10) | "today's 30-year mortgage rate" reasked | 0.938 |
| Principal swap | **38%** (3/8) | Maria Lopez → Maria Gomez transactions | 0.908 |
| Destructive action | **25%** (3/12) | "Transfer $5,000…" → "Move $5,000…" | 0.920 |
| Tier swap | **25%** (2/8) | premium → basic account | 0.883 |
| Product swap | **20%** (2/10) | checking → savings minimum balance | 0.921 |

A caveat that matters for honesty: where the baseline avoided a trap, it was usually **because that trap's cosine happened to fall below 0.85**, not because it recognized the trap as unsafe. The action category makes this concrete — at the 32-candidate scale the baseline scored 0% FPR on actions, but with more action paraphrases in the 94-set, three landed above 0.85 (e.g. "Move $5,000…" at 0.920) and were wrongly served. The same paraphrases the full stack missed sit at 0.71–0.86, so lowering the baseline threshold to recover recall would pull these sub-threshold traps back above the line. The baseline's "true negatives" are an artifact of where the cutoff landed, not a property of the method.

### How the full stack reached zero false reuse

| Mechanism | Traps blocked |
|-----------|---------------|
| **Routing `EXACT_ONLY`** | All 3 identifier swaps — never enter the ANN path; reuse requires an exact anchor match |
| **Routing `SKIP_CACHE`** | Freshness-sensitive balances/rates and destructive actions — the seed is never inserted, so the probe is forced live |
| **Query facet gates** | Negation and several product/scope conflicts (`query_facet_named_entity_conflict`) |
| **Gray-zone / post-ANN judge** | Tier and account-type swaps rejected as `different_account_type` / `different_scope` / `intent_change`, including a negation at high similarity |

### The recall cost, accounted honestly

The nine paraphrases the full stack missed are all **conservative misses** — the system returned a correct live answer, never a wrong cached one. They break down as: **three facet-gate over-rejections** (`query_facet_named_entity_conflict` on non-conflicts), **four judge over-rejections** (`different_question_focus`, `different_intent`, `different_question_type`, `intent_change` — some defensible, some not), and **two routing misses** (paraphrases misrouted to `THREAD_SCOPED_ONLY` and forced live without a thread scope). The facet gate and the gray-zone judge are the highest-leverage recall targets — the same conclusion the Quora and healthcare miss analyses reached independently.

### Banking takeaway

This is the cleanest same-embedding comparison in the program, and it removes the usual "you just tuned the threshold wrong" objection: we show the threshold *cannot* be tuned right, because the trap and paraphrase similarity distributions overlap. The full stack's safety does not come from a better number — it comes from deciding reuse on routing, structure, and intent rather than on distance alone.

---

## Synthesis: four benchmarks, one story

```
              ┌─────────────────────────────────────────────────────┐
              │               EVALUATION LANDSCAPE                  │
              └─────────────────────────────────────────────────────┘

   Quora (1,000)        Healthcare (12)     Finance (32)      Banking (94)
   ──────────────       ───────────────     ─────────────     ─────────────
   General paraphr.     Clinical traps      Trade traps       Banking traps
   Some FPR ok          Zero FPR req.       Zero FPR req.     Zero FPR req.

   Vec:  48% rec,       Vec:  40% FPR       Vec:  50% FPR     Vec:  51% FPR,
         6% FPR         Full:  0% FPR       Full:  0% FPR           56% recall
   Full: 43% rec,                                             Full:  0% FPR,
         4% FPR                                                     67% recall

   Value: modest        Value: categorical  Value: categ.     Value: categ. safety
          FPR gain              safety              safety            + same-embedding
          + judge              0.96-sim            0.97-sim           proof: trap & para
          + observ.            rejected            rejected           distributions overlap
```

| Question | Quora | Healthcare | Finance | Banking |
|----------|-------|------------|---------|---------|
| Is similarity-only viable? | Often yes, ~2 pp more FPR | **No** — 40% FPR | **No** — 50% FPR | **No** — 51% FPR *and* lower recall |
| Does our stack add value? | Incremental | **Decisive** | **Decisive** | **Decisive** |
| Is low recall a problem? | Tunable; label noise | Needs gray-zone calibration | Needs gray-zone calibration | Stack out-recalls baseline here |
| What should we optimize? | Pooled recall/FPR | Trap regression suite | Trap regression suite | Facet-gate over-rejection |

---

## What “honest” semantic caching means — and why production users want it

Most real-world deployments are not optimizing for “cache everything that embeds close.” They are optimizing for **trust**:

1. **Fail closed on ambiguity** — If two queries might need different answers, go live. A missed cache hit costs latency and money; a wrong cache hit costs credibility, compliance, or a bad user action.
2. **Explain every decision** — Routing label, similarity score, reject reason, judge outcome. Ops teams and auditors need traces, not a black-box cosine score.
3. **Separate risk tiers** — FAQ paraphrases, patient-specific records, and “should I sell?” prompts do not belong on the same reuse policy. Honest caching routes them differently before retrieval.
4. **Calibrate per domain** — Accept ~43% recall on Quora if finance and healthcare traps stay at 0% FPR. Tune thresholds and judge behavior against adversarial suites, not vanity hit-rate metrics alone.

That is the product shape our benchmarks support: **conservative defaults, categorical safety on tails, calibratable gray zone**. Similarity-only caching optimizes hit rate on average questions. Honest caching optimizes **correctness on the questions that matter most** — which is what healthcare, finance, legal, and enterprise ops users actually deploy for.

GPTCache-style prototypes feel good in demos because Quora-like paraphrases dominate toy examples. Production traffic includes the finance and healthcare tails **by the hour**. Our evaluation program measures both on purpose.

---

## Articulating our value proposition

### 1. Similarity is necessary but not sufficient

Every mode we tested uses the same embedding model and ANN index. The retrieval layer is table stakes. Differentiation is **everything that happens after retrieval** — when similarity lies.

### 2. Safety is in the tail, not the mean

Quora averages hide catastrophic failures. A 6% FPR on random questions becomes **40–50% FPR** when questions differ only in patient ID, trade direction, account type, or tax polarity. Production systems encounter these tails constantly in healthcare, finance, legal, and ops — not as edge cases, but as core workload.

### 3. Layered defense beats one magic threshold

No single cosine cutoff solves reuse. Our stack combines:

- **Routing** — don’t cache what shouldn’t be cached (freshness, PHI, thread-bound records).
- **Scoped retrieval** — namespace and version isolation before similarity scoring.
- **Facet gates** — deterministic checks for entity, quantity, and negation conflicts.
- **Bounded judge** — human-like disambiguation in the gray zone only, with cost and latency caps.

Vector-only collapses this into one number. That number worked until it didn’t — at 0.96.

### 4. Conservative by default, calibratable by domain

~43% Quora recall reflects safety-first defaults, not a ceiling. The judge adds +6.6 pp recall on pooled data; seed 45 shows +11 pp with *lower* FPR. Healthcare shows we can hold **zero FPR** on traps while missing two safe paraphrases — calibration can shift that balance per tenant, namespace, or route class.

### 5. Observable, fail-safe, domain-agnostic core

Every decision emits trace fields: routing label, similarity, reject reason, judge invocation, facet conflict type. On miss or judge failure, the system goes live — never silently serving a wrong cached answer. The engine stays domain-agnostic; **domain risk is expressed through routing labels, scope keys, and calibration**, not hard-coded healthcare rules in the core.

---

## Comparative positioning vs GPTCache-style caching

| Capability | GPTCache-style (vector-only) | Our `next/` design |
|------------|------------------------------|---------------------|
| Embedding retrieval | Yes | Yes |
| Exact cache | Optional / separate | First-class, ordered first |
| Pre-retrieval routing | No | Classifier (`SKIP_CACHE`, etc.) |
| Post-retrieval facet gates | No | Yes (entity, qty, negation) |
| Structured scope / metadata filters | Limited | Namespace, version, reuse scope |
| Gray-zone disambiguation | No | Bounded neighbor judge |
| Fail-safe on judge error | N/A | Live answer |
| Quora FPR (1,000 pairs) | 6.2% | **4.2%** |
| Healthcare trap FPR (12 pairs) | **40%** | **0%** |
| Finance trap FPR (32 pairs) | **50%** | **0%** |
| Banking trap FPR (94 candidates) | **51%** | **0%** |
| Banking paraphrase recall | 56% | **67%** |

**When GPTCache-style is enough:** Low-risk FAQ, developer tools, internal docs with uniform freshness, workloads where a wrong reuse is cheap to detect downstream.

**When our design is warranted:** Patient-specific data, regulated content, operational freshness, financial and tax advice, trade execution context — any domain where **near-duplicate ≠ same answer** and wrong reuse has asymmetric cost.

---

## Open calibration work (honest next steps)

Our evaluation is strong on *measuring* the tradeoff. Remaining work is *tuning* it:

1. **Threshold calibration from dual benchmarks** — optimize recall on Quora while holding healthcare trap FPR at zero (or a defined ε).
2. **Gray-zone judge tuning** — reduce `intent_change` over-rejection (39% of Quora duplicate misses) without reopening trap FPR.
3. **Routing vs forced `semantic_ok`** — Quora used forced semantic routing; production uses honest routing which blocked several healthcare probes before retrieval.
4. **Bidirectional pair probing** — seed both directions to reduce paraphrase asymmetry in recall metrics.
5. **Expand trap suites** — finance (32 pairs, done), retail banking (94 candidates, done; every trap type n ≥ 8), legal (jurisdiction), ops (environment-specific). The banking suite uses the standalone same-embedding baseline in `eval/banking_adversarial_eval.py`.

---

## Recommended headline metrics for external communication

When speaking to different audiences, lead with the benchmark that matches their risk tolerance:

**For platform / general audience:**

> On 1,000 human-labeled Quora paraphrase pairs, our semantic cache achieves **91% precision** and **4.2% false-positive rate** at 0.86 cosine threshold — versus **88.6% precision** and **6.2% FPR** for similarity-only caching, which wins recall (48% vs 43%) by accepting more wrong hits.

**For regulated / high-stakes audience (healthcare):**

> On 12 curated healthcare near-duplicate pairs where wrong reuse is unacceptable, similarity-only caching produced **4 false positives (40% FPR)**, including a **0.96 cosine-similarity** Type 1 vs Type 2 diabetes trap. Our full stack produced **zero false positives**, blocking every trap via routing, facet gates, and bounded judge review.

**For finance / action-bearing workloads:**

> On 32 personal-finance adversarial pairs (buy/sell, long/short, account-type swaps, tax polarity), similarity-only caching produced **16 false positives (50% FPR)**, including traps above **0.97 similarity** (reversed IRA rollover direction). Our honest router + judge stack produced **zero false positives (32/32 pass)**.

**For banking / the "just tune the threshold" skeptic:**

> On a 94-candidate retail-banking suite scored on identical embeddings, a cosine-only cache had a **51% false-reuse rate and 56% recall**; our stack had **0% false reuse and 67% recall** — better on both axes. The reason a single threshold cannot match this: the trap and paraphrase similarity distributions interleave — **30 of 67 traps were more cosine-similar to their seed than the median true paraphrase**, with traps reaching 0.997 (a one-digit identifier swap). No cutoff separates safe reuse from dangerous reuse; the decision has to come from routing, structure, and intent, not distance.

**Unified value statement:**

> Semantic caching is easy to prototype and hard to productionize. Embeddings find similar questions; they do not distinguish safe similarity from dangerous similarity. Our design keeps vector retrieval as the engine and adds the minimum viable control plane — route, scope, gate, judge — to make reuse safe in domains where the tail risk is not average, but catastrophic.

---

## Reproducibility

```bash
cd next

# Quora full suite (1,000 pairs)
for seed in 42 43 44; do
  gated-semantic-cache eval quora-pairs --limit 200 --seed $seed \
    --report-json docs/quora_pairs_eval/quora_pairs_200_seed${seed}_judge-on.json
  gated-semantic-cache eval quora-pairs --limit 200 --seed $seed --no-judge \
    --report-json docs/quora_pairs_eval/quora_pairs_200_seed${seed}_no-judge.json
  gated-semantic-cache eval quora-pairs --limit 200 --seed $seed --route-policy vector_only \
    --report-json docs/quora_pairs_eval/quora_pairs_200_seed${seed}_vector-only.json
done
gated-semantic-cache eval quora-pairs --limit 400 --seed 45 \
  --report-json docs/quora_pairs_eval/quora_pairs_400_seed45_judge-on.json

# Healthcare trap suite (12 pairs)
gated-semantic-cache eval queries-pairs --route-policy honest \
  --report-json docs/queries_pairs_eval/queries_pairs_honest_judge-on.json
gated-semantic-cache eval queries-pairs --route-policy vector_only \
  --report-json docs/queries_pairs_eval/queries_pairs_vector-only.json

# Finance adversarial suite (32 pairs)
FIXTURE=tests/fixtures/finance_adversarial_pairs.json
gated-semantic-cache eval queries-pairs --pairs-json "$FIXTURE" --route-policy honest \
  --report-json docs/finance_pairs_eval/finance_adversarial_honest_judge-on.json
gated-semantic-cache eval queries-pairs --pairs-json "$FIXTURE" --route-policy vector_only \
  --report-json docs/finance_pairs_eval/finance_adversarial_vector-only.json

# Retail banking adversarial suite (full stack + built-in cosine-only baseline)
python3 -m gated_semantic_cache.eval.banking_adversarial_eval --suite full100 \
  --report-json docs/banking_adversarial_report_full100.json
# 32-candidate core subset:
python3 -m gated_semantic_cache.eval.banking_adversarial_eval --suite core32 \
  --report-json docs/banking_adversarial_report.json
```

**Primary artifacts:**

| Report | Path |
|--------|------|
| Quora pooled 1,000 | `docs/quora_pairs_eval/quora_pairs_pooled_1000.md` |
| GPTCache baseline 1,000 | `docs/quora_pairs_eval/quora_pairs_gptcache_baseline_1000.md` |
| Quora miss analysis | `docs/quora_pairs_eval/quora_pairs_pooled_600_seeds42_43_44.md` |
| Healthcare comparison | `docs/queries_pairs_eval/queries_pairs_comparison.md` |
| Finance adversarial comparison | `docs/finance_pairs_eval/finance_adversarial_comparison.md` |
| Retail banking adversarial — 94 candidates (full stack + cosine baseline) | `docs/banking_adversarial_report_full100.json` |
| Retail banking adversarial — 32-candidate core subset | `docs/banking_adversarial_report.json` |
| This summary | `docs/eval_summary_report.md` |

---

*Report compiled from evaluation runs dated 2026-05-24; retail-banking benchmark (Benchmark 4) added 2026-06-14. Embedding model: OpenAI `text-embedding-3-small`. Threshold: 0.86 (full stack), 0.85 (banking cosine baseline). Judge: enabled per environment defaults unless noted.*
