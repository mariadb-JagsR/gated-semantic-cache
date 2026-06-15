# Personal finance adversarial pairs — full stack vs GPTCache-style

32 seed/probe pairs from [finance_adversarial_pairs.json](../../tests/fixtures/finance_adversarial_pairs.json). Every pair is labeled **must miss** — near-duplicates where reusing the cached answer would mislead (buy/sell, long/short, account types, tax polarity, etc.).

Threshold: **0.86**, model: **text-embedding-3-small**, judge **ON**.

## A/B summary

| Mode | False positives | FPR | Pair pass rate |
|------|-----------------|-----|----------------|
| **Honest router + judge (our stack)** | **0** | **0%** | **32/32** |
| Vector-only (GPTCache-style) | **16** | **50%** | 16/32 |

**Half of all finance traps fool pure similarity caching at 0.86.** Our stack blocks every one.

Reports: [honest+judge](./finance_adversarial_honest_judge-on.json), [vector-only](./finance_adversarial_vector-only.json).

## Vector-only false positives (16)

| Pair | Similarity | Category |
|------|------------|----------|
| Roth IRA ↔ Traditional IRA conversion (reversed) | **0.975** | Roth conversion direction |
| 401(k) → IRA vs IRA → 401(k) rollover | **0.968** | Account type swaps |
| Buy vs sell Tesla | 0.946 | Buy vs sell |
| Call vs put option (Microsoft) | 0.923 | Options direction |
| Unrealized vs realized gain (AAPL) | 0.922 | Risk direction |
| Short-term vs long-term capital gains rate | 0.920 | Tax-event distinctions |
| Buy vs sell bonds timing | 0.912 | Buy vs sell |
| Long vs short oil futures | 0.901 | Long vs short |
| Roth vs Traditional IRA withdrawal | 0.889 | Account type swaps |
| Market vs limit order (Apple) | 0.886 | Order type |
| Index funds vs mutual funds comparison | 0.884 | Asset class swap |
| Long vs short S&P 500 | 0.884 | Long vs short |
| Pre-tax vs Roth 401(k) contribution | 0.883 | Pre-tax vs post-tax |
| Margin lending vs borrowing | 0.878 | Borrow vs lend |
| Refinance when rates down vs up | 0.868 | Interest rate direction |
| Inherited vs gifted stock basis | 0.867 | Inherited vs gifted |

Several traps exceed **0.96 similarity** — including reversed IRA rollover direction at **0.968** and Roth conversion direction at **0.975**.

## How our stack blocked them

| Mechanism | Pairs blocked (of 16 vector FPs) |
|-----------|----------------------------------|
| Gray-zone **judge** (`intent_change`, `intent_differs`, etc.) | 11 |
| **Query facet** gates (entity/token conflict) | 3 |
| **Routing** (`SKIP_CACHE`) | 2 |

Examples:

- **Roth conversion 0.975 sim** — judge `intent_change`
- **401(k)/IRA rollover reversal 0.968 sim** — judge `intent_change`
- **Roth vs Traditional withdrawal** — facet `query_facet_named_entity_conflict`
- **Short-term vs long-term gains** — facet `query_facet_protected_token_conflict`
- **Buy vs sell Tesla 0.946 sim** — router `SKIP_CACHE` (never enters semantic path)

## Traps both modes correctly missed (16)

Vector-only missed the other 16 because similarity stayed **below 0.86** (e.g. owe vs refund 0.79, wash sale 0.78, ETF vs ETN 0.77, hedge vs profit 0.76). **Threshold alone is not a reliable safety layer** — half the traps cleared it.

## Comparison to healthcare benchmark

| Benchmark | Pairs | Vector-only FPR | Full stack FPR |
|-----------|-------|-----------------|----------------|
| Healthcare (`queries.txt`) | 12 | 40% (4/10 traps) | **0%** |
| **Personal finance (this set)** | **32** | **50% (16/32)** | **0%** |

Finance adversarial pairs are at least as dangerous for similarity-only caching as healthcare traps, with a **higher** observed FPR on this slice.

## Reproduce

```bash
cd next
FIXTURE=tests/fixtures/finance_adversarial_pairs.json

gatecache eval queries-pairs --pairs-json "$FIXTURE" --route-policy honest \
  --report-json docs/finance_pairs_eval/finance_adversarial_honest_judge-on.json

gatecache eval queries-pairs --pairs-json "$FIXTURE" --route-policy vector_only \
  --report-json docs/finance_pairs_eval/finance_adversarial_vector-only.json
```

Run date: 2026-05-24.
