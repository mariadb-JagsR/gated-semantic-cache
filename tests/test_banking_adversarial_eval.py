"""Offline shape/label checks for the banking adversarial suites (no network)."""

from __future__ import annotations

from collections import Counter

from gatecache.eval.banking_adversarial_eval import (
    PARAPHRASE,
    SUITES,
    _category_of,
    build_banking_adversarial_scenarios,
    build_banking_adversarial_scenarios_100,
)


def _candidates(scenarios):
    return [t for sc in scenarios for t in sc.tests]


def test_core32_shape() -> None:
    cands = _candidates(build_banking_adversarial_scenarios())
    assert len(cands) == 32
    # every candidate carries a parseable "CATEGORY :: detail" note
    for t in cands:
        assert "::" in t.note and _category_of(t.note)


def test_full100_balanced_traps() -> None:
    cands = _candidates(build_banking_adversarial_scenarios_100())
    assert len(cands) >= 90
    by_cat = Counter(_category_of(t.note) for t in cands)
    trap_types = {k for k in by_cat if k != PARAPHRASE}
    # rebalanced goal: every trap type carried at n>=8 for meaningful per-type FPR
    for cat in trap_types:
        assert by_cat[cat] >= 8, (cat, by_cat[cat])


def test_label_category_consistency() -> None:
    # PARAPHRASE candidates must be should-reuse; every other category must be must-miss.
    for builder in (build_banking_adversarial_scenarios, build_banking_adversarial_scenarios_100):
        for t in _candidates(builder()):
            is_paraphrase = _category_of(t.note) == PARAPHRASE
            assert t.cache_hit is is_paraphrase, (t.note, t.cache_hit)


def test_suite_registry() -> None:
    assert set(SUITES) == {"core32", "full100"}
