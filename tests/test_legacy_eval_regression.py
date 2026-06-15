"""Regression coverage using prompts from `legacy/eval/` (datasets + JSON artifacts)."""

from __future__ import annotations

import re

import pytest

from gated_semantic_cache.eval.legacy_bridge import (
    iter_legacy_dataset_cases,
    iter_legacy_retrieval_ablation_cases,
    load_legacy_datasets_module,
    unique_legacy_dataset_messages,
    unique_user_messages_from_legacy_eval_jsons,
)
from gated_semantic_cache.routing.classifier import train_default_classifier
from gated_semantic_cache.routing.labels import RoutingLabel
from gated_semantic_cache.eval.datasets import build_routing_dataset
from gated_semantic_cache.structured_exact.structured_query import extract_structured_query


@pytest.fixture(scope="module")
def routing_classifier():
    return train_default_classifier(build_routing_dataset())


_LEGACY_MUTATION_FOLLOWUP_RE = re.compile(r"(?is)^\s*(now do same|also for)\b")
_LEGACY_BRANCH_THREAD_RE = re.compile(r"(?is)^\s*(now do same|go back to|the first one)\b")


@pytest.fixture(scope="module")
def legacy_retrieval_ablation_cases():
    cases = list(iter_legacy_retrieval_ablation_cases())
    assert len(cases) >= 80
    return cases


def test_legacy_datasets_module_available() -> None:
    assert load_legacy_datasets_module() is not None, "expected tests/fixtures/legacy_eval/datasets.py"


def test_legacy_bypass_turns_are_skip_cache(routing_classifier) -> None:
    """Legacy `expected_bypass` marks cache bypass; routing should not send these down semantic/exact reuse."""
    mod = load_legacy_datasets_module()
    assert mod is not None
    cases = list(iter_legacy_dataset_cases())
    assert len(cases) >= 100
    bad: list[str] = []
    for c in cases:
        if not c.expected_bypass:
            continue
        if routing_classifier.predict(c.user_message).label is not RoutingLabel.SKIP_CACHE:
            bad.append(c.user_message)
    assert not bad, f"bypass cases not SKIP_CACHE: {bad[:10]}"


def test_legacy_retrieval_ablation_lookup_sessions_are_exact_only(
    routing_classifier, legacy_retrieval_ablation_cases
) -> None:
    """Legacy `ra_lookup_*` sessions are anchored order/customer lookups → `EXACT_ONLY` in the new router."""
    bad: list[str] = []
    for c in legacy_retrieval_ablation_cases:
        if not c.session_id.startswith("ra_lookup"):
            continue
        if routing_classifier.predict(c.user_message).label is not RoutingLabel.EXACT_ONLY:
            bad.append(c.user_message)
    assert not bad, f"ra_lookup not EXACT_ONLY: {bad}"


def test_legacy_retrieval_ablation_mutation_followups_are_thread_scoped(
    routing_classifier, legacy_retrieval_ablation_cases
) -> None:
    """Short region pivots from legacy mutation sessions → `THREAD_SCOPED_ONLY`."""
    bad: list[str] = []
    for c in legacy_retrieval_ablation_cases:
        if not c.session_id.startswith("ra_mutation"):
            continue
        if not _LEGACY_MUTATION_FOLLOWUP_RE.match(c.user_message):
            continue
        if routing_classifier.predict(c.user_message).label is not RoutingLabel.THREAD_SCOPED_ONLY:
            bad.append(c.user_message)
    assert not bad, f"mutation follow-up not THREAD_SCOPED_ONLY: {bad}"


def test_legacy_retrieval_ablation_branch_thread_turns(
    routing_classifier, legacy_retrieval_ablation_cases
) -> None:
    """Legacy `ra_branch_1` follow-ups (`go back to`, `the first one`, …) depend on thread state."""
    bad: list[str] = []
    for c in legacy_retrieval_ablation_cases:
        if not c.session_id.startswith("ra_branch"):
            continue
        if not _LEGACY_BRANCH_THREAD_RE.match(c.user_message):
            continue
        if routing_classifier.predict(c.user_message).label is not RoutingLabel.THREAD_SCOPED_ONLY:
            bad.append(c.user_message)
    assert not bad, f"ra_branch thread turn not THREAD_SCOPED_ONLY: {bad}"


def test_structured_exact_smoke_on_legacy_dataset_messages() -> None:
    msgs = unique_legacy_dataset_messages()
    assert len(msgs) >= 150
    for m in msgs:
        sq = extract_structured_query(m)
        assert sq.normalized_text
        assert 0.0 <= sq.confidence <= 1.0


def test_structured_exact_smoke_on_legacy_eval_json_messages() -> None:
    msgs = unique_user_messages_from_legacy_eval_jsons()
    assert len(msgs) >= 50
    for m in msgs:
        sq = extract_structured_query(m)
        assert sq.normalized_text
        assert 0.0 <= sq.confidence <= 1.0
