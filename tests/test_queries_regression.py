"""Regression expectations for ``tests/queries.txt`` (routing labels on training distribution)."""

from pathlib import Path

import pytest

from gatecache.routing.classifier import train_default_classifier
from gatecache.routing.labels import RoutingLabel
from gatecache.eval.datasets import build_routing_dataset


def _queries() -> list[str]:
    path = Path(__file__).resolve().parent / "queries.txt"
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


@pytest.fixture(scope="module")
def router():
    return train_default_classifier(build_routing_dataset())


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Summarize the HIPAA guidelines for patient data sharing.", RoutingLabel.SKIP_CACHE),
        ("What is the current wait time at the St. Jude's ER right now?", RoutingLabel.SKIP_CACHE),
        ("Show me the last lab results for patient ID: 88291.", RoutingLabel.EXACT_ONLY),
        ("What were the laboratory findings for patient ID: 11022?", RoutingLabel.EXACT_ONLY),
        ("Can you retrieve the discharge summary for John Doe?", RoutingLabel.THREAD_SCOPED_ONLY),
        ("I need the discharge summary for Jane Smith.", RoutingLabel.THREAD_SCOPED_ONLY),
        ("List the symptoms of a pulmonary embolism.", RoutingLabel.SEMANTIC_OK),
        ("Which patients are ineligible for the clinical trial?", RoutingLabel.SKIP_CACHE),
    ],
)
def test_queries_txt_routing_expectations(router, query: str, expected: RoutingLabel) -> None:
    assert router.predict(query).label is expected


def test_queries_txt_file_non_empty() -> None:
    assert len(_queries()) >= 10
