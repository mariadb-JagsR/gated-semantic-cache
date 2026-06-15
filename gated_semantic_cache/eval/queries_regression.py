"""Regression report over ``tests/queries.txt`` (routing + optional full pipeline with OpenAI embeddings)."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gated_semantic_cache.eval.datasets import build_routing_dataset
from gated_semantic_cache.models.context import RequestContext
from gated_semantic_cache.routing.classifier import RoutingClassifier, train_default_classifier


@dataclass(slots=True)
class QueriesRegressionRow:
    query: str
    routing_label: str
    routing_confidence: float
    routing_probabilities: dict[str, float]
    source: str | None = None
    trace: dict[str, Any] | None = None
    total_latency_ms: float | None = None
    error: str | None = None


@dataclass(slots=True)
class QueriesRegressionReport:
    generated_at_unix: float
    queries_file: str | None
    classifier_path: str | None
    mode: str
    rows: list[QueriesRegressionRow]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at_unix": self.generated_at_unix,
            "queries_file": self.queries_file,
            "classifier_path": self.classifier_path,
            "mode": self.mode,
            "rows": [asdict(r) for r in self.rows],
        }


def _default_queries_path() -> Path:
    # gated_semantic_cache/eval/ -> parents[2] == repo root
    return Path(__file__).resolve().parents[2] / "tests" / "queries.txt"


def load_queries_from_file(path: Path) -> list[str]:
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def run_queries_regression_report(
    *,
    queries: list[str] | None = None,
    queries_file: Path | str | None = None,
    classifier_path: Path | str | None = None,
    mode: str = "routing",
    request_context: RequestContext | None = None,
) -> QueriesRegressionReport:
    """Build a JSON-serializable report.

    ``mode``:
    - ``routing`` — classifier predictions only (no OpenAI; deterministic).
    - ``pipeline`` — full ``SemanticCachePipeline`` with OpenAI embeddings (requires ``OPENAI_API_KEY``).
    """
    path = Path(queries_file) if queries_file is not None else _default_queries_path()
    qlist = list(queries) if queries is not None else load_queries_from_file(path)
    ctx = request_context or RequestContext()

    if classifier_path:
        router = RoutingClassifier.load(classifier_path)
    else:
        router = train_default_classifier(build_routing_dataset())

    rows: list[QueriesRegressionRow] = []
    t0 = time.time()

    if mode == "routing":
        for query in qlist:
            pred = router.predict(query)
            rows.append(
                QueriesRegressionRow(
                    query=query,
                    routing_label=pred.label.value,
                    routing_confidence=pred.confidence,
                    routing_probabilities={k.value: v for k, v in pred.probabilities.items()},
                )
            )
        return QueriesRegressionReport(
            generated_at_unix=t0,
            queries_file=str(path) if queries is None else None,
            classifier_path=str(classifier_path) if classifier_path else None,
            mode=mode,
            rows=rows,
        )

    if mode != "pipeline":
        raise ValueError(f"Unknown mode: {mode}")

    from gated_semantic_cache.cli import build_pipeline, default_live_answer, run_single_query

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("pipeline mode requires OPENAI_API_KEY")

    pipeline = build_pipeline()
    for query in qlist:
        try:
            out = run_single_query(query, context=ctx, pipeline=pipeline, live_answer=default_live_answer)
            tr = out["trace"]
            rows.append(
                QueriesRegressionRow(
                    query=query,
                    routing_label=str(tr.get("routing_label") or ""),
                    routing_confidence=float(tr.get("routing_confidence") or 0.0),
                    routing_probabilities={},
                    source=str(out.get("source")),
                    trace=tr if isinstance(tr, dict) else None,
                    total_latency_ms=float(out.get("total_latency_ms") or 0.0),
                )
            )
        except Exception as e:
            rows.append(
                QueriesRegressionRow(
                    query=query,
                    routing_label="",
                    routing_confidence=0.0,
                    routing_probabilities={},
                    error=str(e),
                )
            )

    return QueriesRegressionReport(
        generated_at_unix=t0,
        queries_file=str(path) if queries is None else None,
        classifier_path=str(classifier_path) if classifier_path else None,
        mode=mode,
        rows=rows,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Regression JSON report for tests/queries.txt")
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=_default_queries_path(),
        help="Default: next/tests/queries.txt",
    )
    parser.add_argument("--classifier", type=Path, default=None, help="Optional saved router .pkl")
    parser.add_argument(
        "--mode",
        choices=("routing", "pipeline"),
        default="routing",
        help="routing: classifier only; pipeline: OpenAI embeddings + full handler (needs key)",
    )
    parser.add_argument("--output-json", "-o", type=Path, default=None)
    parser.add_argument("--pretty", action="store_true", default=True)
    args = parser.parse_args(argv)

    report = run_queries_regression_report(
        queries_file=args.queries_file,
        classifier_path=args.classifier,
        mode=args.mode,
    )
    text = json.dumps(report.to_dict(), indent=2 if args.pretty else None)
    if args.output_json:
        args.output_json.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
