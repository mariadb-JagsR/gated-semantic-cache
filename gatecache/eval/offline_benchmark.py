from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from gatecache.eval.datasets import RoutingExample, build_routing_dataset
from gatecache.routing.classifier import RoutingClassifier, train_default_classifier
from gatecache.routing.labels import ALL_ROUTING_LABELS, RoutingLabel


@dataclass(slots=True)
class BenchmarkReport:
    total_examples: int
    class_distribution: dict[str, int]
    average_precision_by_label: dict[str, float]
    average_recall_by_label: dict[str, float]
    average_f1_by_label: dict[str, float]
    confusion_matrix: dict[str, dict[str, int]]
    routing_latency_ms_p50: float
    routing_latency_ms_p95: float
    confidence_by_label: dict[str, float]
    semantic_route_rate: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class MisclassificationRecord:
    fold: int
    query: str
    expected_label: str
    predicted_label: str
    confidence: float
    slice_id: str
    source: str
    notes: str
    namespace_policy: str


@dataclass(slots=True)
class ErrorAnalysisReport:
    total_examples: int
    total_misclassified: int
    overall_error_rate: float
    confusion_pairs: dict[str, int]
    top_confusions: list[dict[str, object]]
    skip_cache_confusions: list[dict[str, object]]
    misclassified_examples: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def benchmark_routing_classifier(
    examples: list[RoutingExample] | None = None,
    *,
    folds: int = 4,
) -> BenchmarkReport:
    data = examples or build_routing_dataset()
    queries = [example.query for example in data]
    labels = [example.label.value for example in data]

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=7)
    precision_acc: dict[str, list[float]] = defaultdict(list)
    recall_acc: dict[str, list[float]] = defaultdict(list)
    f1_acc: dict[str, list[float]] = defaultdict(list)
    confidence_acc: dict[str, list[float]] = defaultdict(list)
    latency_samples: list[float] = []
    confusion_counts = [[0 for _ in ALL_ROUTING_LABELS] for _ in ALL_ROUTING_LABELS]
    semantic_predictions = 0
    total_predictions = 0

    for train_idx, test_idx in splitter.split(queries, labels):
        train_examples = [data[index] for index in train_idx]
        classifier = train_default_classifier(train_examples)

        fold_truth = [labels[index] for index in test_idx]
        fold_pred: list[str] = []
        for index in test_idx:
            t0 = time.perf_counter()
            prediction = classifier.predict(queries[index])
            latency_samples.append((time.perf_counter() - t0) * 1000)
            fold_pred.append(prediction.label.value)
            confidence_acc[prediction.label.value].append(prediction.confidence)
            semantic_predictions += int(prediction.label is RoutingLabel.SEMANTIC_OK)
            total_predictions += 1

        precision, recall, f1, _ = precision_recall_fscore_support(
            fold_truth,
            fold_pred,
            labels=list(ALL_ROUTING_LABELS),
            zero_division=0.0,
        )
        for idx, label in enumerate(ALL_ROUTING_LABELS):
            precision_acc[label].append(float(precision[idx]))
            recall_acc[label].append(float(recall[idx]))
            f1_acc[label].append(float(f1[idx]))

        fold_confusion = confusion_matrix(fold_truth, fold_pred, labels=list(ALL_ROUTING_LABELS))
        for row_idx, row in enumerate(fold_confusion):
            for col_idx, value in enumerate(row):
                confusion_counts[row_idx][col_idx] += int(value)

    return BenchmarkReport(
        total_examples=len(data),
        class_distribution=dict(Counter(labels)),
        average_precision_by_label=_mean_map(precision_acc),
        average_recall_by_label=_mean_map(recall_acc),
        average_f1_by_label=_mean_map(f1_acc),
        confusion_matrix=_format_confusion_matrix(confusion_counts),
        routing_latency_ms_p50=_percentile(latency_samples, 50),
        routing_latency_ms_p95=_percentile(latency_samples, 95),
        confidence_by_label={label: round(statistics.mean(values), 4) for label, values in confidence_acc.items()},
        semantic_route_rate=round((semantic_predictions / total_predictions), 4) if total_predictions else 0.0,
    )


def analyze_routing_errors(
    examples: list[RoutingExample] | None = None,
    *,
    folds: int = 4,
    max_examples: int = 40,
) -> ErrorAnalysisReport:
    data = examples or build_routing_dataset()
    queries = [example.query for example in data]
    labels = [example.label.value for example in data]
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=7)

    mistakes: list[MisclassificationRecord] = []
    pair_counts: Counter[str] = Counter()

    for fold_num, (train_idx, test_idx) in enumerate(splitter.split(queries, labels), start=1):
        train_examples = [data[index] for index in train_idx]
        classifier = train_default_classifier(train_examples)
        for index in test_idx:
            example = data[index]
            prediction = classifier.predict(example.query)
            expected_label = example.label.value
            predicted_label = prediction.label.value
            if predicted_label == expected_label:
                continue
            pair_key = f"{expected_label} -> {predicted_label}"
            pair_counts[pair_key] += 1
            mistakes.append(
                MisclassificationRecord(
                    fold=fold_num,
                    query=example.query,
                    expected_label=expected_label,
                    predicted_label=predicted_label,
                    confidence=round(prediction.confidence, 4),
                    slice_id=example.slice_id,
                    source=example.source,
                    notes=example.notes,
                    namespace_policy=example.namespace_policy,
                )
            )

    mistakes.sort(key=lambda item: (-item.confidence, item.expected_label, item.predicted_label, item.query))
    total_misclassified = len(mistakes)

    return ErrorAnalysisReport(
        total_examples=len(data),
        total_misclassified=total_misclassified,
        overall_error_rate=round(total_misclassified / len(data), 4) if data else 0.0,
        confusion_pairs=dict(pair_counts.most_common()),
        top_confusions=[
            {"pair": pair, "count": count} for pair, count in pair_counts.most_common(10)
        ],
        skip_cache_confusions=[
            asdict(item) for item in mistakes if item.expected_label == RoutingLabel.SKIP_CACHE.value
        ][: max_examples // 2],
        misclassified_examples=[asdict(item) for item in mistakes[:max_examples]],
    )


def fit_and_save_classifier(output_path: str | Path, examples: list[RoutingExample] | None = None) -> RoutingClassifier:
    classifier = train_default_classifier(examples or build_routing_dataset())
    classifier.save(output_path)
    return classifier


def inspect_queries(queries: list[str], classifier_path: str | Path | None = None) -> list[dict[str, object]]:
    classifier = RoutingClassifier.load(classifier_path) if classifier_path else train_default_classifier(build_routing_dataset())
    results: list[dict[str, object]] = []
    for query in queries:
        prediction = classifier.predict(query)
        results.append(
            {
                "query": query,
                "label": prediction.label.value,
                "confidence": round(prediction.confidence, 4),
                "probabilities": {label.value: round(score, 4) for label, score in prediction.probabilities.items()},
            }
        )
    return results


def _mean_map(values: dict[str, list[float]]) -> dict[str, float]:
    return {key: round(statistics.mean(items), 4) for key, items in values.items()}


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((p / 100.0) * (len(ordered) - 1)))
    return round(ordered[index], 4)


def _format_confusion_matrix(matrix: list[list[int]]) -> dict[str, dict[str, int]]:
    formatted: dict[str, dict[str, int]] = {}
    for row_label, row in zip(ALL_ROUTING_LABELS, matrix, strict=True):
        formatted[row_label] = {
            col_label: value for col_label, value in zip(ALL_ROUTING_LABELS, row, strict=True)
        }
    return formatted


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the routing classifier benchmark or inspect sample queries.")
    parser.add_argument("--save-model", help="Optional path to write a trained classifier artifact")
    parser.add_argument("--report-json", help="Optional path to write benchmark metrics")
    parser.add_argument("--error-report-json", help="Optional path to write fold-level misclassification analysis")
    parser.add_argument("--inspect-query", action="append", default=[], help="Query string to inspect with the classifier")
    args = parser.parse_args()

    report = benchmark_routing_classifier()
    error_report = analyze_routing_errors()
    if args.save_model:
        fit_and_save_classifier(args.save_model)
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report.to_dict(), indent=2))
    if args.error_report_json:
        Path(args.error_report_json).write_text(json.dumps(error_report.to_dict(), indent=2))
    print(json.dumps(report.to_dict(), indent=2))

    if args.inspect_query:
        print(json.dumps(inspect_queries(args.inspect_query, args.save_model), indent=2))


if __name__ == "__main__":
    main()
