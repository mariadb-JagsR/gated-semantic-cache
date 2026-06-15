from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from gatecache.routing.features import EngineeredFeatureTransformer, normalize_query_text
from gatecache.routing.labels import RoutingLabel, parse_routing_label


@dataclass(slots=True)
class RoutingPrediction:
    label: RoutingLabel
    confidence: float
    probabilities: dict[RoutingLabel, float]


def _build_feature_union() -> FeatureUnion:
    return FeatureUnion(
        transformer_list=[
            (
                "word_tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_query_text,
                    ngram_range=(1, 2),
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_query_text,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                ),
            ),
            (
                "engineered",
                Pipeline(
                    steps=[
                        ("extract", EngineeredFeatureTransformer()),
                        ("vectorize", DictVectorizer()),
                    ]
                ),
            ),
        ]
    )


def build_training_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("features", _build_feature_union()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                ),
            ),
        ]
    )


class RoutingClassifier:
    def __init__(self, pipeline: Pipeline | None = None) -> None:
        self.pipeline = pipeline or build_training_pipeline()

    def fit(self, queries: list[str], labels: list[str | RoutingLabel]) -> "RoutingClassifier":
        target = [parse_routing_label(label).value for label in labels]
        self.pipeline.fit(queries, target)
        return self

    def predict(self, query: str) -> RoutingPrediction:
        label_raw = self.pipeline.predict([query])[0]
        probabilities_raw = self.pipeline.predict_proba([query])[0]
        classifier = self.pipeline.named_steps["classifier"]
        classes = [RoutingLabel(value) for value in classifier.classes_]
        probabilities = {label: float(score) for label, score in zip(classes, probabilities_raw, strict=True)}
        prediction = RoutingLabel(label_raw)
        return RoutingPrediction(
            label=prediction,
            confidence=max(probabilities.values()),
            probabilities=probabilities,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(pickle.dumps(self.pipeline))

    @classmethod
    def load(cls, path: str | Path) -> "RoutingClassifier":
        pipeline = pickle.loads(Path(path).read_bytes())
        if not isinstance(pipeline, Pipeline):
            raise TypeError("Serialized routing classifier did not contain a sklearn Pipeline")
        return cls(pipeline=pipeline)

    def predict_many(self, queries: list[str]) -> list[RoutingPrediction]:
        return [self.predict(query) for query in queries]


def train_default_classifier(examples: list[Any]) -> RoutingClassifier:
    classifier = RoutingClassifier()
    classifier.fit(
        [example.query for example in examples],
        [example.label for example in examples],
    )
    return classifier
