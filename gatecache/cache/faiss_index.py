from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np


@dataclass(slots=True)
class FaissCandidate:
    cache_id: str
    similarity: float


class FaissHnswIndex:
    def __init__(self, dimension: int, m: int = 32) -> None:
        self.dimension = dimension
        self._index = faiss.IndexHNSWFlat(dimension, m, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efConstruction = 40
        self._index.hnsw.efSearch = 32
        self._cache_ids: list[str] = []

    @classmethod
    def from_deserialized(
        cls,
        *,
        dimension: int,
        index: faiss.Index,
        cache_ids: list[str],
        m: int = 32,
    ) -> FaissHnswIndex:
        """Wrap a Faiss index restored from disk (IDs must match insertion order)."""

        obj = object.__new__(cls)
        obj.dimension = dimension
        obj._index = index
        obj._cache_ids = list(cache_ids)
        return obj

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal)

    @property
    def cache_ids(self) -> list[str]:
        return list(self._cache_ids)

    def add(self, cache_id: str, vector: list[float]) -> None:
        normalized = _normalize(vector)
        self._index.add(np.array([normalized], dtype=np.float32))
        self._cache_ids.append(cache_id)

    def search(self, vector: list[float], top_k: int) -> list[FaissCandidate]:
        if not self._cache_ids:
            return []
        k = min(top_k, len(self._cache_ids))
        distances, indices = self._index.search(np.array([_normalize(vector)], dtype=np.float32), k)
        candidates: list[FaissCandidate] = []
        for similarity, idx in zip(distances[0], indices[0], strict=True):
            if idx < 0:
                continue
            candidates.append(FaissCandidate(cache_id=self._cache_ids[idx], similarity=float(similarity)))
        return candidates


def _normalize(vector: list[float]) -> np.ndarray:
    arr = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    return arr / norm
