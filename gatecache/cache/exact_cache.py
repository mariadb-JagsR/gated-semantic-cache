from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ExactCacheValue:
    payload: dict[str, Any]
    source: str = "exact_cache"


class ExactCache:
    def __init__(self) -> None:
        self._values: dict[str, ExactCacheValue] = {}

    def get(self, key: str) -> ExactCacheValue | None:
        return self._values.get(key)

    def put(self, key: str, value: ExactCacheValue) -> None:
        self._values[key] = value

    def replace_all(self, values: dict[str, ExactCacheValue]) -> None:
        """Bulk replace for hydration from durable storage."""

        self._values.clear()
        self._values.update(values)

    def __len__(self) -> int:
        return len(self._values)
