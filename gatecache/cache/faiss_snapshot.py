"""On-disk FAISS snapshot helpers (namespace-scoped files beside the SQLite DB)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import faiss


def namespace_tag(namespace: str) -> str:
    return hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:16]


def faiss_paths(db_path: Path, namespace: str) -> tuple[Path, Path]:
    """Return (faiss_index_path, cache_ids_json_path)."""
    tag = namespace_tag(namespace)
    base = db_path.parent / f"{db_path.stem}.ns.{tag}"
    return Path(str(base) + ".faiss"), Path(str(base) + ".ids.json")


def save_index(index: Any, cache_ids: list[str], faiss_path: Path, ids_path: Path) -> None:
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_path))
    ids_path.write_text(json.dumps(cache_ids), encoding="utf-8")


def load_index(faiss_path: Path, ids_path: Path) -> tuple[Any, list[str]] | None:
    if not faiss_path.is_file() or not ids_path.is_file():
        return None
    idx = faiss.read_index(str(faiss_path))
    cache_ids: list[str] = json.loads(ids_path.read_text(encoding="utf-8"))
    if idx.ntotal != len(cache_ids):
        return None
    return idx, cache_ids


def remove_files(faiss_path: Path, ids_path: Path) -> None:
    for p in (faiss_path, ids_path):
        try:
            p.unlink(missing_ok=True)
        except TypeError:
            if p.is_file():
                p.unlink()
