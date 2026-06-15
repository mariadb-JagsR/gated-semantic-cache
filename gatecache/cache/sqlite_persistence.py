"""SQLite-backed cache entry persistence (development default).

Swap this module for a GridGain/JDBC or HTTP-backed adapter implementing the same
protocol in ``ports.py`` without changing the semantic cache engine logic.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gatecache.cache.faiss_index import FaissHnswIndex
from gatecache.cache.faiss_snapshot import faiss_paths, load_index, remove_files, save_index
from gatecache.cache.ports import CacheEntryPersistence, NamespaceHydrationBundle, VectorIndexPersistence
from gatecache.cache.semantic_store import SemanticStore
from gatecache.models.cache_entry import SemanticCacheEntry


SCHEMA_VERSION = "2"


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()


def _parse_dt(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    return datetime.fromisoformat(raw)


class SqliteCachePersistence(CacheEntryPersistence):
    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def init_schema(self) -> None:
        c = self._connection()
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS cache_meta (
              k TEXT PRIMARY KEY,
              v TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS exact_row (
              exact_key_sha256 TEXT PRIMARY KEY,
              namespace TEXT NOT NULL,
              scope_fingerprint TEXT,
              payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_exact_namespace ON exact_row(namespace);
            CREATE TABLE IF NOT EXISTS semantic_row (
              cache_id TEXT PRIMARY KEY,
              namespace TEXT NOT NULL,
              query_original TEXT NOT NULL,
              query_normalized TEXT NOT NULL,
              embedding_json TEXT NOT NULL,
              response_payload_json TEXT NOT NULL,
              response_preview TEXT NOT NULL,
              created_at TEXT NOT NULL,
              expires_at TEXT,
              cache_policy_class TEXT NOT NULL,
              agent_version TEXT NOT NULL,
              corpus_version TEXT,
              tool_version TEXT,
              thread_scope_key TEXT,
              exact_anchor_key TEXT,
              freshness_class TEXT NOT NULL,
              reuse_scope_key TEXT,
              structured_critical_signature TEXT,
              structured_confidence_at_insert REAL,
              confidence_metadata_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_semantic_namespace ON semantic_row(namespace);
            CREATE TABLE IF NOT EXISTS anchor_row (
              anchor_key_sha256 TEXT PRIMARY KEY,
              cache_id TEXT NOT NULL,
              namespace TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_anchor_namespace ON anchor_row(namespace);
            """
        )
        cur = c.execute("SELECT v FROM cache_meta WHERE k = 'schema_version'")
        row = cur.fetchone()
        if row is None:
            c.execute(
                "INSERT INTO cache_meta(k, v) VALUES ('schema_version', ?)",
                (SCHEMA_VERSION,),
            )
        c.commit()

    def upsert_exact(
        self,
        *,
        namespace: str,
        exact_key_sha256: str,
        scope_fingerprint: str | None,
        payload: dict[str, Any],
    ) -> None:
        c = self._connection()
        c.execute(
            """
            INSERT INTO exact_row(exact_key_sha256, namespace, scope_fingerprint, payload_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(exact_key_sha256) DO UPDATE SET
              namespace=excluded.namespace,
              scope_fingerprint=excluded.scope_fingerprint,
              payload_json=excluded.payload_json
            """,
            (exact_key_sha256, namespace, scope_fingerprint, json.dumps(payload)),
        )
        c.commit()

    def upsert_semantic(self, entry: SemanticCacheEntry) -> None:
        c = self._connection()
        c.execute(
            """
            INSERT INTO semantic_row(
              cache_id, namespace, query_original, query_normalized, embedding_json,
              response_payload_json, response_preview, created_at, expires_at,
              cache_policy_class, agent_version, corpus_version, tool_version,
              thread_scope_key, exact_anchor_key, freshness_class, reuse_scope_key,
              structured_critical_signature, structured_confidence_at_insert,
              confidence_metadata_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(cache_id) DO UPDATE SET
              namespace=excluded.namespace,
              query_original=excluded.query_original,
              query_normalized=excluded.query_normalized,
              embedding_json=excluded.embedding_json,
              response_payload_json=excluded.response_payload_json,
              response_preview=excluded.response_preview,
              expires_at=excluded.expires_at,
              cache_policy_class=excluded.cache_policy_class,
              agent_version=excluded.agent_version,
              corpus_version=excluded.corpus_version,
              tool_version=excluded.tool_version,
              thread_scope_key=excluded.thread_scope_key,
              exact_anchor_key=excluded.exact_anchor_key,
              freshness_class=excluded.freshness_class,
              reuse_scope_key=excluded.reuse_scope_key,
              structured_critical_signature=excluded.structured_critical_signature,
              structured_confidence_at_insert=excluded.structured_confidence_at_insert,
              confidence_metadata_json=excluded.confidence_metadata_json
            """,
            (
                entry.cache_id,
                entry.namespace,
                entry.query_text_original,
                entry.query_text_normalized,
                json.dumps(entry.embedding_vector),
                json.dumps(entry.response_payload),
                entry.response_preview,
                _iso(entry.created_at) or "",
                _iso(entry.expires_at),
                entry.cache_policy_class,
                entry.agent_version,
                entry.corpus_version,
                entry.tool_or_schema_version,
                entry.thread_scope_key,
                entry.exact_anchor_key,
                entry.freshness_class,
                entry.reuse_scope_key,
                entry.structured_critical_signature,
                entry.structured_confidence_at_insert,
                json.dumps(entry.confidence_metadata),
            ),
        )
        c.commit()

    def upsert_anchor(self, *, namespace: str, anchor_key_sha256: str, cache_id: str) -> None:
        c = self._connection()
        c.execute(
            """
            INSERT INTO anchor_row(anchor_key_sha256, cache_id, namespace)
            VALUES (?, ?, ?)
            ON CONFLICT(anchor_key_sha256) DO UPDATE SET
              cache_id=excluded.cache_id,
              namespace=excluded.namespace
            """,
            (anchor_key_sha256, cache_id, namespace),
        )
        c.commit()

    def load_namespace(self, namespace: str, *, embedding_dimension: int) -> NamespaceHydrationBundle:
        c = self._connection()
        exact_payloads: dict[str, dict[str, Any]] = {}
        for row in c.execute(
            "SELECT exact_key_sha256, payload_json FROM exact_row WHERE namespace = ?",
            (namespace,),
        ):
            exact_payloads[row[0]] = json.loads(row[1])

        semantic_entries: dict[str, SemanticCacheEntry] = {}
        for row in c.execute(
            """
            SELECT cache_id, namespace, query_original, query_normalized, embedding_json,
                   response_payload_json, response_preview, created_at, expires_at,
                   cache_policy_class, agent_version, corpus_version, tool_version,
                   thread_scope_key, exact_anchor_key, freshness_class, reuse_scope_key,
                   structured_critical_signature, structured_confidence_at_insert,
                   confidence_metadata_json
            FROM semantic_row WHERE namespace = ?
            ORDER BY cache_id
            """,
            (namespace,),
        ):
            semantic_entries[row[0]] = SemanticCacheEntry(
                cache_id=row[0],
                namespace=row[1],
                query_text_original=row[2],
                query_text_normalized=row[3],
                embedding_vector=json.loads(row[4]),
                response_payload=json.loads(row[5]),
                response_preview=row[6],
                created_at=_parse_dt(row[7]) or datetime.now(tz=UTC),
                expires_at=_parse_dt(row[8]),
                cache_policy_class=row[9],
                agent_version=row[10],
                corpus_version=row[11],
                tool_or_schema_version=row[12],
                thread_scope_key=row[13],
                exact_anchor_key=row[14],
                freshness_class=row[15],
                reuse_scope_key=row[16],
                structured_critical_signature=row[17],
                structured_confidence_at_insert=row[18],
                confidence_metadata=json.loads(row[19]) if row[19] else {},
            )

        anchor_map: dict[str, str] = {}
        for row in c.execute(
            "SELECT anchor_key_sha256, cache_id FROM anchor_row WHERE namespace = ?",
            (namespace,),
        ):
            anchor_map[row[0]] = row[1]

        return NamespaceHydrationBundle(
            namespace=namespace,
            exact_payloads=exact_payloads,
            semantic_entries=semantic_entries,
            anchor_map=anchor_map,
            embedding_dimension=embedding_dimension,
        )

    def clear_namespace(self, namespace: str) -> None:
        c = self._connection()
        c.execute("DELETE FROM exact_row WHERE namespace = ?", (namespace,))
        c.execute("DELETE FROM semantic_row WHERE namespace = ?", (namespace,))
        c.execute("DELETE FROM anchor_row WHERE namespace = ?", (namespace,))
        c.commit()
        FaissVectorPersistence(self._db_path).remove(namespace=namespace)

    def stats_global(self) -> dict[str, Any]:
        c = self._connection()
        n_exact = c.execute("SELECT COUNT(*) FROM exact_row").fetchone()[0]
        n_sem = c.execute("SELECT COUNT(*) FROM semantic_row").fetchone()[0]
        return {
            "db_path": str(self._db_path.resolve()),
            "exact_rows": int(n_exact),
            "semantic_rows": int(n_sem),
        }

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class FaissVectorPersistence(VectorIndexPersistence):
    """Namespace-scoped FAISS snapshot beside the SQLite file."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)

    def save(self, *, namespace: str, faiss_index: FaissHnswIndex, cache_ids: list[str]) -> None:
        fp, ip = faiss_paths(self._db_path, namespace)
        save_index(faiss_index._index, cache_ids, fp, ip)

    def load_or_none(self, *, namespace: str, embedding_dimension: int) -> tuple[Any, list[str]] | None:
        fp, ip = faiss_paths(self._db_path, namespace)
        loaded = load_index(fp, ip)
        if loaded is None:
            return None
        idx, ids = loaded
        if idx.d != embedding_dimension:
            remove_files(fp, ip)
            return None
        return idx, ids

    def remove(self, *, namespace: str) -> None:
        fp, ip = faiss_paths(self._db_path, namespace)
        remove_files(fp, ip)


def _ann_cache_ids(entries: dict[str, SemanticCacheEntry]) -> set[str]:
    return {cid for cid, e in entries.items() if e.cache_policy_class != "exact_only"}


def hydrate_semantic_store(
    *,
    bundle: NamespaceHydrationBundle,
    semantic_store: SemanticStore,
    vector_persistence: FaissVectorPersistence,
    force_rebuild_index: bool,
) -> bool:
    """Load semantic entries into ``semantic_store``. Return True if FAISS snapshot was used."""

    dim = bundle.embedding_dimension
    entries = bundle.semantic_entries
    ann_expected = _ann_cache_ids(entries)

    loaded = None if force_rebuild_index else vector_persistence.load_or_none(
        namespace=bundle.namespace, embedding_dimension=dim
    )
    if loaded is not None:
        idx, ids = loaded
        if getattr(idx, "d", None) != dim or idx.ntotal != len(ids):
            loaded = None
        elif set(ids) != ann_expected:
            loaded = None
        elif idx.ntotal != len(ann_expected):
            loaded = None

    if loaded is not None:
        idx, ids = loaded
        wrapped = FaissHnswIndex.from_deserialized(dimension=dim, index=idx, cache_ids=ids)
        semantic_store.replace_hydrated(
            entries=dict(entries),
            anchor_map=dict(bundle.anchor_map),
            index=wrapped,
        )
        return True

    semantic_store.clear()
    for _cid, entry in sorted(entries.items(), key=lambda kv: kv[0]):
        semantic_store.insert(entry)

    vector_persistence.save(
        namespace=bundle.namespace,
        faiss_index=semantic_store._index,
        cache_ids=semantic_store._index.cache_ids,
    )
    return False
