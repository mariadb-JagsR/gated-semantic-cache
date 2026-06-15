"""Persistent SQLite + FAISS snapshot round-trips (no OpenAI)."""

from __future__ import annotations

from gated_semantic_cache.api import JudgePolicy, PutPolicy, SemanticCache
from gated_semantic_cache.cache.faiss_snapshot import faiss_paths
from gated_semantic_cache.cache.sqlite_persistence import SqliteCachePersistence
from gated_semantic_cache.embeddings.backends import embedding_dim_for_openai_model, make_offline_fake_embedder


def test_sqlite_put_get_survives_new_instance(tmp_path) -> None:
    db = tmp_path / "c.sqlite3"
    ns = "tenant-a"
    dim = embedding_dim_for_openai_model("text-embedding-3-small")
    embedder = make_offline_fake_embedder(dimension=dim)

    c1 = SemanticCache.from_sqlite(
        db_path=db,
        namespace=ns,
        embedder=embedder,
        default_judge_policy=JudgePolicy(enabled=False),
        use_default_llm_judge=False,
    )
    try:
        c1.put(
            "What is semantic caching?",
            {"answer": "Reuse prior responses.", "success": True},
            policy=PutPolicy(semantic_mode="always"),
        )
    finally:
        c1.close()

    c2 = SemanticCache.from_sqlite(
        db_path=db,
        namespace=ns,
        embedder=embedder,
        default_judge_policy=JudgePolicy(enabled=False),
        use_default_llm_judge=False,
    )
    try:
        hit = c2.get(
            "What is semantic caching?",
            judge_policy=JudgePolicy(enabled=False),
            semantic_threshold=0.5,
        )
        assert hit is not None
        assert hit.payload["answer"] == "Reuse prior responses."
    finally:
        c2.close()


def test_namespace_isolation(tmp_path) -> None:
    db = tmp_path / "c.sqlite3"
    dim = embedding_dim_for_openai_model("text-embedding-3-small")
    embedder = make_offline_fake_embedder(dimension=dim)

    a = SemanticCache.from_sqlite(
        db_path=db,
        namespace="n1",
        embedder=embedder,
        default_judge_policy=JudgePolicy(enabled=False),
        use_default_llm_judge=False,
    )
    try:
        a.put("q1", {"answer": "n1"}, policy=PutPolicy())
    finally:
        a.close()

    b = SemanticCache.from_sqlite(
        db_path=db,
        namespace="n2",
        embedder=embedder,
        default_judge_policy=JudgePolicy(enabled=False),
        use_default_llm_judge=False,
    )
    try:
        assert b.get("q1", judge_policy=JudgePolicy(enabled=False), semantic_threshold=0.1) is None
    finally:
        b.close()


def test_rebuild_when_faiss_snapshot_removed(tmp_path) -> None:
    db = tmp_path / "c.sqlite3"
    ns = "one"
    dim = embedding_dim_for_openai_model("text-embedding-3-small")
    embedder = make_offline_fake_embedder(dimension=dim)

    c1 = SemanticCache.from_sqlite(
        db_path=db,
        namespace=ns,
        embedder=embedder,
        default_judge_policy=JudgePolicy(enabled=False),
        use_default_llm_judge=False,
    )
    try:
        c1.put("hello world", {"answer": "x"}, policy=PutPolicy())
    finally:
        c1.close()

    fp, ip = faiss_paths(db, ns)
    fp.unlink(missing_ok=True)
    ip.unlink(missing_ok=True)

    c2 = SemanticCache.from_sqlite(
        db_path=db,
        namespace=ns,
        embedder=embedder,
        default_judge_policy=JudgePolicy(enabled=False),
        use_default_llm_judge=False,
    )
    try:
        hit = c2.get("hello world", judge_policy=JudgePolicy(enabled=False), semantic_threshold=0.5)
        assert hit is not None
        assert hit.payload["answer"] == "x"
    finally:
        c2.close()


def test_stats_global(tmp_path) -> None:
    db = tmp_path / "c.sqlite3"
    p = SqliteCachePersistence(db)
    p.init_schema()
    try:
        assert p.stats_global()["exact_rows"] == 0
    finally:
        p.close()

