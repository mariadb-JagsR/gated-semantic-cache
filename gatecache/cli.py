"""Unified command-line interface for the semantic cache (routing, exact, semantic, tracing)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from gatecache.api import JudgePolicy, PutPolicy, SemanticCache
from gatecache.cache.exact_cache import ExactCache
from gatecache.cache.sqlite_persistence import SqliteCachePersistence
from gatecache.cache.semantic_store import SemanticStore
from gatecache.eval.offline_benchmark import benchmark_routing_classifier, inspect_queries
from gatecache.eval.shadow_compare import run_shadow_compare
from gatecache.eval.queries_regression import run_queries_regression_report
from gatecache.eval.structured_extract_eval import run_legacy_structured_coverage
from gatecache.eval.quora_pairs_eval import (
    default_quora_dataset_path,
    default_quora_report_path,
    run_quora_pairs_eval,
    write_quora_pair_report,
)
from gatecache.eval.queries_pairs_eval import (
    default_finance_pairs_path,
    default_finance_pairs_reports_dir,
    default_queries_pairs_reports_dir,
    default_queries_path,
    run_queries_pairs_eval,
    write_queries_pairs_report,
)
from gatecache.eval.structured_exact_benchmark import run_structured_exact_benchmark
from gatecache.models.context import DEFAULT_MAX_PRIOR_USER_QUERIES, DEFAULT_SEMANTIC_LOW_WATERMARK, RequestContext
from gatecache.routing.classifier import train_default_classifier
from gatecache.eval.datasets import build_routing_dataset
from gatecache.embeddings.backends import (
    caching_embedder,
    embedding_dim_for_openai_model,
    make_offline_fake_embedder,
    make_openai_embedder,
)
from gatecache.serving.neighbor_judge import noop_allow_neighbor_judge
from gatecache.serving.pipeline import SemanticCachePipeline


def _env_float(key: str) -> float | None:
    raw = os.environ.get(key)
    if raw is None or str(raw).strip() == "":
        return None
    return float(raw)


def _env_int(key: str) -> int | None:
    raw = os.environ.get(key)
    if raw is None or str(raw).strip() == "":
        return None
    return int(raw)


def _load_dotenv_files() -> None:
    """Load ``.env`` from the repo root (dev layout), then cwd (cwd wins)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    pkg_parent = Path(__file__).resolve().parents[1]
    if (pkg_parent / "pyproject.toml").is_file():
        load_dotenv(pkg_parent / ".env")
    load_dotenv(Path.cwd() / ".env", override=True)


def _neighbor_judge_from_env() -> Any:
    if os.environ.get("NEIGHBOR_JUDGE_ENABLED", "").strip().lower() not in ("1", "true", "yes"):
        return None
    from gatecache.serving.llm_judge import default_llm_neighbor_judge_from_env

    llm_judge = default_llm_neighbor_judge_from_env()
    if llm_judge is not None:
        return llm_judge
    # Fallback when judge is enabled but no LLM credentials/config are available.
    return noop_allow_neighbor_judge


_USE_ENV_NEIGHBOR_JUDGE = object()


def build_pipeline(
    *,
    openai_model: str = "text-embedding-3-small",
    openai_api_key: str | None = None,
    openai_dimensions: int | None = None,
    embed_cache: bool = False,
    neighbor_judge: Any | None = _USE_ENV_NEIGHBOR_JUDGE,
) -> SemanticCachePipeline:
    """Production pipeline: OpenAI ``text-embedding-3-*`` (requires ``OPENAI_API_KEY``).

    Pass ``neighbor_judge=`` an explicit callable, or ``None`` to disable. The default sentinel
    reads ``NEIGHBOR_JUDGE_ENABLED`` and attaches ``noop_allow_neighbor_judge`` when set.
    """
    router = train_default_classifier(build_routing_dataset())
    dim = embedding_dim_for_openai_model(openai_model, dimensions=openai_dimensions)
    embedder = make_openai_embedder(
        model=openai_model,
        api_key=openai_api_key,
        dimensions=openai_dimensions,
    )
    if embed_cache:
        embedder = caching_embedder(embedder)
    nj: Any | None
    if neighbor_judge is _USE_ENV_NEIGHBOR_JUDGE:
        nj = _neighbor_judge_from_env()
    else:
        nj = neighbor_judge
    return SemanticCachePipeline(
        router=router,
        exact_cache=ExactCache(),
        semantic_store=SemanticStore(dimension=dim),
        embedder=embedder,
        neighbor_judge=nj,
    )


def build_default_pipeline() -> SemanticCachePipeline:
    """Offline deterministic fake embeddings (same dimension as ``text-embedding-3-small``); for unit tests."""
    dim = embedding_dim_for_openai_model("text-embedding-3-small")
    return SemanticCachePipeline(
        router=train_default_classifier(build_routing_dataset()),
        exact_cache=ExactCache(),
        semantic_store=SemanticStore(dimension=dim),
        embedder=make_offline_fake_embedder(dimension=dim),
    )


def _pipeline_from_embedding_args(args: argparse.Namespace) -> SemanticCachePipeline:
    return build_pipeline(
        openai_model=args.openai_model,
        openai_api_key=getattr(args, "openai_api_key", None),
        openai_dimensions=getattr(args, "openai_dimensions", None),
        embed_cache=getattr(args, "embed_cache", False),
    )


def default_live_answer(query: str, _: RequestContext) -> dict[str, Any]:
    return {"answer": f"live:{query}", "success": True}


def run_single_query(
    query: str,
    *,
    context: RequestContext,
    pipeline: SemanticCachePipeline | None = None,
    live_answer: Any = None,
) -> dict[str, Any]:
    pl = pipeline or build_default_pipeline()
    live = live_answer or default_live_answer
    t0 = time.perf_counter()
    response = pl.answer_query(query, context, live)
    total_ms = round((time.perf_counter() - t0) * 1000, 3)
    payload = response.payload
    return {
        "query": query,
        "source": response.source,
        "total_latency_ms": total_ms,
        "trace": response.trace.to_dict(),
        "payload_preview": {
            "answer": str(payload.get("answer", ""))[:500],
            "success": payload.get("success", True),
        },
    }


def _cmd_query(args: argparse.Namespace) -> int:
    context = _repl_context(args)
    queries: list[str] = []
    if args.query:
        queries.append(args.query)
    if args.queries_file:
        text = Path(args.queries_file).read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                queries.append(line)
    if not queries:
        print("error: provide --query or --queries-file", file=sys.stderr)
        return 2

    out: list[dict[str, Any]] | dict[str, Any]
    if args.stateless:
        out = [
            run_single_query(q, context=context, pipeline=_pipeline_from_embedding_args(args)) for q in queries
        ]
    else:
        pl = _pipeline_from_embedding_args(args)
        if len(queries) == 1:
            out = run_single_query(queries[0], context=context, pipeline=pl)
        else:
            out = [run_single_query(q, context=context, pipeline=pl) for q in queries]

    text = json.dumps(out, indent=2 if args.pretty else None)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")
    print(text)
    return 0


def _repl_context(args: argparse.Namespace) -> RequestContext:
    ceiling = getattr(args, "neighbor_judge_ceiling", None)
    if ceiling is None:
        ceiling = _env_float("NEIGHBOR_JUDGE_SIMILARITY_CEILING")
    margin = getattr(args, "neighbor_judge_ambiguity_margin", None)
    if margin is None:
        margin = _env_float("NEIGHBOR_JUDGE_AMBIGUITY_MARGIN")
    max_calls = getattr(args, "neighbor_judge_max_calls", None)
    if max_calls is None:
        max_calls = _env_int("NEIGHBOR_JUDGE_MAX_CALLS")
    min_conf = getattr(args, "semantic_ok_min_route_confidence", None)
    if min_conf is None:
        min_conf = _env_float("SEMANTIC_OK_MIN_ROUTE_CONFIDENCE")
    low_watermark = getattr(args, "semantic_low_watermark", None)
    if low_watermark is None:
        low_watermark = _env_float("SEMANTIC_LOW_WATERMARK")
    return RequestContext(
        namespace=args.namespace,
        agent_version=args.agent_version,
        corpus_version=args.corpus_version,
        tool_or_schema_version=args.tool_version,
        thread_scope_key=args.thread_scope,
        semantic_threshold=args.semantic_threshold,
        semantic_low_watermark=low_watermark if low_watermark is not None else DEFAULT_SEMANTIC_LOW_WATERMARK,
        cache_namespace=getattr(args, "cache_namespace", None),
        reuse_scope_key=getattr(args, "reuse_scope", None),
        neighbor_judge_similarity_ceiling=ceiling,
        neighbor_judge_ambiguity_margin=margin,
        neighbor_judge_max_calls=max_calls,
        semantic_ok_min_route_confidence=min_conf,
    )


def _ensure_repl_thread_scope(args: argparse.Namespace, *, prefix: str = "repl") -> str:
    thread_scope = getattr(args, "thread_scope", None)
    if not thread_scope:
        thread_scope = f"{prefix}-{uuid.uuid4().hex}"
        setattr(args, "thread_scope", thread_scope)
    print(f"thread_scope_key={thread_scope}", file=sys.stderr)
    return thread_scope


def _cmd_repl(args: argparse.Namespace) -> int:
    """One process, one pipeline: repeat the same text to see exact_cache_hit / semantic_cache."""
    _ensure_repl_thread_scope(args)
    context = _repl_context(args)
    pl = _pipeline_from_embedding_args(args)
    live = default_live_answer
    print(
        "Interactive mode: same process keeps exact + semantic stores. "
        "Empty line, quit, or Ctrl-D to exit.",
        file=sys.stderr,
    )
    prior_user_queries: list[str] = []
    while True:
        try:
            line = input("query> ")
        except EOFError:
            print(file=sys.stderr)
            break
        query = line.strip()
        if not query or query.lower() in ("quit", "exit", "q"):
            break
        context = dataclasses.replace(
            context,
            prior_user_queries=tuple(prior_user_queries[-DEFAULT_MAX_PRIOR_USER_QUERIES:]),
        )
        out = run_single_query(query, context=context, pipeline=pl, live_answer=live)
        print(json.dumps(out, indent=2 if args.pretty else None))
        prior_user_queries.append(query)
    return 0


def _cmd_route(args: argparse.Namespace) -> int:
    rows = inspect_queries(args.queries, classifier_path=args.model)
    text = json.dumps(rows, indent=2 if args.pretty else None)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")
    print(text)
    return 0


def _cmd_eval_routing(args: argparse.Namespace) -> int:
    report = benchmark_routing_classifier(folds=args.folds)
    text = json.dumps(report.to_dict(), indent=2 if args.pretty else None)
    if args.report_json:
        Path(args.report_json).write_text(text, encoding="utf-8")
    print(text)
    return 0


def _cmd_eval_shadow(_: argparse.Namespace) -> int:
    report = run_shadow_compare()
    print(json.dumps(report, indent=2))
    return 0


def _cmd_eval_structured(_: argparse.Namespace) -> int:
    report = run_structured_exact_benchmark()
    print(json.dumps(report.to_dict(), indent=2))
    return 0


def _cmd_eval_coverage(_: argparse.Namespace) -> int:
    report = run_legacy_structured_coverage()
    print(json.dumps(report.to_dict(), indent=2))
    return 0


def _cmd_eval_quora_pairs(args: argparse.Namespace) -> int:
    judge_enabled = not args.no_judge and args.route_policy != "vector_only"
    if args.route_policy == "vector_only" and not args.no_judge:
        print("note: --route-policy vector_only ignores neighbor judge (GPTCache-style baseline)", file=sys.stderr)
    report = run_quora_pairs_eval(
        dataset_path=args.dataset,
        limit=args.limit,
        seed=args.seed,
        balanced=not args.unbalanced,
        semantic_threshold=args.semantic_threshold,
        semantic_low_watermark=args.semantic_low_watermark,
        openai_model=args.openai_model,
        route_policy=args.route_policy,
        neighbor_judge=None if args.no_judge else _USE_ENV_NEIGHBOR_JUDGE,
        progress_every=args.progress_every,
    )
    text = json.dumps(report.to_dict(), indent=2 if args.pretty else None)
    if not args.no_save_report:
        report_path = args.report_json
        if report_path is None:
            report_path = default_quora_report_path(
                limit=report.limit,
                seed=report.seed,
                judge_enabled=judge_enabled,
                route_policy=report.route_policy,
                semantic_threshold=report.semantic_threshold,
            )
        saved = write_quora_pair_report(
            report,
            report_path,
            judge_enabled=judge_enabled,
            balanced=not args.unbalanced,
        )
        print(f"Report saved to: {saved}", file=sys.stderr)
    elif args.report_json:
        Path(args.report_json).write_text(text, encoding="utf-8")
    print(text)
    return 0


def _cmd_eval_queries_regression(args: argparse.Namespace) -> int:
    report = run_queries_regression_report(
        queries_file=args.queries_file,
        classifier_path=args.classifier,
        mode=args.mode,
    )
    text = json.dumps(report.to_dict(), indent=2 if args.pretty else None)
    if args.report_json:
        Path(args.report_json).write_text(text, encoding="utf-8")
    print(text)
    return 0


def _cmd_eval_queries_pairs(args: argparse.Namespace) -> int:
    judge_enabled = not args.no_judge and args.route_policy != "vector_only"
    if args.route_policy == "vector_only" and not args.no_judge:
        print("note: --route-policy vector_only ignores neighbor judge (GPTCache-style baseline)", file=sys.stderr)
    report = run_queries_pairs_eval(
        queries_file=args.queries_file,
        pairs_json=args.pairs_json,
        semantic_threshold=args.semantic_threshold,
        semantic_low_watermark=args.semantic_low_watermark,
        openai_model=args.openai_model,
        route_policy=args.route_policy,
        neighbor_judge=None if args.no_judge else _USE_ENV_NEIGHBOR_JUDGE,
    )
    text = json.dumps(report.to_dict(), indent=2 if args.pretty else None)
    if not args.no_save_report:
        report_path = args.report_json
        if report_path is None:
            judge_part = "vector-only" if args.route_policy == "vector_only" else ("judge-on" if judge_enabled else "no-judge")
            thresh = f"thresh{report.semantic_threshold:.2f}".replace(".", "p")
            if args.pairs_json is not None:
                fixture_stem = Path(args.pairs_json).stem
                reports_dir = default_finance_pairs_reports_dir() if "finance" in fixture_stem else default_queries_pairs_reports_dir()
                report_path = reports_dir / f"{fixture_stem}_{judge_part}_{args.route_policy}_{thresh}.json"
            else:
                report_path = default_queries_pairs_reports_dir() / f"queries_pairs_{judge_part}_{args.route_policy}_{thresh}.json"
        saved = write_queries_pairs_report(report, report_path, judge_enabled=judge_enabled)
        print(f"Report saved to: {saved}", file=sys.stderr)
    elif args.report_json:
        Path(args.report_json).write_text(text, encoding="utf-8")
    print(text)
    return 0


def _default_persistent_db_path() -> Path:
    raw = os.environ.get("GATECACHE_DB")
    if raw is not None and str(raw).strip():
        return Path(raw).expanduser()
    return Path.cwd() / ".gatecache" / "cache.sqlite3"


def _resolve_cache_db(raw: Path | None) -> Path:
    if raw is not None:
        return Path(raw).expanduser()
    return _default_persistent_db_path()


def _parse_scope_kv(pairs: list[str] | None) -> dict[str, str]:
    if not pairs:
        return {}
    out: dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"invalid scope pair {item!r}; expected KEY=VALUE")
        k, v = item.split("=", 1)
        k, v = k.strip(), v.strip()
        if not k:
            raise ValueError(f"invalid scope pair {item!r}")
        out[k] = v
    return out


def _judge_policy_for_cache_cli(args: argparse.Namespace) -> JudgePolicy:
    ceiling = getattr(args, "neighbor_judge_ceiling", None)
    if ceiling is None:
        ceiling = _env_float("NEIGHBOR_JUDGE_SIMILARITY_CEILING")
    margin = getattr(args, "neighbor_judge_ambiguity_margin", None)
    if margin is None:
        margin = _env_float("NEIGHBOR_JUDGE_AMBIGUITY_MARGIN")
    max_calls = getattr(args, "neighbor_judge_max_calls", None)
    if max_calls is None:
        max_calls = _env_int("NEIGHBOR_JUDGE_MAX_CALLS")
    if getattr(args, "no_judge", False):
        return JudgePolicy(
            enabled=False,
            similarity_floor=float(args.semantic_threshold),
            similarity_ceiling=ceiling,
            ambiguity_margin=margin,
            max_calls=max_calls,
            fail_closed_on_missing_judge=False,
        )
    return JudgePolicy(
        enabled=True,
        similarity_floor=float(args.semantic_threshold),
        similarity_ceiling=ceiling,
        ambiguity_margin=margin,
        max_calls=max_calls,
        fail_closed_on_missing_judge=True,
    )


def _open_semantic_cache_from_cli(args: argparse.Namespace) -> SemanticCache:
    db = _resolve_cache_db(getattr(args, "db", None))
    return SemanticCache.from_sqlite(
        db_path=db,
        namespace=args.namespace,
        openai_model=args.openai_model,
        openai_api_key=getattr(args, "openai_api_key", None),
        openai_dimensions=getattr(args, "openai_dimensions", None),
        embed_cache=getattr(args, "embed_cache", False),
        neighbor_judge=_neighbor_judge_from_env(),
        default_judge_policy=_judge_policy_for_cache_cli(args),
        use_default_llm_judge=False,
        force_rebuild_index=getattr(args, "force_rebuild_index", False),
    )


def _cmd_cache_get(args: argparse.Namespace) -> int:
    cache = _open_semantic_cache_from_cli(args)
    try:
        scope = _parse_scope_kv(getattr(args, "scope", None))
        hit = cache.get(
            args.query,
            scope_keys=scope or None,
            semantic_mode=args.semantic_mode,
            judge_policy=_judge_policy_for_cache_cli(args),
            semantic_threshold=args.semantic_threshold,
            semantic_low_watermark=getattr(args, "semantic_low_watermark", None),
        )
        if hit is None:
            out: dict[str, Any] = {"hit": False}
        else:
            out = {
                "hit": True,
                "source": hit.source,
                "payload": hit.payload,
                "similarity": hit.similarity,
                "trace": hit.trace,
            }
        print(json.dumps(out, indent=2 if args.pretty else None))
        return 0
    finally:
        cache.close()


def _cmd_cache_put(args: argparse.Namespace) -> int:
    metadata: dict[str, Any] = {}
    if getattr(args, "metadata_json", None):
        metadata = json.loads(args.metadata_json)
        if not isinstance(metadata, dict):
            raise ValueError("metadata-json must decode to a JSON object")
    ttl = None if getattr(args, "no_ttl", False) else args.ttl_seconds
    policy = PutPolicy(semantic_mode=args.put_semantic_mode, ttl_seconds=ttl, metadata=metadata)
    cache = _open_semantic_cache_from_cli(args)
    try:
        payload = json.loads(args.response_json)
        if not isinstance(payload, dict):
            raise ValueError("response-json must decode to a JSON object")
        scope = _parse_scope_kv(getattr(args, "scope", None))
        ref = cache.put(args.query, payload, policy=policy, scope_keys=scope or None)
        out = {
            "namespace": ref.namespace,
            "exact_key_sha256": ref.exact_key_sha256,
            "semantic_cache_id": ref.semantic_cache_id,
            "anchor_key_sha256": ref.anchor_key_sha256,
            "semantic_indexed": ref.semantic_indexed,
            "routing_label": ref.routing_label,
            "routing_confidence": ref.routing_confidence,
            "trace": ref.trace,
        }
        print(json.dumps(out, indent=2 if args.pretty else None))
        return 0
    finally:
        cache.close()


def _cmd_cache_repl(args: argparse.Namespace) -> int:
    cache = _open_semantic_cache_from_cli(args)
    thread_scope = _ensure_repl_thread_scope(args, prefix="cache-repl")
    print(
        "Persistent cache REPL: lookups hit SQLite + restored FAISS. Empty line or quit exits.",
        file=sys.stderr,
    )
    try:
        scope = _parse_scope_kv(getattr(args, "scope", None))
        policy = _judge_policy_for_cache_cli(args)
        while True:
            try:
                line = input("query> ")
            except EOFError:
                print(file=sys.stderr)
                break
            query = line.strip()
            if not query or query.lower() in ("quit", "exit", "q"):
                break
            hit = cache.get(
                query,
                scope_keys=scope or None,
                semantic_mode=args.semantic_mode,
                judge_policy=policy,
                semantic_threshold=args.semantic_threshold,
                semantic_low_watermark=getattr(args, "semantic_low_watermark", None),
                thread_scope_key=thread_scope,
            )
            if hit is None:
                out = {"hit": False}
            else:
                out = {
                    "hit": True,
                    "source": hit.source,
                    "payload": hit.payload,
                    "similarity": hit.similarity,
                    "trace": hit.trace,
                }
            print(json.dumps(out, indent=2 if args.pretty else None))
        return 0
    finally:
        cache.close()


def _cmd_cache_stats(args: argparse.Namespace) -> int:
    db = _resolve_cache_db(getattr(args, "db", None))
    persistence = SqliteCachePersistence(db)
    persistence.init_schema()
    try:
        stats = persistence.stats_global()
        print(json.dumps(stats, indent=2 if args.pretty else None))
        return 0
    finally:
        persistence.close()


def _cmd_cache_clear(args: argparse.Namespace) -> int:
    if not args.yes:
        print("error: refuse to clear without --yes", file=sys.stderr)
        return 2
    db = _resolve_cache_db(getattr(args, "db", None))
    persistence = SqliteCachePersistence(db)
    persistence.init_schema()
    try:
        persistence.clear_namespace(args.namespace)
        print(
            json.dumps({"cleared_namespace": args.namespace, "db": str(db.resolve())}, indent=2 if args.pretty else None)
        )
        return 0
    finally:
        persistence.close()


def _add_cache_shared_core(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite path (default: $GATECACHE_DB or .gatecache/cache.sqlite3)",
    )
    p.add_argument("--namespace", default="default", help="Logical partition / SQLite namespace")
    p.add_argument(
        "--openai-model",
        default=os.environ.get("OPENAI_MODEL", "text-embedding-3-small"),
        help="Embedding model for hydration + new inserts",
    )
    p.add_argument("--openai-api-key", default=None, help="Override OPENAI_API_KEY for this run")
    p.add_argument("--openai-dimensions", type=int, default=None, help="Embedding width for text-embedding-3-*")
    p.add_argument("--embed-cache", action="store_true", help="LRU cache embedding calls in-process")
    p.add_argument(
        "--force-rebuild-index",
        action="store_true",
        help="Ignore on-disk FAISS snapshot and rebuild from SQLite embeddings",
    )
    p.add_argument(
        "--semantic-threshold",
        type=float,
        default=float(os.environ.get("SEMANTIC_THRESHOLD", "0.86")),
        help="Cosine similarity floor for semantic hits",
    )
    p.add_argument(
        "--semantic-low-watermark",
        type=float,
        default=_env_float("SEMANTIC_LOW_WATERMARK"),
        help="Cosine similarity floor for gray-zone judge candidates (default: env SEMANTIC_LOW_WATERMARK or 0.70)",
    )
    p.add_argument("--no-judge", action="store_true", help="Disable neighbor-judge gray-zone policy")
    p.add_argument("--neighbor-judge-ceiling", type=float, default=None)
    p.add_argument("--neighbor-judge-ambiguity-margin", type=float, default=None)
    p.add_argument("--neighbor-judge-max-calls", type=int, default=None)
    p.add_argument("--pretty", action="store_true", default=True)
    p.add_argument("--no-pretty", dest="pretty", action="store_false")


def _add_cache_lookup_extras(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--semantic-mode",
        choices=("auto", "always", "never"),
        default="always",
        help="Semantic retrieval mode for get (default: always)",
    )
    p.add_argument(
        "--scope",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Optional scope pair for isolation (repeatable)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gatecache",
        description="Semantic cache: full pipeline queries, routing eval, and structured-extract benchmarks.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    p_query = sub.add_parser("query", help="Run one or more queries through the full pipeline (trace + latency).")
    p_query.add_argument("--query", "-q", help="Single user message")
    p_query.add_argument("--queries-file", "-f", help="File with one query per line (# comments allowed)")
    p_query.add_argument("--namespace", default="default")
    p_query.add_argument(
        "--cache-namespace",
        default=None,
        help="Override cache partition for ANN/exact keys (defaults to --namespace)",
    )
    p_query.add_argument(
        "--reuse-scope",
        default=None,
        help="Optional reuse scope key (agent/surface); must match entry to reuse",
    )
    p_query.add_argument("--thread-scope", default=None, help="Thread scope key for thread-scoped reuse")
    p_query.add_argument(
        "--semantic-threshold",
        type=float,
        default=float(os.environ.get("SEMANTIC_THRESHOLD", "0.86")),
        help="Cosine similarity floor for semantic cache (default: env SEMANTIC_THRESHOLD or 0.86)",
    )
    p_query.add_argument(
        "--semantic-low-watermark",
        type=float,
        default=_env_float("SEMANTIC_LOW_WATERMARK"),
        help="Cosine similarity floor for gray-zone judge candidates (default: env SEMANTIC_LOW_WATERMARK or 0.70)",
    )
    p_query.add_argument(
        "--neighbor-judge-ceiling",
        type=float,
        default=None,
        help="Gray zone high bound: at or above this cosine similarity the neighbor judge is skipped "
        "(env NEIGHBOR_JUDGE_SIMILARITY_CEILING)",
    )
    p_query.add_argument(
        "--neighbor-judge-ambiguity-margin",
        type=float,
        default=None,
        help="Skip judge when top1 - runner_up >= this margin (env NEIGHBOR_JUDGE_AMBIGUITY_MARGIN)",
    )
    p_query.add_argument(
        "--neighbor-judge-max-calls",
        type=int,
        default=None,
        help="Max neighbor judge invocations per request (env NEIGHBOR_JUDGE_MAX_CALLS)",
    )
    p_query.add_argument(
        "--semantic-ok-min-route-confidence",
        type=float,
        default=None,
        help="Downgrade SEMANTIC_OK to SKIP_CACHE when router confidence is below this floor "
        "(env SEMANTIC_OK_MIN_ROUTE_CONFIDENCE)",
    )
    p_query.add_argument("--agent-version", default="v1")
    p_query.add_argument("--corpus-version", default=None)
    p_query.add_argument("--tool-version", default=None)
    p_query.add_argument(
        "--stateless",
        action="store_true",
        help="Each query uses a new pipeline (empty caches). Default shares one pipeline across queries.",
    )
    p_query.add_argument(
        "--openai-model",
        default=os.environ.get("OPENAI_MODEL", "text-embedding-3-small"),
        help="Embedding model (default: env OPENAI_MODEL or text-embedding-3-small).",
    )
    p_query.add_argument(
        "--openai-api-key",
        default=None,
        help="Override OPENAI_API_KEY for this run.",
    )
    p_query.add_argument(
        "--openai-dimensions",
        type=int,
        default=None,
        help="Optional reduced embedding width for text-embedding-3-* (FAISS index must match).",
    )
    p_query.add_argument(
        "--embed-cache",
        action="store_true",
        help="LRU cache repeated embedding calls (same normalized string) in-process.",
    )
    p_query.add_argument("--output-json", "-o", help="Write JSON result to file")
    p_query.add_argument("--pretty", action="store_true", default=True)
    p_query.add_argument("--no-pretty", dest="pretty", action="store_false")
    p_query.set_defaults(func=_cmd_query)

    p_repl = sub.add_parser(
        "repl",
        help="Interactive loop: one pipeline per process so you can test cache hits (vs one-shot query).",
    )
    p_repl.add_argument("--namespace", default="default")
    p_repl.add_argument("--cache-namespace", default=None)
    p_repl.add_argument("--reuse-scope", default=None)
    p_repl.add_argument("--thread-scope", default=None, help="Thread scope key for thread-scoped reuse")
    p_repl.add_argument(
        "--semantic-threshold",
        type=float,
        default=float(os.environ.get("SEMANTIC_THRESHOLD", "0.86")),
    )
    p_repl.add_argument("--semantic-low-watermark", type=float, default=_env_float("SEMANTIC_LOW_WATERMARK"))
    p_repl.add_argument("--neighbor-judge-ceiling", type=float, default=None)
    p_repl.add_argument("--neighbor-judge-ambiguity-margin", type=float, default=None)
    p_repl.add_argument("--neighbor-judge-max-calls", type=int, default=None)
    p_repl.add_argument("--semantic-ok-min-route-confidence", type=float, default=None)
    p_repl.add_argument("--agent-version", default="v1")
    p_repl.add_argument("--corpus-version", default=None)
    p_repl.add_argument("--tool-version", default=None)
    p_repl.add_argument(
        "--openai-model",
        default=os.environ.get("OPENAI_MODEL", "text-embedding-3-small"),
    )
    p_repl.add_argument("--openai-api-key", default=None)
    p_repl.add_argument("--openai-dimensions", type=int, default=None)
    p_repl.add_argument("--embed-cache", action="store_true")
    p_repl.add_argument("--pretty", action="store_true", default=True)
    p_repl.add_argument("--no-pretty", dest="pretty", action="store_false")
    p_repl.set_defaults(func=_cmd_repl)

    p_route = sub.add_parser("route", help="Classifier-only: label + probabilities (no cache).")
    p_route.add_argument("queries", nargs="+", help="One or more queries")
    p_route.add_argument("--model", "-m", help="Optional saved classifier .pkl path")
    p_route.add_argument("--output-json", "-o")
    p_route.add_argument("--pretty", action="store_true", default=True)
    p_route.add_argument("--no-pretty", dest="pretty", action="store_false")
    p_route.set_defaults(func=_cmd_route)

    p_eval = sub.add_parser("eval", help="Offline benchmarks and corpus coverage")
    eval_sub = p_eval.add_subparsers(dest="eval_command", required=True)

    p_er = eval_sub.add_parser("routing", help="Cross-validated routing classifier metrics + latency percentiles")
    p_er.add_argument("--folds", type=int, default=4)
    p_er.add_argument("--report-json", help="Write report JSON to path")
    p_er.add_argument("--pretty", action="store_true", default=True)
    p_er.add_argument("--no-pretty", dest="pretty", action="store_false")
    p_er.set_defaults(func=_cmd_eval_routing)

    eval_sub.add_parser("shadow", help="Shadow comparison harness (expected vs actual source)").set_defaults(
        func=_cmd_eval_shadow
    )
    eval_sub.add_parser("structured", help="Structured exact pair benchmark").set_defaults(func=_cmd_eval_structured)
    eval_sub.add_parser("coverage", help="Legacy corpus structured-extraction coverage stats").set_defaults(
        func=_cmd_eval_coverage
    )

    p_qp = eval_sub.add_parser(
        "quora-pairs",
        help="Quora Question Pairs cache precision/recall benchmark (labeled duplicate pairs)",
    )
    p_qp.add_argument(
        "--dataset",
        type=Path,
        default=default_quora_dataset_path(),
        help="Path to quora_duplicate_questions.tsv",
    )
    p_qp.add_argument("--limit", type=int, default=200, help="Max pairs to evaluate (sampled with --seed)")
    p_qp.add_argument("--seed", type=int, default=42, help="Random seed for pair sampling")
    p_qp.add_argument(
        "--unbalanced",
        action="store_true",
        help="Sample from the full dataset instead of a balanced duplicate/non-duplicate mix",
    )
    p_qp.add_argument(
        "--route-policy",
        choices=("semantic_ok", "honest", "vector_only"),
        default="semantic_ok",
        help=(
            "honest uses the trained router; semantic_ok forces SEMANTIC_OK to isolate retrieval+gates; "
            "vector_only is GPTCache-style (embed + ANN + threshold only, no classifier/judge/gates)"
        ),
    )
    p_qp.add_argument("--semantic-threshold", type=float, default=None)
    p_qp.add_argument("--semantic-low-watermark", type=float, default=None)
    p_qp.add_argument("--openai-model", default=os.environ.get("OPENAI_MODEL", "text-embedding-3-small"))
    p_qp.add_argument("--no-judge", action="store_true", help="Disable gray-zone neighbor judge")
    p_qp.add_argument("--progress-every", type=int, default=25, help="Print progress every N pairs (0 disables)")
    p_qp.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Write report JSON to path (default: docs/quora_pairs_eval/ with timestamped name)",
    )
    p_qp.add_argument(
        "--no-save-report",
        action="store_true",
        help="Skip writing a persistent report under docs/quora_pairs_eval/",
    )
    p_qp.add_argument("--pretty", action="store_true")
    p_qp.set_defaults(func=_cmd_eval_quora_pairs)

    p_qr = eval_sub.add_parser(
        "queries-regression",
        help="JSON report for tests/queries.txt (routing by default; pipeline mode with OpenAI embeddings)",
    )
    p_qr.add_argument(
        "--queries-file",
        type=Path,
        default=None,
        help="Defaults to next/tests/queries.txt relative to the package dev tree",
    )
    p_qr.add_argument("--classifier", type=Path, default=None, help="Optional saved router .pkl")
    p_qr.add_argument(
        "--mode",
        choices=("routing", "pipeline"),
        default="routing",
        help="routing: classifier only; pipeline: full handler with OpenAI embeddings (needs OPENAI_API_KEY)",
    )
    p_qr.add_argument("--report-json", help="Write report JSON to path")
    p_qr.add_argument("--pretty", action="store_true", default=True)
    p_qr.add_argument("--no-pretty", dest="pretty", action="store_false")
    p_qr.set_defaults(func=_cmd_eval_queries_regression)

    p_qpairs = eval_sub.add_parser(
        "queries-pairs",
        help="Curated tests/queries.txt pair benchmark: full stack vs GPTCache-style vector-only",
    )
    p_qpairs.add_argument(
        "--queries-file",
        type=Path,
        default=default_queries_path(),
        help="Path to queries.txt (even lines = consecutive seed/probe pairs)",
    )
    p_qpairs.add_argument(
        "--pairs-json",
        type=Path,
        default=None,
        help="Adversarial pair fixture JSON (e.g. tests/fixtures/finance_adversarial_pairs.json)",
    )
    p_qpairs.add_argument(
        "--route-policy",
        choices=("honest", "semantic_ok", "vector_only"),
        default="honest",
        help="honest=trained router; semantic_ok=force SEMANTIC_OK; vector_only=GPTCache-style",
    )
    p_qpairs.add_argument("--semantic-threshold", type=float, default=None)
    p_qpairs.add_argument("--semantic-low-watermark", type=float, default=None)
    p_qpairs.add_argument("--openai-model", default=os.environ.get("OPENAI_MODEL", "text-embedding-3-small"))
    p_qpairs.add_argument("--no-judge", action="store_true", help="Disable gray-zone neighbor judge")
    p_qpairs.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Write report JSON (default: docs/queries_pairs_eval/)",
    )
    p_qpairs.add_argument(
        "--no-save-report",
        action="store_true",
        help="Skip writing a persistent report under docs/queries_pairs_eval/",
    )
    p_qpairs.add_argument("--pretty", action="store_true")
    p_qpairs.set_defaults(func=_cmd_eval_queries_pairs)

    p_cache = sub.add_parser(
        "cache",
        help="SQLite-backed durable cache: get/put survive restarts (FAISS snapshot beside DB).",
    )
    cache_sub = p_cache.add_subparsers(dest="cache_command", required=True)

    p_cget = cache_sub.add_parser("get", help="Lookup (exact + semantic) against hydrated stores")
    _add_cache_shared_core(p_cget)
    _add_cache_lookup_extras(p_cget)
    p_cget.add_argument("--query", "-q", required=True)
    p_cget.set_defaults(func=_cmd_cache_get)

    p_cput = cache_sub.add_parser("put", help="Write-through to SQLite + refresh FAISS snapshot")
    _add_cache_shared_core(p_cput)
    p_cput.add_argument(
        "--semantic-mode",
        choices=("auto", "always", "never"),
        default="always",
        dest="put_semantic_mode",
        help="How ANN indexing follows routing for this insert (default: always)",
    )
    p_cput.add_argument(
        "--scope",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Optional scope pair for isolation (repeatable)",
    )
    p_cput.add_argument("--query", "-q", required=True)
    p_cput.add_argument(
        "--response-json",
        required=True,
        help='JSON object for cached payload (e.g. \'{"answer":"..."}\')',
    )
    p_cput.add_argument("--ttl-seconds", type=int, default=3600, dest="ttl_seconds")
    p_cput.add_argument("--no-ttl", action="store_true", help="Store semantic row without expires_at")
    p_cput.add_argument("--metadata-json", default=None, help="Optional JSON object merged into entry metadata")
    p_cput.set_defaults(func=_cmd_cache_put)

    p_crepl = cache_sub.add_parser("repl", help="Interactive lookups against the persistent cache")
    _add_cache_shared_core(p_crepl)
    _add_cache_lookup_extras(p_crepl)
    p_crepl.add_argument("--thread-scope", default=None, help="Thread scope key for thread-scoped reuse")
    p_crepl.set_defaults(func=_cmd_cache_repl)

    p_cst = cache_sub.add_parser("stats", help="Row counts for the SQLite cache file (no OpenAI calls)")
    p_cst.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite path (default: $GATECACHE_DB or .gatecache/cache.sqlite3)",
    )
    p_cst.add_argument("--pretty", action="store_true", default=True)
    p_cst.add_argument("--no-pretty", dest="pretty", action="store_false")
    p_cst.set_defaults(func=_cmd_cache_stats)

    p_clr = cache_sub.add_parser("clear", help="Delete all rows + FAISS snapshot for one namespace")
    p_clr.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite path (default: $GATECACHE_DB or .gatecache/cache.sqlite3)",
    )
    p_clr.add_argument("--namespace", default="default")
    p_clr.add_argument("--yes", action="store_true", help="Required confirmation flag")
    p_clr.add_argument("--pretty", action="store_true", default=True)
    p_clr.add_argument("--no-pretty", dest="pretty", action="store_false")
    p_clr.set_defaults(func=_cmd_cache_clear)

    return parser


def main(argv: list[str] | None = None) -> int:
    _load_dotenv_files()
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 2
    need_openai = False
    if getattr(args, "command", None) in ("query", "repl"):
        need_openai = True
    elif getattr(args, "command", None) == "cache" and getattr(args, "cache_command", None) in (
        "get",
        "put",
        "repl",
    ):
        need_openai = True
    if need_openai:
        key = getattr(args, "openai_api_key", None) or os.environ.get("OPENAI_API_KEY")
        if not key:
            print(
                "error: query, repl, and cache get/put/repl require OPENAI_API_KEY (or pass --openai-api-key once)",
                file=sys.stderr,
            )
            return 2
    try:
        return int(func(args))
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
