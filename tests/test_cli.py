import argparse

from gatecache.cli import _ensure_repl_thread_scope, build_default_pipeline, build_parser, run_single_query
from gatecache.models.context import RequestContext


def test_cli_query_returns_trace_and_latency() -> None:
    ctx = RequestContext(semantic_threshold=0.55)
    row = run_single_query("Lookup order #Z999 status", context=ctx)
    assert row["source"] == "live"
    assert "total_latency_ms" in row
    tr = row["trace"]
    assert tr["routing_label"] == "EXACT_ONLY"
    assert tr["structured_extraction_attempted"] is True
    assert "exact_cache_key_sha256" in tr


def test_parser_query_subcommand() -> None:
    args = build_parser().parse_args(["query", "-q", "hello"])
    assert args.command == "query"
    assert args.query == "hello"


def test_parser_eval_routing() -> None:
    args = build_parser().parse_args(["eval", "routing", "--folds", "2"])
    assert args.command == "eval"
    assert args.eval_command == "routing"
    assert args.folds == 2


def test_parser_eval_queries_regression() -> None:
    args = build_parser().parse_args(["eval", "queries-regression", "--mode", "routing"])
    assert args.command == "eval"
    assert args.eval_command == "queries-regression"
    assert args.mode == "routing"


def test_parser_cache_get() -> None:
    args = build_parser().parse_args(["cache", "get", "-q", "hello", "--no-judge"])
    assert args.command == "cache"
    assert args.cache_command == "get"
    assert args.query == "hello"


def test_parser_cache_stats_no_openai_branch() -> None:
    args = build_parser().parse_args(["cache", "stats"])
    assert args.command == "cache"
    assert args.cache_command == "stats"


def test_repl_default_thread_scope_is_minted() -> None:
    args = argparse.Namespace(thread_scope=None)
    thread_scope = _ensure_repl_thread_scope(args)

    assert thread_scope.startswith("repl-")
    assert args.thread_scope == thread_scope


def test_repl_preserves_explicit_thread_scope() -> None:
    args = argparse.Namespace(thread_scope="thread-1")
    thread_scope = _ensure_repl_thread_scope(args)

    assert thread_scope == "thread-1"
    assert args.thread_scope == "thread-1"


def test_parser_cache_repl_accepts_thread_scope() -> None:
    args = build_parser().parse_args(["cache", "repl", "--thread-scope", "thread-1", "--no-judge"])

    assert args.command == "cache"
    assert args.cache_command == "repl"
    assert args.thread_scope == "thread-1"


def test_shared_pipeline_exact_hit_second_query() -> None:
    """Same pipeline instance: second identical query hits exact cache."""
    pl = build_default_pipeline()
    ctx = RequestContext()
    r1 = run_single_query("What is semantic caching?", context=ctx, pipeline=pl)
    r2 = run_single_query("What is semantic caching?", context=ctx, pipeline=pl)
    assert r1["trace"]["exact_cache_hit"] is False
    assert r2["trace"]["exact_cache_hit"] is True
    assert r2["source"] == "exact_cache"
