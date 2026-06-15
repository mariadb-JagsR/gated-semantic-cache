from datetime import UTC, datetime

from gatecache.models.cache_entry import SemanticCacheEntry
from gatecache.models.context import RequestContext
from gatecache.serving.llm_judge import (
    DEFAULT_JUDGE_ANSWER_PREVIEW_CHARS,
    DEFAULT_JUDGE_CHAT_MAX_TOKENS,
    DEFAULT_JUDGE_MAX_OUTPUT_TOKENS,
    _JUDGE_SYSTEM_PROMPT,
    _judge_prompt,
    _parse_decision,
)


def test_llm_judge_system_prompt_requires_strict_intent_match() -> None:
    prompt = _JUDGE_SYSTEM_PROMPT.lower()

    for phrase in [
        "different surface wording alone is not a reason to reject",
        "list or recommendation requests",
        "provider",
        "intent",
        "answer type",
        "entity",
    ]:
        assert phrase in prompt


def test_llm_judge_system_prompt_allows_same_subject_paraphrases() -> None:
    prompt = _JUDGE_SYSTEM_PROMPT.lower()

    assert "best programming books" in prompt
    assert "some top books for programming" in prompt
    assert "informational paraphrases" in prompt


def test_llm_judge_user_prompt_is_compact_and_case_focused() -> None:
    entry = _entry("Compare A vs B", {"answer": "a/b comparison"})
    context = RequestContext()
    prompt = _judge_prompt(query="Is A better than B?", entry=entry, context=context).lower()
    system = _JUDGE_SYSTEM_PROMPT.lower()

    for phrase in [
        "new: is a better than b?",
        "cached: compare a vs b",
        "answer preview:",
        "same-subject paraphrases",
        "provider/product/version",
        'json only: {"reuse": true|false, "reason": "short_code"}',
    ]:
        assert phrase in prompt

    assert "different surface wording alone" not in prompt
    assert system.count("same intent") <= 1
    assert len(prompt) < 600


def test_llm_judge_books_paraphrase_prompt_mentions_allowance() -> None:
    entry = _entry(
        "best programming books",
        {"answer": "live:best programming books"},
    )
    prompt = _judge_prompt(
        query="some top books for programming",
        entry=entry,
        context=RequestContext(),
    ).lower()

    assert "some top books for programming" in prompt
    assert "best programming books" in prompt
    assert "same-subject paraphrases" in prompt


def test_llm_judge_gcp_vs_aws_prompt_surfaces_target_difference() -> None:
    entry = _entry(
        "how do I deploy node.js app to aws?",
        {"answer": "live:how do I deploy node.js app to aws?"},
    )
    prompt = _judge_prompt(
        query="how do I deploy node.js app to gcp?",
        entry=entry,
        context=RequestContext(),
    ).lower()

    assert "new: how do i deploy node.js app to gcp?" in prompt
    assert "cached: how do i deploy node.js app to aws?" in prompt
    assert "provider/product/version" in prompt


def test_llm_judge_answer_preview_is_capped() -> None:
    long_answer = "x" * 500
    entry = _entry("short query", {"answer": long_answer})
    prompt = _judge_prompt(query="short query", entry=entry, context=RequestContext())

    assert long_answer not in prompt
    assert "x" * DEFAULT_JUDGE_ANSWER_PREVIEW_CHARS in prompt


def test_llm_judge_paraphrase_prompt_includes_both_queries() -> None:
    entry = _entry(
        "easy to deploy node js app to gcp ?",
        {"answer": "GCP deploy steps here"},
    )
    prompt = _judge_prompt(
        query="is it simple to deploy a node.js app to gcp?",
        entry=entry,
        context=RequestContext(),
    ).lower()

    assert "easy to deploy node js app to gcp" in prompt
    assert "is it simple to deploy a node.js app to gcp?" in prompt


def test_use_responses_api_for_gpt5_models() -> None:
    from gatecache.serving.llm_judge import _use_responses_api

    assert _use_responses_api("gpt-5-mini") is True
    assert _use_responses_api("gpt-4o-mini") is False


def test_llm_judge_user_prompt_includes_prior_thread_queries() -> None:
    entry = _entry("react vs hue", {"answer": "live:react vs hue"})
    context = RequestContext(prior_user_queries=("react vs hue", "is react better than hue ?"))
    prompt = _judge_prompt(query="how about react vs hue vs angular", entry=entry, context=context)

    assert "Earlier user queries in this thread" in prompt
    assert "react vs hue" in prompt
    assert "is react better than hue ?" in prompt


def test_llm_judge_parse_decision_handles_fenced_json() -> None:
    decision = _parse_decision('```json\n{"reuse": false, "reason": "intent_or_scope_changed"}\n```')

    assert decision == {"reuse": False, "reason": "intent_or_scope_changed"}


def test_llm_judge_parse_decision_marks_empty_content_invalid() -> None:
    decision = _parse_decision("")

    assert decision == {"reuse": False, "reason": "neighbor_judge_invalid_response"}


def test_llm_judge_parse_decision_accepts_compact_json() -> None:
    decision = _parse_decision('{"reuse": true, "reason": "same_intent"}')

    assert decision == {"reuse": True, "reason": "same_intent"}


def test_llm_judge_default_reasoning_effort_is_low() -> None:
    from gatecache.serving.llm_judge import DEFAULT_JUDGE_REASONING_EFFORT

    assert DEFAULT_JUDGE_REASONING_EFFORT == "low"


def test_llm_judge_default_max_output_tokens_is_128() -> None:
    assert DEFAULT_JUDGE_MAX_OUTPUT_TOKENS == 128


def test_llm_judge_default_chat_max_tokens_is_80() -> None:
    assert DEFAULT_JUDGE_CHAT_MAX_TOKENS == 80


def _entry(query: str, payload: dict[str, object]) -> SemanticCacheEntry:
    return SemanticCacheEntry(
        cache_id="id",
        namespace="default",
        query_text_original=query,
        query_text_normalized=query.lower(),
        embedding_vector=[1.0, 0.0],
        response_payload=payload,
        response_preview=str(payload.get("answer", ""))[:140],
        created_at=datetime.now(tz=UTC),
        expires_at=None,
        cache_policy_class="semantic",
        agent_version="v1",
        corpus_version=None,
        tool_or_schema_version=None,
        thread_scope_key=None,
        exact_anchor_key=None,
        freshness_class="stable",
    )
