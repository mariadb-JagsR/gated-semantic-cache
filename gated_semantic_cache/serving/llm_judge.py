from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any

from gated_semantic_cache.models.cache_entry import SemanticCacheEntry
from gated_semantic_cache.models.context import DEFAULT_MAX_PRIOR_USER_QUERIES, RequestContext
from gated_semantic_cache.serving.neighbor_judge import NeighborJudge, set_neighbor_judge_observation


DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
DEFAULT_JUDGE_TIMEOUT_SECONDS = 5.0
DEFAULT_JUDGE_REASONING_EFFORT = "low"
# GPT-5 Responses API: 128 is enough for compact JSON with reasoning_effort=low; lower values often truncate.
DEFAULT_JUDGE_MAX_OUTPUT_TOKENS = 128
DEFAULT_JUDGE_CHAT_MAX_TOKENS = 80
DEFAULT_JUDGE_ANSWER_PREVIEW_CHARS = 200
_JUDGE_SYSTEM_PROMPT = (
    "You are a strict semantic cache verifier. Return only compact JSON with keys reuse (bool) and reason (string). "
    "Allow reuse when the new and cached queries request the same answer for the same subject, scope, and answer type. "
    "This includes informational paraphrases, list or recommendation requests, and how-to requests with equivalent intent "
    "(e.g. 'best programming books' vs 'some top books for programming', or 'how do I deploy X' vs 'steps to deploy X'). "
    "Reject reuse when intent or answer type changes, or when any answer-critical entity, target, destination, provider, "
    "product, version, identifier, number, unit, timeframe, polarity, or scope differs. "
    "Different surface wording alone is not a reason to reject if the cached answer would still directly answer the new query. "
    "When uncertain about a safety-critical mismatch, reject reuse."
)


@dataclass(frozen=True, slots=True)
class JudgeModelResponse:
    content: str
    status: str | None = None


def default_llm_neighbor_judge_from_env() -> NeighborJudge | None:
    """Build the default LLM judge when credentials/config are available.

    Set ``SEMANTIC_CACHE_DEFAULT_JUDGE=0`` to disable auto-wiring. If no OpenAI API
    key is configured, return ``None`` and let the API's fail-closed policy handle it.
    """

    enabled = os.environ.get("SEMANTIC_CACHE_DEFAULT_JUDGE", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return make_openai_neighbor_judge(
        api_key=api_key,
        model=os.environ.get("SEMANTIC_CACHE_JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        timeout_seconds=float(
            os.environ.get("SEMANTIC_CACHE_JUDGE_TIMEOUT_SECONDS", str(DEFAULT_JUDGE_TIMEOUT_SECONDS))
        ),
    )


def make_openai_neighbor_judge(
    *,
    api_key: str | None = None,
    model: str = DEFAULT_JUDGE_MODEL,
    timeout_seconds: float = DEFAULT_JUDGE_TIMEOUT_SECONDS,
) -> NeighborJudge:
    """Create a bounded yes/no verifier for post-retrieval semantic cache hits."""

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OpenAI judge requires OPENAI_API_KEY or api_key=.")

    from openai import OpenAI

    client = OpenAI(api_key=key, timeout=timeout_seconds)
    reasoning_effort = os.environ.get("SEMANTIC_CACHE_JUDGE_REASONING_EFFORT", DEFAULT_JUDGE_REASONING_EFFORT)

    def judge(query: str, entry: SemanticCacheEntry, context: RequestContext) -> str | None:
        prompt = _judge_prompt(query=query, entry=entry, context=context)
        try:
            model_response = _invoke_judge_model(
                client,
                model=model,
                system_prompt=_JUDGE_SYSTEM_PROMPT,
                user_prompt=prompt,
                reasoning_effort=reasoning_effort,
            )
        except Exception:
            _record_judge_observation(content="", decision={"reuse": False, "reason": "neighbor_judge_error"}, status="error")
            return "neighbor_judge_error"

        decision = _parse_decision(model_response.content)
        _record_judge_observation(
            content=model_response.content,
            decision=decision,
            status=model_response.status,
        )
        if decision.get("reuse") is True:
            return None
        reason = str(decision.get("reason") or "neighbor_judge_rejected").strip()
        return _machine_reason(reason)

    return judge


def _invoke_judge_model(
    client: Any,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    reasoning_effort: str,
) -> JudgeModelResponse:
    """Call the judge model. GPT-5 family uses Responses API + structured JSON output."""

    max_output_tokens = int(
        os.environ.get("SEMANTIC_CACHE_JUDGE_MAX_OUTPUT_TOKENS", str(DEFAULT_JUDGE_MAX_OUTPUT_TOKENS))
    )

    if _use_responses_api(model):
        response = client.responses.create(
            model=model,
            reasoning={"effort": reasoning_effort},
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text={"format": {"type": "json_object"}},
            max_output_tokens=max_output_tokens,
        )
        status = getattr(response, "status", None)
        content = str(getattr(response, "output_text", "") or "")
        if status == "incomplete" and not content.strip():
            return JudgeModelResponse(content="", status=str(status))
        return JudgeModelResponse(content=content, status=str(status) if status is not None else None)

    chat_max_tokens = int(
        os.environ.get("SEMANTIC_CACHE_JUDGE_CHAT_MAX_TOKENS", str(DEFAULT_JUDGE_CHAT_MAX_TOKENS))
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=chat_max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content or ""
    return JudgeModelResponse(content=content, status="completed")


def _use_responses_api(model: str) -> bool:
    override = os.environ.get("SEMANTIC_CACHE_JUDGE_USE_RESPONSES_API", "").strip().lower()
    if override in {"0", "false", "no", "off"}:
        return False
    if override in {"1", "true", "yes", "on"}:
        return True
    return model.startswith("gpt-5")


def _judge_answer_preview_chars() -> int:
    return int(
        os.environ.get(
            "SEMANTIC_CACHE_JUDGE_ANSWER_PREVIEW_CHARS",
            str(DEFAULT_JUDGE_ANSWER_PREVIEW_CHARS),
        )
    )


def _judge_prompt(*, query: str, entry: SemanticCacheEntry, context: RequestContext) -> str:
    payload = entry.response_payload
    answer = payload.get("answer", payload)
    preview = str(answer)[: _judge_answer_preview_chars()]
    parts = [_prior_queries_block(context)]
    parts.extend(
        [
            f"New: {query}\n",
            f"Cached: {entry.query_text_original}\n",
            f"Answer preview: {preview}\n",
            "Allow reuse for same-subject paraphrases (lists/recommendations, how-to variants, FAQ rewordings) "
            "when the preview still fits.\n",
            "Reject reuse if intent, answer type, entities, numbers, provider/product/version, "
            "polarity, scope, or timeframe differ, or if the preview would not answer the new query.\n",
            'JSON only: {"reuse": true|false, "reason": "short_code"}',
        ]
    )
    return "".join(parts)


def _prior_queries_block(context: RequestContext) -> str:
    prior = context.prior_user_queries[-DEFAULT_MAX_PRIOR_USER_QUERIES:]
    if not prior:
        return ""
    lines = "\n".join(f"- {item}" for item in prior)
    return f"Earlier user queries in this thread (oldest first):\n{lines}\n\n"


def _parse_decision(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"reuse": False, "reason": "neighbor_judge_invalid_response"}
    return parsed if isinstance(parsed, dict) else {"reuse": False, "reason": "neighbor_judge_invalid_response"}


def _machine_reason(reason: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in reason)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:80] or "neighbor_judge_rejected"


def _judge_debug_enabled() -> bool:
    return os.environ.get("SEMANTIC_CACHE_JUDGE_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _record_judge_observation(*, content: str, decision: dict[str, Any], status: str | None) -> None:
    observation = {"raw": content, "decision": decision, "status": status}
    invalid = decision.get("reason") == "neighbor_judge_invalid_response"
    if _judge_debug_enabled() or invalid or status == "incomplete":
        set_neighbor_judge_observation(observation)
    if _judge_debug_enabled() or invalid or status == "incomplete":
        print(
            f"[neighbor_judge] status={status!r} raw={content!r} parsed={json.dumps(decision, sort_keys=True)}",
            file=sys.stderr,
            flush=True,
        )
