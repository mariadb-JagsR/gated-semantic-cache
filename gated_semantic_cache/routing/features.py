from __future__ import annotations

import re
from collections.abc import Sequence

from sklearn.base import BaseEstimator, TransformerMixin

from gated_semantic_cache.routing.tech_tokens import filter_plausible_hostnames, is_dotted_tech_token


FIRST_PERSON_RE = re.compile(r"\b(i|i'm|i’ve|i'd|me|my|mine|we|we're|we've|us|our|ours)\b", re.IGNORECASE)
AMBIGUOUS_REFERENCE_RE = re.compile(r"\b(this|that|it|same|again|instead|one|those|these)\b", re.IGNORECASE)
MUTATION_VERB_RE = re.compile(r"\b(delete|update|change|switch|cancel|remove|revoke|restart|reset|modify)\b", re.IGNORECASE)
FRESHNESS_RE = re.compile(r"\b(latest|today|current|now|recent|live|real-?time)\b", re.IGNORECASE)
TTL_WINDOW_RE = re.compile(r"\b(this week|this month|yesterday|last week|last month|past week|past month)\b", re.IGNORECASE)
GENERIC_QUESTION_RE = re.compile(r"^(what is|what are|does|how do|how does|can|is)\b", re.IGNORECASE)
UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
HOSTNAME_RE = re.compile(r"\b[a-z0-9][a-z0-9.-]+\.[a-z]{2,}\b", re.IGNORECASE)
LONG_NUMERIC_RE = re.compile(r"\b\d{5,}\b")
CODE_LIKE_RE = re.compile(
    r"\b(?:v?\d+(?:\.\d+){1,4}|[a-z]{2}-[a-z]+-\d|[a-z0-9]+(?:[._-][a-z0-9]+)+|[a-z]+\d+[a-z0-9-]*|\d+[a-z]+[a-z0-9-]*)\b",
    re.IGNORECASE,
)
ORDER_TOKEN_RE = re.compile(r"\b(order|ticket|incident|customer|account|case|uuid|host)\b", re.IGNORECASE)
ACTION_REQUEST_RE = re.compile(
    r"\b(delete|update|change|switch|cancel|remove|restart|reset|rotate|reroute|recreate|revoke|fix|apply)\b",
    re.IGNORECASE,
)
URGENCY_RE = re.compile(r"\b(right now|immediately|urgent|asap|current|latest|today|now|active)\b", re.IGNORECASE)
POLICY_OR_EXPLANATION_RE = re.compile(
    r"\b(policy|policies|steps|documentation|docs|explain|difference|configure|how do i|get the invoice|download|fees?|pricing|cost|limits?|dimensions?|requirements?|guidelines?|how much)\b",
    re.IGNORECASE,
)
ADVICE_OR_EXPLANATION_RE = re.compile(
    r"\b(?:what should i do|should i|could this|would this|is this often|is it common|is this common|"
    r"often due to|due to|caused by|because of|after|what are signs|what are symptoms|when should)\b",
    re.IGNORECASE,
)
POSSESSIVE_SCOPE_RE = re.compile(r"\b(my|our|me|us)\b", re.IGNORECASE)
EXECUTION_PHRASE_RE = re.compile(r"\b(for me|for my|right now|on my behalf)\b", re.IGNORECASE)
# Short dispute / correction / challenge turns — route to SKIP_CACHE (never cache or reuse).
CHALLENGE_DISPUTE_RE = re.compile(
    r"(?is)"
    r"(?:"
    r"^\s*(?:recheck|retry|redo)\b"
    r"|^\s*(?:that'?s|this is|it'?s|you'?re)\s+(?:wrong|incorrect|not right|off|bad)\b"
    r"|^\s*(?:are you sure|really\??|seriously\??|come on)\b"
    r"|^\s*(?:what\??|huh\??|wait)\s*$"
    r"|^\s*(?:what\??|no|nah)\s*,?\s*(?:that'?s|this is|it'?s)\s+(?:wrong|incorrect|not right)\b"
    r"|^\s*(?:duh|ugh|nope)\b"
    r"|\b(?:makes?|doesn'?t)\s+no\s+sense\b"
    r"|\b(?:try|do)\s+again\b"
    r"|\b(?:wrong answer|not what i asked|missed the point|you missed|that can'?t be right)\b"
    r"|^\s*incorrect\b"
    r")",
)
# Backward-compatible alias used in engineered feature name.
LEGACY_BYPASS_LEADING_RE = CHALLENGE_DISPUTE_RE
# Regulatory / health-record phrasing (route toward cache bypass; domain-agnostic keyword cues).
PHI_REGULATORY_RE = re.compile(
    r"\b(hipaa|phi|protected health|patient data sharing|medical record|ehr|emr|clinical trial eligibility)\b",
    re.IGNORECASE,
)
DISCHARGE_OR_CHART_RE = re.compile(
    r"\b(discharge summary|chart notes|patient chart|my dexcom|my cgm|my glucose)\b",
    re.IGNORECASE,
)
FRESHNESS_STRONG_RE = re.compile(
    r"\b(right now|at this moment|currently waiting|live wait|real-?time status|current cpu|current balance|active incidents|latest status)\b",
    re.IGNORECASE,
)


def normalize_query_text(query: str) -> str:
    return " ".join(query.strip().lower().split())


def token_count(query: str) -> int:
    return len(normalize_query_text(query).split())


def identifier_like_tokens(query: str) -> list[str]:
    lowered = normalize_query_text(query)
    tokens: list[str] = []
    for match in UUID_RE.findall(lowered):
        tokens.append(match)
    for match in EMAIL_RE.findall(lowered):
        tokens.append(match)
    for match in filter_plausible_hostnames(HOSTNAME_RE.findall(lowered)):
        tokens.append(match)
    for match in LONG_NUMERIC_RE.findall(lowered):
        tokens.append(match)
    for match in CODE_LIKE_RE.findall(lowered):
        if not is_dotted_tech_token(match):
            tokens.append(match)
    if ORDER_TOKEN_RE.search(lowered):
        tokens.append("__order_token__")
    return tokens


def identifier_like_token_count(query: str) -> int:
    return len(set(identifier_like_tokens(query)))


def engineered_features(query: str) -> dict[str, float]:
    normalized = normalize_query_text(query)
    identifier_count = identifier_like_token_count(normalized)
    has_first_person = bool(FIRST_PERSON_RE.search(normalized))
    has_mutation_verb = bool(MUTATION_VERB_RE.search(normalized))
    has_action_request = bool(ACTION_REQUEST_RE.search(normalized))
    has_freshness_marker = bool(FRESHNESS_RE.search(normalized))
    has_ttl_window_marker = bool(TTL_WINDOW_RE.search(normalized))
    has_urgency_marker = bool(URGENCY_RE.search(normalized))
    has_policy_or_explanation = bool(POLICY_OR_EXPLANATION_RE.search(normalized))
    has_advice_or_explanation = bool(ADVICE_OR_EXPLANATION_RE.search(normalized))
    has_possessive_scope = bool(POSSESSIVE_SCOPE_RE.search(normalized))
    return {
        "token_count": float(token_count(normalized)),
        "has_first_person": float(has_first_person),
        "has_ambiguous_reference": float(bool(AMBIGUOUS_REFERENCE_RE.search(normalized))),
        "has_mutation_verb": float(has_mutation_verb),
        "has_action_request": float(has_action_request),
        "has_freshness_marker": float(has_freshness_marker),
        "has_ttl_window_marker": float(has_ttl_window_marker),
        "has_urgency_marker": float(has_urgency_marker),
        "has_identifier_like_token": float(identifier_count > 0),
        "identifier_like_token_count": float(identifier_count),
        "short_concrete_standalone": float(
            token_count(normalized) <= 6
            and identifier_count > 0
            and not AMBIGUOUS_REFERENCE_RE.search(normalized)
        ),
        "has_generic_question_marker": float(bool(GENERIC_QUESTION_RE.search(normalized))),
        "has_policy_or_explanation_marker": float(has_policy_or_explanation),
        "has_advice_or_explanation_marker": float(has_advice_or_explanation),
        "has_possessive_scope": float(has_possessive_scope),
        "has_execution_phrase": float(bool(EXECUTION_PHRASE_RE.search(normalized))),
        "action_with_personal_scope": float(has_action_request and has_possessive_scope),
        "freshness_with_personal_scope": float(has_freshness_marker and has_possessive_scope),
        "freshness_with_identifier": float(has_freshness_marker and identifier_count > 0),
        "ttl_window_with_policy_or_explanation": float(has_ttl_window_marker and has_policy_or_explanation),
        "ttl_window_with_advice_or_explanation": float(has_ttl_window_marker and has_advice_or_explanation),
        "personal_scope_with_advice_or_explanation": float(has_first_person and has_advice_or_explanation),
        "policy_with_identifier": float(has_policy_or_explanation and identifier_count > 0),
        "ends_with_question_mark": float(normalized.endswith("?")),
        "starts_with_short_followup": float(normalized.startswith(("same", "what about", "instead", "again", "also"))),
        "starts_with_action_verb": float(normalized.startswith(("delete", "update", "change", "switch", "cancel", "remove", "restart", "reset", "rotate", "fix"))),
        "has_legacy_bypass_leading": float(bool(LEGACY_BYPASS_LEADING_RE.search(query))),
        "has_phi_regulatory": float(bool(PHI_REGULATORY_RE.search(normalized))),
        "has_discharge_or_personal_device": float(bool(DISCHARGE_OR_CHART_RE.search(normalized))),
        "has_freshness_strong": float(bool(FRESHNESS_STRONG_RE.search(normalized))),
    }


class EngineeredFeatureTransformer(BaseEstimator, TransformerMixin):
    """Small sklearn-compatible transformer for cheap routing features."""

    def fit(self, X: Sequence[str], y: Sequence[str] | None = None) -> "EngineeredFeatureTransformer":
        return self

    def transform(self, X: Sequence[str]) -> list[dict[str, float]]:
        return [engineered_features(query) for query in X]
