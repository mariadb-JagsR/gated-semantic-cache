from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from gated_semantic_cache.models.cache_entry import SemanticCacheEntry
from gated_semantic_cache.routing.features import normalize_query_text

_CODE_LIKE_RE = re.compile(
    r"\b(?:"
    r"v?\d+(?:\.\d+){1,4}"
    r"|[a-z]{2}-[a-z]+-\d"
    r"|[a-z0-9]+(?:[._-][a-z0-9]+)+"
    r"|[a-z]+\d+[a-z0-9-]*"
    r"|\d+[a-z]+[a-z0-9-]*"
    r")\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"(?<![\w.])\$?\d+(?:,\d{3})*(?:\.\d+)?%?(?![\w.])")
_QUOTED_RE = re.compile(r"(['\"])(.+?)\1")
_CAPITALIZED_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9]*|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z0-9]*|[A-Z]{2,}|\d+|I{1,3}|IV|V))*\b"
)
_UNIT_RE = re.compile(
    r"\b(?:"
    r"mm|ml|cm|m|km|inch|inches|ft|feet|lb|lbs|kg|g|mg|gb|tb|mb|kb|usd|eur|gbp|jpy|dollars?|euros?|pounds?|"
    r"ms|sec|secs|seconds?|minutes?|hours?|days?|weeks?|months?|years?|percent|%"
    r")\b",
    re.IGNORECASE,
)
_FRESHNESS_RE = re.compile(
    r"\b(?:today|yesterday|tomorrow|latest|current|currently|now|recent|real-?time|"
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|this month|last month|next month|"
    r"this week|last week|next week|this year|last year|next year|(?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(r"\b(?:no|not|without|exclude|excluding|never|must not|doesn'?t|isn'?t|do not|don't)\b", re.IGNORECASE)
_WITH_TERM_RE = re.compile(
    r"\b(?:with|including|include|includes)\s+(?:\ba\b|\ban\b|\bthe\b\s+)?([a-z0-9][a-z0-9._-]*)",
    re.IGNORECASE,
)
_VS_SPLIT_RE = re.compile(r"\s+(?:vs\.?|versus)\s+", re.IGNORECASE)
_VERDICT_THAN_RE = re.compile(
    r"\b(.+?)\s+(?:better|worse|prefer(?:red)?)\s+than\s+(.+?)(?:\?|$)",
    re.IGNORECASE,
)
_WITHOUT_TERM_RE = re.compile(
    r"\b(?:without|excluding|exclude|no)\s+(?:a|an|the\s+)?([a-z0-9][a-z0-9._-]*)",
    re.IGNORECASE,
)
_COMPARATOR_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("upper_bound", ("under", "below", "less than", "at most", "maximum", "max", "<=", "<")),
    ("lower_bound", ("over", "above", "more than", "at least", "minimum", "min", ">=", ">")),
    ("range", ("between", "from")),
    ("equality", ("equal to", "equals", "exactly")),
)
_TERM_STOPWORDS = {
    "a",
    "an",
    "and",
    "but",
    "for",
    "in",
    "of",
    "or",
    "the",
    "to",
}
_ENTITY_STOPWORDS = {
    "Can",
    "Compare",
    "Convert",
    "Current",
    "Best",
    "CEO",
    "Does",
    "Explain",
    "Give",
    "How",
    "I",
    "Is",
    "List",
    "Recipe",
    "Recommend",
    "Show",
    "Steps",
    "Summarize",
    "Tell",
    "Top",
    "What",
    "When",
    "Where",
    "Which",
    "Who",
    "Advantages",
    "Applications",
    "Specifications",
    "USD",
    "EUR",
    "GBP",
    "JPY",
}
_SYNONYMS = {
    "nyc": "new york",
    "world war ii": "world war 2",
    "wwii": "world war 2",
    "world war i": "world war 1",
    "wwi": "world war 1",
    "dollars": "usd",
    "dollar": "usd",
    "euros": "eur",
    "euro": "eur",
    "pounds": "gbp",
    "pound": "gbp",
}
_CONTRAST_SYNONYMS = {
    "array": "list",
    "dict": "dictionary",
    "better": "verdict",
    "comparison": "compare",
    "differences": "compare",
    "difference": "compare",
    "versus": "compare",
    "vs": "compare",
    "worse": "verdict",
    "prefer": "verdict",
    "preferred": "verdict",
    "did": "past",
}
# Syntax-level contrast groups only. Each group captures mutually exclusive phrasing within
# one query shape (e.g. string vs list, deploy vs undeploy). Domain nouns (AWS, illnesses,
# restaurants, etc.) belong in protected_tokens / comparison_terms, not here.
_CONTRAST_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("code_object", ("string", "list", "array", "dict", "dictionary")),
    ("code_operation", ("reverse", "sort", "search", "parse", "format")),
    ("answer_type", ("definition", "example", "use_case", "symptoms", "treatment", "cause", "overview", "recipe")),
    ("compare_mode", ("compare", "comparison", "differences", "difference", "better", "worse", "prefer", "preferred", "versus", "vs", "use with")),
    ("deploy_intent", ("deploy", "undeploy", "did", "why")),
)


def extract_query_facets(query: str) -> dict[str, list[str]]:
    """Extract generic lexical safety facets from a query.

    Facets are syntax-level signals only. They are used after ANN retrieval to block or
    escalate obvious mismatches, never to authorize reuse on their own.
    """

    normalized = normalize_query_text(query)
    facets = {
        "protected_tokens": _sorted_unique(_CODE_LIKE_RE.findall(normalized)),
        "named_entities": _named_entities(query),
        "route_pairs": _route_pairs(normalized),
        "product_models": _product_models(normalized),
        "numbers": _sorted_unique(_normalize_number(m.group(0)) for m in _NUMBER_RE.finditer(normalized)),
        "units": _sorted_unique(_normalize_unit(m.group(0)) for m in _UNIT_RE.finditer(normalized)),
        "currency_pairs": _currency_pairs(normalized),
        "comparators": _sorted_unique(_comparator_groups(normalized)),
        "contrast_facets": _contrast_facets(normalized),
        "comparison_terms": _comparison_terms(normalized),
        "negations": _sorted_unique(m.group(0).lower() for m in _NEGATION_RE.finditer(normalized)),
        "freshness_terms": _sorted_unique(m.group(0).lower() for m in _FRESHNESS_RE.finditer(normalized)),
        "quoted_strings": _sorted_unique(match.group(2).strip().lower() for match in _QUOTED_RE.finditer(query)),
        "with_terms": _terms(_WITH_TERM_RE, normalized),
        "without_terms": _terms(_WITHOUT_TERM_RE, normalized),
    }
    return {key: value for key, value in facets.items() if value}


def facet_conflict_reason(query: str, entry: SemanticCacheEntry) -> str | None:
    current = extract_query_facets(query)
    cached = _cached_facets(entry)

    if _disjoint_nonempty(current, cached, "protected_tokens"):
        return "query_facet_protected_token_conflict"
    if _named_entity_conflict(current, cached):
        return "query_facet_named_entity_conflict"
    if _disjoint_nonempty(current, cached, "product_models"):
        return "query_facet_protected_token_conflict"
    if _disjoint_nonempty(current, cached, "route_pairs"):
        return "query_facet_direction_conflict"
    if _compare_mode_conflict(current, cached):
        return "query_facet_contrast_conflict"
    if _comparison_term_conflict(current, cached):
        return "query_facet_comparison_term_conflict"
    if _contrast_conflict(current, cached):
        return "query_facet_contrast_conflict"
    if _disjoint_nonempty(current, cached, "quoted_strings"):
        return "query_facet_quoted_string_conflict"
    if _date_or_freshness_conflict(current, cached):
        return "query_facet_freshness_conflict"
    if _quantitative_conflict(current, cached):
        return "query_facet_quantity_conflict"
    if _currency_pair_conflict(current, cached):
        return "query_facet_quantity_conflict"
    if _negation_polarity_conflict(current, cached):
        return "query_facet_negation_conflict"
    if _with_without_conflict(current, cached):
        return "query_facet_negation_conflict"
    return None


def _cached_facets(entry: SemanticCacheEntry) -> dict[str, list[str]]:
    raw = (entry.confidence_metadata or {}).get("query_facets")
    if isinstance(raw, Mapping):
        out: dict[str, list[str]] = {}
        for key, value in raw.items():
            if isinstance(key, str) and isinstance(value, list):
                out[key] = sorted({str(item).lower() for item in value if str(item).strip()})
        if out:
            return out
    return extract_query_facets(entry.query_text_original or entry.query_text_normalized)


def _quantitative_conflict(current: dict[str, list[str]], cached: dict[str, list[str]]) -> bool:
    current_numbers = set(current.get("numbers", []))
    cached_numbers = set(cached.get("numbers", []))
    if current_numbers and cached_numbers and current_numbers != cached_numbers:
        if current.get("units") or cached.get("units") or current.get("comparators") or cached.get("comparators"):
            return True

    current_comparators = set(current.get("comparators", []))
    cached_comparators = set(cached.get("comparators", []))
    if current_comparators and cached_comparators and current_comparators != cached_comparators:
        if current_numbers & cached_numbers or not current_numbers or not cached_numbers:
            return True

    current_units = set(current.get("units", []))
    cached_units = set(cached.get("units", []))
    return bool(current_units and cached_units and current_units != cached_units and current_numbers & cached_numbers)


def _currency_pair_conflict(current: dict[str, list[str]], cached: dict[str, list[str]]) -> bool:
    current_pairs = set(current.get("currency_pairs", []))
    cached_pairs = set(cached.get("currency_pairs", []))
    return bool(current_pairs and cached_pairs and current_pairs != cached_pairs)


def _with_without_conflict(current: dict[str, list[str]], cached: dict[str, list[str]]) -> bool:
    current_with = set(current.get("with_terms", []))
    current_without = set(current.get("without_terms", []))
    cached_with = set(cached.get("with_terms", []))
    cached_without = set(cached.get("without_terms", []))
    return bool((current_with & cached_without) or (current_without & cached_with))


def _negation_polarity_conflict(current: dict[str, list[str]], cached: dict[str, list[str]]) -> bool:
    return bool(current.get("negations")) != bool(cached.get("negations"))


def _named_entity_conflict(current: dict[str, list[str]], cached: dict[str, list[str]]) -> bool:
    current_routes = set(current.get("route_pairs", []))
    cached_routes = set(cached.get("route_pairs", []))
    if current_routes and cached_routes and current_routes == cached_routes:
        return False

    current_entities = set(current.get("named_entities", []))
    cached_entities = set(cached.get("named_entities", []))
    if not current_entities or not cached_entities:
        return False
    if current_entities == cached_entities:
        return False
    return not (_entities_mutually_align(current_entities, cached_entities))


def _entities_mutually_align(left: set[str], right: set[str]) -> bool:
    return _entities_align(left, right) and _entities_align(right, left)


def _entities_align(source: set[str], target: set[str]) -> bool:
    return all(any(_entity_compatible(entity, other) for other in target) for entity in source)


def _entity_compatible(left: str, right: str) -> bool:
    if left == right:
        return True
    return left in right or right in left


def _compare_mode_conflict(current: dict[str, list[str]], cached: dict[str, list[str]]) -> bool:
    current_modes = _compare_modes(current)
    cached_modes = _compare_modes(cached)
    if not current_modes or not cached_modes:
        return False
    if current_modes == cached_modes:
        return False
    return "verdict" in current_modes or "verdict" in cached_modes


def _comparison_term_conflict(current: dict[str, list[str]], cached: dict[str, list[str]]) -> bool:
    current_terms = set(current.get("comparison_terms", []))
    cached_terms = set(cached.get("comparison_terms", []))
    if not current_terms or not cached_terms:
        return False
    return current_terms != cached_terms


def _contrast_conflict(current: dict[str, list[str]], cached: dict[str, list[str]]) -> bool:
    current_by_group = _group_values(current.get("contrast_facets", []))
    cached_by_group = _group_values(cached.get("contrast_facets", []))
    for group, current_values in current_by_group.items():
        if group == "compare_mode":
            continue
        cached_values = cached_by_group.get(group)
        if group == "answer_type" and cached_values and current_values.intersection(cached_values):
            continue
        if cached_values and current_values != cached_values and not current_values.intersection(cached_values):
            return True
    return False


def _compare_modes(facets: dict[str, list[str]]) -> set[str]:
    modes = set(_group_values(facets.get("contrast_facets", [])).get("compare_mode", set()))
    if facets.get("comparison_terms") and "verdict" not in modes:
        modes.add("compare")
    return modes


def _comparison_terms(normalized: str) -> list[str]:
    terms: set[str] = set()
    if re.search(r"\b(?:vs\.?|versus)\b", normalized):
        for segment in _VS_SPLIT_RE.split(normalized):
            terms.update(_comparison_tokens(segment))
    match = _VERDICT_THAN_RE.search(normalized)
    if match:
        terms.update(_comparison_tokens(match.group(1)))
        terms.update(_comparison_tokens(match.group(2)))
    return _sorted_unique(terms)


def _comparison_tokens(segment: str) -> set[str]:
    stopwords = _TERM_STOPWORDS | {
        "about",
        "are",
        "can",
        "compare",
        "comparison",
        "could",
        "do",
        "does",
        "how",
        "include",
        "is",
        "make",
        "please",
        "should",
        "sure",
        "the",
        "what",
        "which",
        "will",
        "would",
    }
    tokens: set[str] = set()
    for raw in re.findall(r"[a-z0-9][a-z0-9._-]*", segment.lower()):
        token = raw.strip(".,;:!?")
        if token and len(token) > 1 and token not in stopwords:
            tokens.add(token)
    return tokens


def _date_or_freshness_conflict(current: dict[str, list[str]], cached: dict[str, list[str]]) -> bool:
    current_terms = set(current.get("freshness_terms", []))
    cached_terms = set(cached.get("freshness_terms", []))
    return bool(current_terms and cached_terms and current_terms != cached_terms)


def _disjoint_nonempty(current: dict[str, list[str]], cached: dict[str, list[str]], key: str) -> bool:
    left = set(current.get(key, []))
    right = set(cached.get(key, []))
    return bool(left and right and not left.intersection(right))


def _named_entities(query: str) -> list[str]:
    entities: list[str] = []
    for match in _CAPITALIZED_PHRASE_RE.finditer(query):
        raw = match.group(0).strip()
        raw = " ".join(token for token in raw.split() if token not in _ENTITY_STOPWORDS)
        if not raw:
            continue
        normalized = _normalize_entity(raw)
        if normalized.isdigit():
            continue
        if normalized:
            entities.append(normalized)

    normalized = normalize_query_text(query)
    for pair in _route_pairs(normalized):
        origin, destination = pair.split(">", 1)
        for endpoint in (origin, destination):
            if endpoint:
                entities.append(endpoint)
    return _sorted_unique(entities)


def _normalize_entity(value: str) -> str:
    normalized = " ".join(value.lower().replace("'s", "").split())
    normalized = _SYNONYMS.get(normalized, normalized)
    return normalized


def _contrast_facets(normalized: str) -> list[str]:
    padded = f" {normalized} "
    facets: list[str] = []
    if re.search(r"\b(what is|define|explain)\b", normalized):
        facets.append("answer_type:definition")
    if re.search(r"\bexample\b", normalized):
        facets.append("answer_type:example")
    if re.search(r"\b(when should|should i use|use recursion|use this)\b", normalized):
        facets.append("answer_type:use_case")
    if re.search(r"\b(used for|applications? of)\b", normalized):
        facets.append("answer_type:use_case")
    if re.search(r"\bhow many\b", normalized):
        facets.append("answer_type:quantity")
    if re.search(r"\b(symptoms?|look out for)\b", normalized):
        facets.append("answer_type:symptoms")
    if re.search(r"\b(treat|treatment|remedy|cure)\b", normalized):
        facets.append("answer_type:treatment")
    if re.search(r"\b(caused by|what caused|cause of|reasons?)\b", normalized):
        facets.append("answer_type:cause")
    if re.search(r"\b(overview|summarize|tell me about)\b", normalized):
        facets.append("answer_type:overview")
    if re.search(r"\b(recipe|bake|make a cake)\b", normalized):
        facets.append("answer_type:recipe")

    for group, terms in _CONTRAST_GROUPS:
        if group == "answer_type":
            continue
        for term in terms:
            if re.search(rf"\b{re.escape(term)}\b", normalized):
                facets.append(f"{group}:{_facet_value(group, term)}")
    if re.search(r"\buse\b.+\bwith\b", normalized):
        facets.append("compare_mode:integration")
    return _sorted_unique(facets)


def _facet_value(_group: str, term: str) -> str:
    return _CONTRAST_SYNONYMS.get(term, term)


def _group_values(facets: list[str]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for facet in facets:
        if ":" not in facet:
            continue
        group, value = facet.split(":", 1)
        grouped.setdefault(group, set()).add(value)
    return grouped


def _comparator_groups(normalized: str) -> list[str]:
    found: list[str] = []
    padded = f" {normalized} "
    for group, terms in _COMPARATOR_GROUPS:
        for term in terms:
            if term in ("<", ">", "<=", ">="):
                if term in normalized:
                    found.append(group)
            elif f" {term} " in padded:
                found.append(group)
    return found


def _terms(pattern: re.Pattern[str], normalized: str) -> list[str]:
    terms: list[str] = []
    for match in pattern.finditer(normalized):
        term = match.group(1).strip(".,;:!?").lower()
        if term and term not in _TERM_STOPWORDS:
            terms.append(term)
    return _sorted_unique(terms)


def _normalize_number(value: str) -> str:
    return value.lower().replace("$", "").replace(",", "").rstrip("%")


def _normalize_unit(value: str) -> str:
    raw = value.lower()
    return _SYNONYMS.get(raw, raw)


def _currency_pairs(normalized: str) -> list[str]:
    currencies = [_normalize_unit(match.group(0)) for match in _UNIT_RE.finditer(normalized)]
    currencies = [currency for currency in currencies if currency in {"usd", "eur", "gbp", "jpy"}]
    if len(currencies) < 2:
        return []
    return [f"{currencies[0]}>{currencies[1]}"]


def _route_pairs(normalized: str) -> list[str]:
    match = re.search(r"\bfrom\s+([a-z0-9. -]+?)\s+to\s+([a-z0-9. -]+?)(?:\?|$|\s+(?:by|via|with|today|tomorrow|now))", normalized)
    if not match:
        return []
    origin = _clean_route_endpoint(match.group(1))
    destination = _clean_route_endpoint(match.group(2))
    if not origin or not destination:
        return []
    return [f"{origin}>{destination}"]


def _clean_route_endpoint(value: str) -> str:
    stop = {"airport", "best", "way", "get", "the", "a", "an"}
    tokens = [token.strip(" .") for token in value.split() if token.strip(" .") and token not in stop]
    return " ".join(tokens)


def _product_models(normalized: str) -> list[str]:
    models: list[str] = []
    iphone = re.search(r"\biphone\s+(\d+)\s+pro(?:\s+max)?\b", normalized)
    if iphone:
        suffix = " max" if "pro max" in iphone.group(0) else ""
        models.append(f"iphone {iphone.group(1)} pro{suffix}")
    return _sorted_unique(models)


def _sorted_unique(values: Any) -> list[str]:
    return sorted({str(value).lower() for value in values if str(value).strip()})
