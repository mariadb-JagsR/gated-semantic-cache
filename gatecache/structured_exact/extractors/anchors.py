from __future__ import annotations

import re


STOPWORDS = {
    "a",
    "all",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "find",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "lookup",
    "me",
    "my",
    "of",
    "on",
    "possible",
    "placed",
    "submitted",
    "show",
    "size",
    "status",
    "that",
    "the",
    "today",
    "to",
    "what",
    "with",
}
NON_ANCHOR_TOKENS = {
    "brown",
    "cotton",
    "stretch",
    "nonstop",
    "refundable",
    "wireless",
    "organic",
    "usd",
}


def extract_anchors(text: str) -> tuple[str, ...]:
    candidates: list[str] = []
    for token in re.findall(r"[a-z]{3,}", text):
        if token in STOPWORDS or token in NON_ANCHOR_TOKENS:
            continue
        if token.endswith("ing") or token.endswith("ed"):
            continue
        candidates.append(token)
    deduped = _dedupe_preserve_order(candidates)
    if not deduped:
        return ()
    return (deduped[0],)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output
