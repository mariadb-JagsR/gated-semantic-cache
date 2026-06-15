from __future__ import annotations

import re
import unicodedata


WHITESPACE_RE = re.compile(r"\s+")
MONEY_SYMBOL_RE = re.compile(r"\$(\d+(?:\.\d+)?)")
DIMENSION_SPACE_RE = re.compile(r"(\d+)\s*[xX]\s*(\d+)")
LAST_SEVEN_DAYS_RE = re.compile(r"\b(last seven days|past week|last week)\b")
LAST_THIRTY_DAYS_RE = re.compile(r"\b(last thirty days|past month|last month)\b")
PUNCT_RE = re.compile(r"[“”]")


def normalize_query(text: str) -> str:
    text = normalize_unicode(text)
    text = text.lower().strip()
    text = normalize_currency(text)
    text = normalize_dimensions(text)
    text = normalize_ranges(text)
    text = collapse_whitespace(text)
    return text


def normalize_unicode(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return PUNCT_RE.sub('"', normalized)


def collapse_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_currency(text: str) -> str:
    text = MONEY_SYMBOL_RE.sub(r"\1 usd", text)
    text = text.replace("dollars", "usd")
    return text


def normalize_dimensions(text: str) -> str:
    return DIMENSION_SPACE_RE.sub(r"\1x\2", text)


def normalize_ranges(text: str) -> str:
    text = LAST_SEVEN_DAYS_RE.sub("last 7 days", text)
    text = LAST_THIRTY_DAYS_RE.sub("last 30 days", text)
    return text
