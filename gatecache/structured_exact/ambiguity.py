from __future__ import annotations

import re


AMBIGUOUS_REFERENCE_RE = re.compile(r"\b(this|that|it|same|again|one|those|these|other)\b", re.IGNORECASE)
FOLLOWUP_RE = re.compile(r"^(what about|same|instead|again|also|do that|use that)\b", re.IGNORECASE)


def detect_ambiguity(text: str) -> tuple[str, ...]:
    flags: list[str] = []
    if FOLLOWUP_RE.search(text):
        flags.append("followup_reference")
    if AMBIGUOUS_REFERENCE_RE.search(text):
        flags.append("ambiguous_reference")
    if len(text.split()) <= 5 and not any(char.isdigit() for char in text):
        flags.append("low_information")
    return tuple(sorted(set(flags)))
