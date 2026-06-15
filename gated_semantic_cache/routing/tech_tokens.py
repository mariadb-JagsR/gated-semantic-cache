from __future__ import annotations

import re

# Dotted runtime/framework tokens that must not count as hostnames or stable identifiers.
KNOWN_DOTTED_TECH_TOKENS: frozenset[str] = frozenset(
    {
        "node.js",
        "next.js",
        "vue.js",
        "nuxt.js",
        "nest.js",
        "express.js",
        "three.js",
        "d3.js",
        "chart.js",
        "socket.io",
        "react.js",
        "angular.js",
    }
)

_DOTTED_TECH_SUFFIX_RE = re.compile(
    r"^(?:node|next|vue|nuxt|nest|express|three|react|angular|ember|backbone|dojo|polymer)\."
    r"(?:js|ts|jsx|tsx|mjs|cjs)$",
    re.IGNORECASE,
)


def normalize_dotted_token(token: str) -> str:
    return token.strip().lower()


def is_dotted_tech_token(token: str) -> bool:
    normalized = normalize_dotted_token(token)
    if normalized in KNOWN_DOTTED_TECH_TOKENS:
        return True
    return bool(_DOTTED_TECH_SUFFIX_RE.match(normalized))


def is_plausible_hostname(token: str) -> bool:
    normalized = normalize_dotted_token(token)
    if is_dotted_tech_token(normalized):
        return False
    if "." not in normalized:
        return False
    return True


def filter_plausible_hostnames(matches: list[str]) -> list[str]:
    return [match for match in matches if is_plausible_hostname(match)]
