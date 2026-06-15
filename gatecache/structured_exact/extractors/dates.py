from __future__ import annotations

import re

from gatecache.structured_exact.schema import Constraint


LAST_DAYS_RE = re.compile(r"\blast (\d+) days\b", re.I)


def extract_dates(text: str) -> list[Constraint]:
    constraints: list[Constraint] = []
    for match in LAST_DAYS_RE.finditer(text):
        constraints.append(
            Constraint(
                kind="date_window",
                name="relative_days",
                value=f"last_{int(match.group(1))}_days",
                confidence=0.95,
                span_text=match.group(0),
            )
        )
    return constraints
