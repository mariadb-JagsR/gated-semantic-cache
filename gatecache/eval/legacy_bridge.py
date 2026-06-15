"""Load vendored legacy eval datasets and JSON artifacts for regression tests."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType


def legacy_eval_dir() -> Path:
    """Vendored prompts from the old prototype eval harness (`tests/fixtures/legacy_eval/`)."""
    return Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "legacy_eval"


@lru_cache(maxsize=1)
def load_legacy_datasets_module() -> ModuleType | None:
    path = legacy_eval_dir() / "datasets.py"
    if not path.is_file():
        return None
    name = "legacy_eval_datasets_runtime"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass(frozen=True, slots=True)
class LegacyEvalCaseView:
    """Minimal view of legacy `EvalCase` without importing its type."""

    session_id: str
    user_message: str
    expected_hit_allowed: bool
    expected_bypass: bool
    expected_answer_class: str
    expected_scope: str


def iter_legacy_dataset_cases() -> Iterator[LegacyEvalCaseView]:
    mod = load_legacy_datasets_module()
    if mod is None:
        return
    build = getattr(mod, "build_comprehensive_dataset", None)
    if build is None:
        return
    for row in build():
        yield LegacyEvalCaseView(
            session_id=row.session_id,
            user_message=row.user_message,
            expected_hit_allowed=row.expected_hit_allowed,
            expected_bypass=row.expected_bypass,
            expected_answer_class=row.expected_answer_class,
            expected_scope=row.expected_scope,
        )


def iter_legacy_general_robustness_cases() -> Iterator[LegacyEvalCaseView]:
    """IBM / Fin-inspired FAQ-style threads (`build_general_robustness_dataset`)."""
    mod = load_legacy_datasets_module()
    if mod is None:
        return
    build = getattr(mod, "build_general_robustness_dataset", None)
    if build is None:
        return
    for row in build():
        yield LegacyEvalCaseView(
            session_id=row.session_id,
            user_message=row.user_message,
            expected_hit_allowed=row.expected_hit_allowed,
            expected_bypass=row.expected_bypass,
            expected_answer_class=row.expected_answer_class,
            expected_scope=row.expected_scope,
        )


def iter_legacy_retrieval_ablation_cases() -> Iterator[LegacyEvalCaseView]:
    """Rows from legacy `build_retrieval_ablation_dataset()` (domain / ANN eval slice)."""
    mod = load_legacy_datasets_module()
    if mod is None:
        return
    build = getattr(mod, "build_retrieval_ablation_dataset", None)
    if build is None:
        return
    for row in build():
        yield LegacyEvalCaseView(
            session_id=row.session_id,
            user_message=row.user_message,
            expected_hit_allowed=row.expected_hit_allowed,
            expected_bypass=row.expected_bypass,
            expected_answer_class=row.expected_answer_class,
            expected_scope=row.expected_scope,
        )


def iter_legacy_novel_domain_cases() -> Iterator[LegacyEvalCaseView]:
    mod = load_legacy_datasets_module()
    if mod is None:
        return
    build = getattr(mod, "build_novel_domain_dataset", None)
    if build is None:
        return
    for row in build():
        yield LegacyEvalCaseView(
            session_id=row.session_id,
            user_message=row.user_message,
            expected_hit_allowed=row.expected_hit_allowed,
            expected_bypass=row.expected_bypass,
            expected_answer_class=row.expected_answer_class,
            expected_scope=row.expected_scope,
        )


def unique_legacy_dataset_messages() -> list[str]:
    seen: dict[str, None] = {}
    for c in iter_legacy_dataset_cases():
        msg = c.user_message.strip()
        if msg:
            seen.setdefault(msg, None)
    for c in iter_legacy_novel_domain_cases():
        msg = c.user_message.strip()
        if msg:
            seen.setdefault(msg, None)
    return list(seen.keys())


def unique_messages_for_structured_eval() -> list[str]:
    """Deduped user prompts from vendored legacy datasets plus fixture JSON trace `user_message` fields."""
    seen: dict[str, None] = {}
    for it in (
        iter_legacy_dataset_cases(),
        iter_legacy_novel_domain_cases(),
        iter_legacy_general_robustness_cases(),
        iter_legacy_retrieval_ablation_cases(),
    ):
        for c in it:
            msg = c.user_message.strip()
            if msg:
                seen.setdefault(msg, None)
    for msg in unique_user_messages_from_legacy_eval_jsons():
        seen.setdefault(msg, None)
    return list(seen.keys())


def _walk_user_messages(obj: object, out: list[str]) -> None:
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key == "user_message" and isinstance(val, str) and val.strip():
                out.append(val.strip())
            else:
                _walk_user_messages(val, out)
    elif isinstance(obj, list):
        for item in obj:
            _walk_user_messages(item, out)


def unique_user_messages_from_legacy_eval_jsons() -> list[str]:
    """All distinct `user_message` strings under vendored `tests/fixtures/legacy_eval/*.json`."""
    eval_dir = legacy_eval_dir()
    if not eval_dir.is_dir():
        return []
    seen: dict[str, None] = {}
    for path in sorted(eval_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        buf: list[str] = []
        _walk_user_messages(payload, buf)
        for msg in buf:
            seen.setdefault(msg, None)
    return list(seen.keys())
