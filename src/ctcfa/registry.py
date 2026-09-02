"""
CTC-FA v2.0   --   Registry access, integrity and machine-readable export.

The registry is the single source of truth. Loading it runs the V0 structural
gate on every entry (symmetry, shape, declared symbols, generator/coordinate
consistency), so a malformed model cannot enter the system at all.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator

from .metrics import all_models
from .model import MetricModel
from .status import CTCStatus, ClaimScope, ModelKind


@lru_cache(maxsize=1)
def REGISTRY() -> dict[str, MetricModel]:
    return all_models()


def get(name: str) -> MetricModel:
    reg = REGISTRY()
    if name not in reg:
        raise KeyError(f"unknown model {name!r}. Registered: {sorted(reg)}")
    return reg[name]


def ids() -> tuple[str, ...]:
    return tuple(sorted(REGISTRY()))


def models() -> Iterator[MetricModel]:
    for k in ids():
        yield REGISTRY()[k]


def with_metric() -> tuple[str, ...]:
    return tuple(m.id for m in models() if m.has_metric)


def numeric_models() -> tuple[str, ...]:
    """Models that can be evaluated numerically at their default point."""
    out = []
    for m in models():
        if not m.has_metric:
             continue
        if m.metadata.get("numeric_evaluable") is False:
             continue
        if not m.default_point:
             continue
        out.append(m.id)
    return tuple(out)


def by_role(role: str) -> tuple[str, ...]:
    return tuple(m.id for m in models() if m.metadata.get("role") == role)


def by_kind(kind: ModelKind) -> tuple[str, ...]:
    return tuple(m.id for m in models() if m.kind is kind)


def summary() -> list[dict[str, Any]]:
    rows = []
    for m in models():
        rows.append({
            "model_id": m.id, "title": m.title, "kind": m.kind.value,
            "dim": m.dim, "has_metric": m.has_metric,
            "n_params": len(m.params), "n_compact": len(m.compact),
            "n_deck": len(m.quotient.generators),
            "expected": m.expected.value, "scope": m.expected_scope.value,
            "role": m.metadata.get("role", ""),
            "refs": ",".join(m.provenance.source_refs),
        })
    return rows


def export_records(directory: str | Path) -> list[Path]:
    """Write one YAML-ish JSON record per model (schemas/metric.schema.json)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = []
    for m in models():
        p = directory / f"{m.id}.json"
        p.write_text(json.dumps(m.as_record(), indent=2, sort_keys=True,
                                ensure_ascii=True) + "\n", encoding="utf-8")
        out.append(p)
    return out


def registry_audit() -> dict[str, Any]:
    """REQ-015 / REQ-016 audit: provenance completeness and kind discipline."""
    problems: list[str] = []
    for m in models():
        if not m.provenance.source_refs:
            problems.append(f"{m.id}: no source references")
        if m.kind in (ModelKind.EXTERNAL_EXACT_SPEC, ModelKind.COMPOSITE_GLOBAL) \
                and m.has_metric:
            problems.append(f"{m.id}: {m.kind.value} must not carry a local metric")
        if m.kind is ModelKind.LOCAL_METRIC and not m.has_metric:
            problems.append(f"{m.id}: LOCAL_METRIC without a metric")
        if m.has_metric and not m.domain_notes:
            problems.append(f"{m.id}: empty domain notes")
        if m.global_margin is not None and \
                m.global_margin_scope is ClaimScope.GLOBAL:
            problems.append(f"{m.id}: reduced predicate declared at GLOBAL scope")
        if m.expected is CTCStatus.NO_CTC_CERTIFIED and \
                m.expected_scope is ClaimScope.GLOBAL and \
                m.temporal_function is None:
            problems.append(f"{m.id}: GLOBAL no-CTC expectation without a "
                            f"candidate temporal function")
    return {"n_models": len(ids()), "n_problems": len(problems),
            "problems": problems, "pass": not problems}


__all__ = ["REGISTRY", "get", "ids", "models", "with_metric", "numeric_models",
           "by_role", "by_kind", "summary", "export_records", "registry_audit"]
