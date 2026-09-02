"""
CTC-FA v2.0    --   Reporting layer.

The reporting layer is a *gate*, not a formatter. It accepts only
:class:`~ctcfa.status.Verdict` objects, refuses to print a claim above the
level the gate record supports, and always prints the scope. A caller that
tries to render a stronger statement gets :class:`OverclaimError`.

It also produces the two aggregate objects the architecture needs:

      ``analysis(model)``        the full per-model dossier across every layer;
      ``regression_matrix()``    the reproducible table that replaces the v1.0
                                 self-test, with the *derivation route* recorded
                                 for every row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import sympy as sp

from .causality import (certify_temporal_function, compact_orbit_test,
                        compact_orbit_verdict, global_predicate_verdict,
                        temporal_verdict)
from .model import MetricModel
from .quotient_ctc import deck_orbit_test, deck_orbit_verdict
from .status import (CLAIM_TEXT, ClaimLevel, ClaimScope, CTCStatus, Gate,
                     GateRecord, GateResult, OrbitStatus, OverclaimError, Verdict)


# --------------------------------------------------------------------------
# Route selection
# --------------------------------------------------------------------------

def primary_route(model: MetricModel) -> str:
    """Which engine is the appropriate primary detector for this model?"""
    if not model.has_metric:
        return "global_predicate"
    if model.compact:
        return "compact_orbit"
    if model.quotient.generators and model.metadata.get("flat_covering"):
        return "deck_orbit"
    if model.temporal_function is not None:
        return "temporal_function"
    return "none"


def analyse(model: MetricModel, values: Mapping[str, float] | None = None,
            include_energy: bool = False,
            include_theorems: bool = False) -> dict[str, Any]:
    """Full per-model dossier."""
    values = dict(values or model.default_point)
    route = primary_route(model)
    out: dict[str, Any] = {
        "model_id": model.id, "title": model.title, "kind": model.kind.value,
        "expected": model.expected.value,
        "expected_scope": model.expected_scope.value,
        "primary_route": route, "point": values,
        "provenance": {"refs": list(model.provenance.source_refs),
                       "convention": model.provenance.convention_note},
        "domain_notes": model.domain_notes,
    }

    verdicts: list[Verdict] = []
    if route == "compact_orbit":
        v = compact_orbit_verdict(model, values)
        verdicts.append(v)
        out["compact_orbit"] = v.as_dict()
    elif route == "deck_orbit":
        v = deck_orbit_verdict(model, values)
        verdicts.append(v)
        out["deck_orbit"] = v.as_dict()
    elif route == "global_predicate":
        if model.global_margin is not None:
            v = global_predicate_verdict(model, values)
            verdicts.append(v)
            out["global_predicate"] = v.as_dict()
        else:
            out["global_predicate"] = {
                "status": CTCStatus.UNRESOLVED.value,
                "reason": "no local metric and no registered predicate: this "
                           "entry is an external exact specification"}

    if model.temporal_function is not None and model.has_metric:
        try:
             c = certify_temporal_function(model)
             out["temporal_certificate"] = c.as_dict()
             if c.proved:
                 verdicts.append(temporal_verdict(model))
        except Exception as exc:                          # noqa: BLE001
             out["temporal_certificate"] = {"error": str(exc)}

    if model.quotient.generators and model.has_metric \
             and model.metadata.get("flat_covering") and route != "deck_orbit":
        try:
             d = deck_orbit_test(model, values)
             out["deck_orbit_secondary"] = d.as_evidence() | {
                 "status": d.status.value, "margin": d.margin}
        except Exception as exc:                          # noqa: BLE001
             out["deck_orbit_secondary"] = {"error": str(exc)}

    if include_energy and model.has_metric:
        try:
             from .energy import energy_report
             out["energy"] = energy_report(model, values).as_dict()
        except Exception as exc:                          # noqa: BLE001
             out["energy"] = {"error": str(exc)}

    if include_theorems:
        from .theorems import applicability_matrix
        out["theorems"] = applicability_matrix(model)

    out["verdicts"] = [v.as_dict() for v in verdicts]
    out["headline"] = headline(model, verdicts)
    return out


def headline(model: MetricModel, verdicts: Sequence[Verdict]) -> dict[str, Any]:
    """Combine the verdicts, respecting the evidence asymmetry.

    A positive witness wins over an unresolved search. A negative certificate
    wins only when nothing positive was found, and only at its own scope.
    """
    positive = [v for v in verdicts if v.status is CTCStatus.CTC_PROVED]
    critical = [v for v in verdicts if v.status is CTCStatus.CRITICAL]
    negative = [v for v in verdicts if v.status is CTCStatus.NO_CTC_CERTIFIED]
    if positive:
        v = max(positive, key=lambda x: (x.scope.value, -x.margin))
    elif negative and not critical:
        v = negative[0]
    elif critical:
        v = critical[0]
    elif verdicts:
        v = verdicts[0]
    else:
        return {"status": CTCStatus.UNRESOLVED.value, "scope": "SECTOR",
                 "claim_level": ClaimLevel.L0_PARSED.name,
                 "line": f"{model.id}: UNRESOLVED (no applicable route)"}
    return {"status": v.status.value, "scope": v.scope.value,
            "claim_level": v.permitted_level().name,
            "claim_text": CLAIM_TEXT[v.permitted_level()],
            "margin": v.margin, "line": v.render()}


# --------------------------------------------------------------------------
# Regression matrix
# --------------------------------------------------------------------------

def regression_matrix(model_ids: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """The reproducible replacement for the v1.0 self-test table.

    Each row records the route that produced it, so a reader can tell a
    sector result from a global one at a glance -- the single most important
    piece of information the v1.0 table omitted.
    """
    from .registry import get, models
    ids = list(model_ids) if model_ids else [m.id for m in models()]
    rows = []
    for mid in ids:
        m = get(mid)
        route = primary_route(m)
        expected = (m.expected_at_default or m.expected).value
        row: dict[str, Any] = {"model_id": mid, "route": route,
                                "expected": expected,
                                "family_expected": m.expected.value,
                                "kind": m.kind.value,
                                "expected_scope": m.expected_scope.value}
        try:
             if route == "compact_orbit":
                 r = compact_orbit_test(m, m.default_point)
                 row.update({"actual": r.status.value, "margin": r.margin,
                             "uncertainty": r.uncertainty,
                             "winding": list(r.winding) if r.winding else None})
             elif route == "deck_orbit":
                 r = deck_orbit_test(m, m.default_point)
                 row.update({"actual": r.status.value, "margin": r.margin,
                             "uncertainty": r.uncertainty,
                             "word": list(r.word) if r.word else None})
             elif route == "global_predicate" and m.global_margin is not None:
                 v = global_predicate_verdict(m, m.default_point or {})
                 row.update({"actual": v.status.value, "margin": v.margin,
                             "scope": v.scope.value})
             elif route == "temporal_function":
                 c = certify_temporal_function(m)
                 row.update({"actual": c.status.value, "margin": float("nan"),
                             "scope": c.scope.value})
             else:
                 row.update({"actual": "NOT_APPLICABLE", "margin": float("nan")})
        except Exception as exc:                           # noqa: BLE001
             row.update({"actual": f"ERROR: {exc}", "margin": float("nan")})
        rows.append(row)
    return rows


def format_matrix(rows: Sequence[Mapping[str, Any]]) -> str:
    w = max(len(str(r["model_id"])) for r in rows) + 2
    out = [f"{'model':<{w}}{'route':<18}{'expected':<22}{'actual':<22}"
           f"{'margin':>16} uncertainty"]
    out.append("-" * (w + 18 + 22 + 22 + 16 + 14))
    for r in rows:
        m = r.get("margin")
        ms = "-" if m is None or (isinstance(m, float) and not np.isfinite(m)) \
            else f"{m:+.6e}"
        u = r.get("uncertainty")
        us = "-" if u is None else f"{u:.2e}"
        out.append(f"{r['model_id']:<{w}}{r['route']:<18}{r['expected']:<22}"
                   f"{str(r['actual']):<22}{ms:>16} {us}")
    return "\n".join(out)


def matrix_agreement(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
      """Does each row's computed status match the registry expectation?"""
      ok, bad = 0, []
      for r in rows:
          exp, act = r["expected"], str(r["actual"])
          # An EXTERNAL_EXACT_SPEC entry has no local tensor by design: the
          # architecture is *supposed* to answer NOT_APPLICABLE for it rather
          # than fabricate a metric (v1.0 REQ-016).
          if r.get("kind") in ("EXTERNAL_EXACT_SPEC",) and act == "NOT_APPLICABLE":
              ok += 1
              continue
          agree = (
              (exp == "CTC_PROVED" and act in ("SECTOR_CTC", "CTC_PROVED"))
              or (exp == "NO_CTC_CERTIFIED" and act in ("SECTOR_SPACELIKE",
                                                        "NO_CTC_CERTIFIED"))
              or (exp == "CRITICAL" and act in ("SECTOR_NULL", "CRITICAL"))
              or (exp == "UNRESOLVED" and act in ("SECTOR_SPACELIKE",
                                                  "SECTOR_UNRESOLVED",
                                                  "UNRESOLVED", "NOT_APPLICABLE",
                                                  "SECTOR_NULL"))
          )
          if agree:
              ok += 1
          else:
              bad.append({"model_id": r["model_id"], "expected": exp,
                          "actual": act, "route": r.get("route")})
      return {"n": len(rows), "agree": ok, "disagree": len(bad),
              "pass": not bad, "mismatches": bad}


def to_json(obj: Any) -> str:
    def default(o: Any) -> Any:
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, sp.Basic):
            return str(o)
        return str(o)
    return json.dumps(obj, indent=2, sort_keys=True, default=default)


__all__ = ["primary_route", "analyse", "headline", "regression_matrix",
           "format_matrix", "matrix_agreement", "to_json"]
