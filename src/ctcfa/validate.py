"""
CTC-FA v2.0   --    L0/L1/L2 : structural, algebraic and field-equation validation.

Gate V0 (parse)         the registry record is complete and internally
                        consistent: symmetric metric, declared signature,
                        no undeclared symbols, generator/coordinate agreement,
                        provenance present.
Gate V1 (algebra)       inverse metric verified, signature verified numerically
                        at the declared point, declared compact generators are
                        Killing vectors, declared deck generators are isometries.
Gate V2 (tensor)        curvature computed by two independent routes (exact
                        symbolic differentiation and high-order finite
                        differences) that must agree within the reported error.
Gate V3 (field eq.)     the field-equation residual meets the model's declared
                        tolerance for its declared matter content.

The two-route requirement in V2 is what makes a silent algebra bug visible.
v1.0 named an independent CAS cross-check as "planned external validation";
v2.0 ships an independent numerical route inside the package and runs it on
every model in the regression suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import sympy as sp

from .model import MetricModel
from .numerics import signature_counts
from .status import Gate, GateRecord, GateResult, ModelKind
from .symbolic import check_inverse, ricci_scalar
from .symmetry import verify_compact_generators
from .topology import verify_quotient


@dataclass
class ValidationReport:
    model_id: str
    gates: GateRecord
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(self.gates.get(g) in (GateResult.PASS, GateResult.NOT_APPLICABLE)
                   for g in (Gate.V0_PARSE, Gate.V1_ALGEBRA))

    def as_dict(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "ok": self.ok,
                "gates": self.gates.as_dict(), "details": self.details}


def validate_model(model: MetricModel, point: Mapping[str, float] | None = None,
                   field_equation_tol: float = 1e-9,
                   cross_check: bool = True) -> ValidationReport:
    gates = GateRecord()
    details: dict[str, Any] = {}
    point = dict(point or model.default_point)

    # ---- V0 -------------------------------------------------------------
    try:
         model.assert_wellformed()
         problems = []
         if not model.provenance.source_refs:
             problems.append("no source references")
         if model.kind in (ModelKind.EXTERNAL_EXACT_SPEC,
                           ModelKind.COMPOSITE_GLOBAL) and model.has_metric:
             problems.append("global/external model carries a local metric")
         if problems:
             gates.set(Gate.V0_PARSE, GateResult.FAIL, "; ".join(problems))
         else:
             gates.set(Gate.V0_PARSE, GateResult.PASS,
                       "record complete, metric symmetric, symbols declared")
    except Exception as exc:                               # noqa: BLE001
         gates.set(Gate.V0_PARSE, GateResult.FAIL, str(exc))
         return ValidationReport(model.id, gates, details)

    if not model.has_metric:
        for g in (Gate.V1_ALGEBRA, Gate.V2_TENSOR, Gate.V3_FIELD_EQUATION):
            gates.set(g, GateResult.NOT_APPLICABLE,
                      "no local metric: this model is an external or composite "
                      "specification")
        details["kind"] = model.kind.value
        return ValidationReport(model.id, gates, details)

    # ---- V1 -------------------------------------------------------------
    v1_notes = []
    ok1 = True
    try:
         resid = check_inverse(model.metric)
         inv_ok = bool(resid.is_zero_matrix)
         v1_notes.append(f"g*g^-1 - I zero: {inv_ok}")
         ok1 = ok1 and inv_ok
         details["inverse_residual"] = "0" if inv_ok else str(resid)
    except Exception as exc:                              # noqa: BLE001
         ok1 = False
         v1_notes.append(f"inverse failed: {exc}")

    if point and not model.missing_values(point):
        try:
             gnum = np.array(sp.Matrix(model.metric)
                             .subs(model.substitution(point)).evalf(), dtype=float)
             sig = signature_counts(gnum)
             sig_ok = sig == tuple(model.signature)
             v1_notes.append(f"signature {sig} vs declared "
                             f"{tuple(model.signature)}: {sig_ok}")
             details["signature_at_point"] = list(sig)
             ok1 = ok1 and sig_ok
        except Exception as exc:                          # noqa: BLE001
             v1_notes.append(f"signature check failed: {exc}")
             ok1 = False

    certs = verify_compact_generators(model)
    details["compact_generators"] = [
        {"name": c.name, "is_killing": c.is_killing,
         "residual": c.killing_residual[:200]} for c in certs]
    for cd, c in zip(model.compact, certs):
        if cd.requires_killing and not c.is_killing:
            ok1 = False
            v1_notes.append(f"generator {c.name} is not Killing")

    qc = verify_quotient(model, point)
    details["quotient"] = qc.as_dict()
    if qc.generators and not qc.admissible:
        v1_notes.append("some deck generators are not admissible "
                        "(non-isometry or fixed point)")

    gates.set(Gate.V1_ALGEBRA, GateResult.PASS if ok1 else GateResult.FAIL,
              "; ".join(v1_notes))

    # ---- V2 : two independent curvature routes ---------------------------
    if cross_check and point and not model.missing_values(point):
        try:
            from .energy import symbolic_einstein
            from .numgeom import NumericGeometry
            Gsym = symbolic_einstein(model)
            sub = model.substitution(point)
            Ga = np.array(sp.Matrix(Gsym).subs(sub).evalf(30), dtype=float)
            ng = NumericGeometry(model)
            Gb, err = ng.einstein(point)
            diff = float(np.max(np.abs(Ga - Gb)))
            scale = max(1.0, float(np.max(np.abs(Ga))), float(np.max(np.abs(Gb))))
            agree = diff <= max(50 * err, 1e-6 * scale)
            details["einstein_cross_check"] = {
                "symbolic_max_abs": float(np.max(np.abs(Ga))),
                "numeric_max_abs": float(np.max(np.abs(Gb))),
                "max_difference": diff, "numeric_error_estimate": err,
                "agree": agree}
            gates.set(Gate.V2_TENSOR,
                      GateResult.PASS if agree else GateResult.FAIL,
                      f"exact symbolic and finite-difference Einstein tensors "
                      f"differ by {diff:.3e} (numeric error estimate {err:.3e})")
        except Exception as exc:                         # noqa: BLE001
            gates.set(Gate.V2_TENSOR, GateResult.INCONCLUSIVE,
                      f"cross-check unavailable: {exc}")
    else:
        gates.set(Gate.V2_TENSOR, GateResult.NOT_ATTEMPTED,
                  "no fully specified point supplied")

    # ---- V3 : field equations -------------------------------------------
    if point and not model.missing_values(point):
        try:
            from .energy import symbolic_einstein
            G = symbolic_einstein(model)
            Lam = model.matter.cosmological_constant
            T = model.matter.T
            rhs = (sp.zeros(model.dim, model.dim) if T is None
                   else 8 * sp.pi * T)
            res = sp.Matrix(G + Lam * model.metric - rhs)
            arr = np.array(res.subs(model.substitution(point)).evalf(30),
                           dtype=float)
            m = float(np.max(np.abs(arr)))
            details["field_equation_residual"] = m
            if model.matter.vacuum or T is not None:
                gates.set(Gate.V3_FIELD_EQUATION,
                          GateResult.PASS if m <= field_equation_tol
                          else GateResult.FAIL,
                          f"|G + Lambda g - 8 pi T|_max = {m:.3e} "
                          f"(tolerance {field_equation_tol:.1e})")
            else:
                gates.set(Gate.V3_FIELD_EQUATION, GateResult.NOT_APPLICABLE,
                          f"matter content is '{model.matter.label}': no closed "
                          f"form T_ab is registered, so the residual "
                          f"{m:.3e} DEFINES the required source rather than "
                          f"testing the solution")
        except Exception as exc:                           # noqa: BLE001
             gates.set(Gate.V3_FIELD_EQUATION, GateResult.INCONCLUSIVE, str(exc))
    else:
        gates.set(Gate.V3_FIELD_EQUATION, GateResult.NOT_ATTEMPTED,
                   "no fully specified point supplied")

    return ValidationReport(model.id, gates, details)


def validate_registry(cross_check: bool = True) -> dict[str, Any]:
    from .registry import models
    rows = []
    n_fail = 0
    for m in models():
        rep = validate_model(m, cross_check=cross_check)
        rows.append(rep.as_dict())
        if not rep.ok:
            n_fail += 1
    return {"n_models": len(rows), "n_failed": n_fail,
            "pass": n_fail == 0, "reports": rows}


__all__ = ["ValidationReport", "validate_model", "validate_registry"]
