"""
CTC-FA v2.0    --   L11 : perturbation engine, with the on-shell gate enforced.

v1.0 provided ``perturb_metric(model, h, epsilon)``, which added an arbitrary
symmetric tensor to the metric and returned a new ``MetricModel`` with
``expected = UNRESOLVED``. Nothing prevented the causal margin of that
object from being quoted as a stability result, and the architecture's own
gate says why that is invalid:

    L11 gate    "Off-shell perturbations cannot establish physical robustness."
    FM-005      "Off-shell perturbation used as physical stability evidence"
                -> INVALID PHYSICS.

v2.0 makes the gate structural. A perturbation carries an
:class:`OnShellStatus`, computed by measuring how far the perturbed metric
moves the field-equation residual, and any causal statement derived from an
``OFF_SHELL`` perturbation is emitted at ``ClaimScope.SECTOR`` with an
explicit ``exploratory`` flag that the reporting layer prints.

Three admissible perturbation classes are supported:

  EXACT_FAMILY        move along a registered exact-solution family by varying
                      its parameters: on shell by construction;
  CONSTRAINT_SOLVED     a perturbation whose field-equation residual is solved
                      or verified below a declared tolerance;
  OFF_SHELL           anything else -- exploratory only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import sympy as sp

from .causality import compact_orbit_test
from .model import MetricModel, Provenance
from .status import ClaimScope, CTCStatus, OrbitStatus


class OnShellStatus(str, Enum):
    EXACT_FAMILY = "EXACT_FAMILY"
    CONSTRAINT_SOLVED = "CONSTRAINT_SOLVED"
    OFF_SHELL = "OFF_SHELL"
    UNKNOWN = "UNKNOWN"


@dataclass
class PerturbationResult:
    model_id: str
    kind: OnShellStatus
    epsilon: float
    base_margin: float
    perturbed_margin: float
    residual_before: float
    residual_after: float
    status_before: str
    status_after: str
    status_changed: bool
    exploratory: bool
    note: str = ""

    @property
    def sensitivity(self) -> float:
        if self.epsilon == 0:
            return float("nan")
        return (self.perturbed_margin - self.base_margin) / self.epsilon

    def as_dict(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "kind": self.kind.value,
                "epsilon": self.epsilon, "base_margin": self.base_margin,
                "perturbed_margin": self.perturbed_margin,
                "sensitivity": self.sensitivity,
                "residual_before": self.residual_before,
                "residual_after": self.residual_after,
                "status_before": self.status_before,
                "status_after": self.status_after,
                "status_changed": self.status_changed,
                "exploratory": self.exploratory, "note": self.note}


def _residual_norm(model: MetricModel, values: Mapping[str, float]) -> float:
    from .energy import symbolic_einstein
    try:
         G = symbolic_einstein(model)
         Lam = model.matter.cosmological_constant
         T = model.matter.T
         rhs = sp.zeros(model.dim, model.dim) if T is None else 8 * sp.pi * T
         res = sp.Matrix(G + Lam * model.metric - rhs)
         sub = model.substitution(values)
         arr = np.array(res.subs(sub).evalf(30), dtype=float)
         return float(np.max(np.abs(arr)))
    except Exception:                                     # noqa: BLE001
         return float("nan")


def perturb_metric(model: MetricModel, h: sp.Matrix, epsilon: sp.Symbol | float,
                   kind: OnShellStatus = OnShellStatus.OFF_SHELL,
                   note: str = "") -> MetricModel:
    """Return ``g + epsilon h`` as a new model, tagged with its on-shell status."""
    g = model.require_metric()
    if h.shape != g.shape:
        raise ValueError("perturbation shape mismatch")
    if not sp.simplify(h - h.T).is_zero_matrix:
        raise ValueError("perturbation tensor must be symmetric")
    eps = sp.Symbol("epsilon_pert", real=True) if not isinstance(
        epsilon, (int, float, sp.Expr)) else epsilon
    new = MetricModel(
        id=model.id + "_pert", title=model.title + " + perturbation",
        coords=model.coords, metric=sp.Matrix(g + eps * h),
        params=tuple(model.params) + ((eps,) if isinstance(eps, sp.Symbol)
                                       and eps not in model.params else ()),
        signature=model.signature, compact=model.compact,
        quotient=model.quotient, kind=model.kind, theory=model.theory,
        matter=model.matter, topology=model.topology,
        expected=CTCStatus.UNRESOLVED, expected_scope=ClaimScope.SECTOR,
        domain=model.domain, default_point=dict(model.default_point),
        reference_timelike=model.reference_timelike,
        domain_notes=("PERTURBED MODEL. " + (note or "") +
                       " On-shell status: " + kind.value),
        provenance=Provenance(model.provenance.source_refs,
                               model.provenance.convention_note,
                               "perturbation of a registry model",
                               "derived"),
        metadata={**model.metadata, "perturbation_kind": kind.value,
                  "exploratory": kind is OnShellStatus.OFF_SHELL})
    return new


def parameter_family_perturbation(model: MetricModel, parameter: str,
                                  values: Mapping[str, float],
                                  delta: float) -> PerturbationResult:
    """Move along the registered exact family: on shell by construction."""
    base = dict(values)
    pert = dict(values)
    pert[parameter] = float(values[parameter]) + delta
    r0 = compact_orbit_test(model, base)
    r1 = compact_orbit_test(model, pert)
    return PerturbationResult(
        model.id, OnShellStatus.EXACT_FAMILY, delta, r0.margin, r1.margin,
        _residual_norm(model, base), _residual_norm(model, pert),
        r0.status.value, r1.status.value, r0.status is not r1.status, False,
        f"variation of the exact-solution parameter {parameter!r}: the "
        f"perturbed configuration is another exact solution of the same "
        f"family, so the field equations are satisfied identically and the "
        f"result is physical robustness evidence, not an off-shell probe")


def tensor_perturbation(model: MetricModel, h: sp.Matrix, epsilon: float,
                        values: Mapping[str, float],
                        residual_tol: float = 1e-8) -> PerturbationResult:
    """Add an arbitrary symmetric tensor and *measure* whether it stays on shell."""
    pert = perturb_metric(model, h, epsilon, OnShellStatus.OFF_SHELL)
    base_vals = dict(values)
    pert_vals = dict(values)
    r0 = compact_orbit_test(model, base_vals)
    try:
         r1 = compact_orbit_test(pert, pert_vals)
         m1, s1 = r1.margin, r1.status.value
    except Exception as exc:                              # noqa: BLE001
         m1, s1 = float("nan"), f"evaluation failed: {exc}"
    res0 = _residual_norm(model, base_vals)
    res1 = _residual_norm(pert, pert_vals)
    on_shell = (np.isfinite(res1) and abs(res1 - (res0 if np.isfinite(res0) else 0.0))
                 <= residual_tol)
    kind = OnShellStatus.CONSTRAINT_SOLVED if on_shell else OnShellStatus.OFF_SHELL
    return PerturbationResult(
         model.id, kind, epsilon, r0.margin, m1, res0, res1,
         r0.status.value, s1, r0.status.value != s1,
         exploratory=kind is OnShellStatus.OFF_SHELL,
         note=("field-equation residual moved from "
               f"{res0:.3e} to {res1:.3e}; "
               + ("within the declared tolerance, so the perturbed metric is "
                  "treated as constraint-solved"
                  if on_shell else
                  "OUTSIDE the declared tolerance, so this perturbation is "
                  "OFF SHELL and its causal margin is EXPLORATORY ONLY: it "
                  "cannot be quoted as stability evidence (v1.0 FM-005)")))


def robustness_scan(model: MetricModel, parameter: str,
                     values: Mapping[str, float],
                     deltas: Sequence[float]) -> list[dict[str, Any]]:
    """On-shell robustness of the causal verdict along an exact family."""
    rows = []
    for d in deltas:
        r = parameter_family_perturbation(model, parameter, values, d)
        rows.append({"delta": float(d), "margin": r.perturbed_margin,
                      "status": r.status_after, "changed": r.status_changed,
                      "sensitivity": r.sensitivity})
    return rows


__all__ = ["OnShellStatus", "PerturbationResult", "perturb_metric",
           "parameter_family_perturbation", "tensor_perturbation",
           "robustness_scan"]
