"""
CTC-FA v2.0   --   L13/L14 : quantum-state layer and semiclassical backreaction.

The rule this layer exists to enforce
-------------------------------------
    No generic renormalised stress tensor is inferred from a metric alone.

``<T_mu_nu>_ren`` is a functional of the quantum field content, the state, the
boundary conditions and the renormalisation prescription. A metric does not
determine it. Every "quantum" statement in the architecture therefore
requires a :class:`QuantumSpecification` and refuses to produce a number
without one. v1.0 stated this (L13 gate, REQ-011) and modelled it as a
dataclass of four ``"UNRESOLVED"`` strings; v2.0 makes it a real gate with
real inputs and a real applicability decision.

What is actually computable here
--------------------------------
* **Hadamard admissibility** on a compactly generated Cauchy horizon. The
  Kay-Radzikowski-Wald theorem shows the two-point function cannot be
  Hadamard on the whole horizon: there are base points at which the standard
  construction fails and ``<T_mu_nu>_ren`` is ill defined or singular. The
  engine checks the theorem's hypotheses and, when they hold, returns
  ``SEMICLASSICAL_QFT_BREAKDOWN`` at the identified base points -- not an
  invented infinity.

* **Polarised hypersurfaces** of a flat quotient (from ``quotient_ctc``) are
  exactly the loci where the image-sum two-point function of a free field
  diverges, so they are reported here as the *computed* candidate locations
  of RSET divergence. This is the one place where v2.0 can give a genuinely
  quantitative semiclassical statement, and it can do so only because the
  deck group is an executable object.

* **Quantum energy inequalities** are evaluated only against a registered
  rule keyed by field, dimension, state class, sampling function and
  trajectory. An unkeyed QEI is refused (v1.0 FM-012).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import sympy as sp

from .model import MetricModel


class QuantumStatus(str, Enum):
    UNRESOLVED = "UNRESOLVED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    ADMISSIBLE = "ADMISSIBLE"
    SEMICLASSICAL_QFT_BREAKDOWN = "SEMICLASSICAL_QFT_BREAKDOWN"
    RSET_DIVERGENT = "RSET_DIVERGENT"


@dataclass
class QuantumSpecification:
    """Everything that must be declared before any quantum statement."""

    field_content: str = ""
    spin: str = ""
    mass: float | None = None
    couplings: Mapping[str, float] = field(default_factory=dict)
    state: str = ""
    state_class: str = ""
    boundary_conditions: str = ""
    renormalisation: str = ""
    sampling_function: str = ""
    trajectory: str = ""
    spacetime_dimension: int = 4

    @property
    def complete(self) -> bool:
        return bool(self.field_content and self.state and self.renormalisation)

    def missing(self) -> list[str]:
        out = []
        for k in ("field_content", "state", "renormalisation"):
            if not getattr(self, k):
                 out.append(k)
        return out

    def as_dict(self) -> dict[str, Any]:
        return {"field_content": self.field_content, "spin": self.spin,
                "mass": self.mass, "couplings": dict(self.couplings),
                "state": self.state, "state_class": self.state_class,
                "boundary_conditions": self.boundary_conditions,
                "renormalisation": self.renormalisation,
                "sampling_function": self.sampling_function,
                "trajectory": self.trajectory,
                "spacetime_dimension": self.spacetime_dimension,
                "complete": self.complete, "missing": self.missing()}


@dataclass
class QuantumAssessment:
    model_id: str
    status: QuantumStatus
    hadamard: str
    qei: str
    renormalised_stress: str
    backreaction: str
    theorem_hypotheses: dict[str, Any] = field(default_factory=dict)
    divergence_loci: list[dict[str, Any]] = field(default_factory=list)
    spec: QuantumSpecification | None = None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "status": self.status.value,
                "hadamard": self.hadamard, "qei": self.qei,
                "renormalised_stress": self.renormalised_stress,
                "backreaction": self.backreaction,
                "theorem_hypotheses": self.theorem_hypotheses,
                "divergence_loci": self.divergence_loci,
                "specification": self.spec.as_dict() if self.spec else None,
                "note": self.note}


# --------------------------------------------------------------------------
# Kay-Radzikowski-Wald gate
# --------------------------------------------------------------------------

def krw_hypotheses(model: MetricModel) -> dict[str, Any]:
    """Record whether the KRW hypotheses are satisfied by this model.

    KRW applies to a spacetime with a **compactly generated** Cauchy horizon.
    The engine reports each hypothesis as satisfied / not satisfied /
    undetermined, and the theorem may only be invoked when none is 'not
    satisfied'.
    """
    meta = model.metadata
    has_horizon = bool(meta.get("chronology_horizon"))
    compactly_generated = meta.get("compactly_generated_horizon")
    if compactly_generated is None:
        # Misner-type quotients by a single boost have a compactly generated
        # chronology horizon; a periodic-time identification of a static
        # spacetime does not (the whole spacetime is chronology violating and
        # there is no horizon at all).
        if model.id in ("misner", "misner_covering", "grant_space"):
             compactly_generated = True
        elif has_horizon:
             compactly_generated = None
        else:
             compactly_generated = False
    return {
        "chronology_horizon_present": has_horizon,
        "compactly_generated": compactly_generated,
        "free_scalar_field_assumed": True,
        "four_dimensional": model.dim == 4,
        "applicable": bool(has_horizon and compactly_generated is True
                            and model.dim == 4),
        "note": ("KRW proves that the two-point function of a free scalar "
                  "field cannot be Hadamard on the whole of a compactly "
                  "generated Cauchy horizon; at the base points of the horizon "
                  "generators the standard construction fails and the "
                  "renormalised stress tensor is ill defined or singular. It "
                  "does NOT prove chronology protection in general."),
    }


def assess(model: MetricModel, spec: QuantumSpecification | None = None,
           polarized: Sequence[Mapping[str, Any]] | None = None
           ) -> QuantumAssessment:
    """L13/L14 assessment. Refuses to produce numbers without a specification."""
    hyp = krw_hypotheses(model)
    if spec is None or not spec.complete:
        missing = spec.missing() if spec else ["field_content", "state",
                                               "renormalisation"]
        return QuantumAssessment(
            model.id, QuantumStatus.UNRESOLVED, "UNRESOLVED", "UNRESOLVED",
            "UNRESOLVED", "UNRESOLVED", hyp, [], spec,
            note=("no quantum statement is possible: the specification is "
                  f"incomplete (missing {missing}). A renormalised stress "
                  "tensor is a functional of the field content, the state, the "
                  "boundary conditions and the renormalisation prescription, "
                  "and cannot be inferred from the metric alone."))

    loci = []
    if polarized:
        for row in polarized:
            loci.append({"winding": row.get("winding"),
                         "variable": row.get("variable"),
                         "value": row.get("value"),
                         "mechanism": "polarised hypersurface of the deck group: "
                                      "the image sum for the two-point function "
                                      "of a free field diverges here"})

    if hyp["applicable"]:
        status = QuantumStatus.SEMICLASSICAL_QFT_BREAKDOWN
        had = ("NOT HADAMARD on the whole horizon (KRW): base points exist at "
               "which the standard construction fails")
        rset = ("ill defined or singular at the KRW base points; no finite "
                "value is invented here")
        back = ("classical CTC survival does not imply semiclassical survival; "
                "the correct output is a breakdown label, not a backreaction "
                "number")
    elif loci:
        status = QuantumStatus.RSET_DIVERGENT
        had = "state-dependent; polarised hypersurfaces located below"
        rset = ("divergent on the listed polarised hypersurfaces for the "
                "declared state class")
        back = "backreaction expected to be strong near those loci"
    else:
        status = QuantumStatus.UNRESOLVED
        had = "undetermined for the declared state"
        rset = "not computed"
        back = "not computed"

    return QuantumAssessment(model.id, status, had, "requires a registered QEI rule",
                             rset, back, hyp, loci, spec,
                             note="assessment is valid only for the declared "
                                  "specification; changing the state or the "
                                  "renormalisation prescription invalidates it")


# --------------------------------------------------------------------------
# Quantum energy inequalities
# --------------------------------------------------------------------------

@dataclass
class QEIRule:
    """A registered quantum energy inequality, keyed by its assumptions."""

    key: str
    field_content: str
    dimension: int
    state_class: str
    sampling: str
    trajectory: str
    bound: Callable[[Mapping[str, float]], float]
    reference: str
    note: str = ""

    def applies_to(self, spec: QuantumSpecification) -> bool:
        return (spec.field_content == self.field_content
                and spec.spacetime_dimension == self.dimension
                and spec.state_class == self.state_class
                and spec.sampling_function == self.sampling
                and spec.trajectory == self.trajectory)


def _minkowski_massless_scalar_bound(p: Mapping[str, float]) -> float:
    """Ford-Roman-type bound for a massless scalar in 4D Minkowski.

    For a static inertial observer with a Lorentzian sampling function of
    width ``tau``,

        int rho(t) f(t) dt   >=   -3 / (32 pi^2 tau^4).

    Registered with its exact assumptions so that it can never be applied to
    a different field, dimension, state class, sampling function or
    trajectory.
    """
    tau = float(p["tau"])
    if tau <= 0:
        raise ValueError("sampling width must be positive")
    return -3.0 / (32.0 * np.pi ** 2 * tau ** 4)


QEI_REGISTRY: dict[str, QEIRule] = {
    "FR_massless_scalar_4d_inertial_lorentzian": QEIRule(
        key="FR_massless_scalar_4d_inertial_lorentzian",
        field_content="massless minimally coupled scalar",
        dimension=4, state_class="Hadamard",
        sampling="Lorentzian", trajectory="static inertial",
        bound=_minkowski_massless_scalar_bound,
        reference="Ford-Roman quantum inequality in 4D Minkowski",
        note="valid only in flat spacetime for the stated observer and "
             "sampling function; it is NOT a general bound on negative energy "
             "in curved spacetime"),
}


@dataclass
class QEIResult:
    rule: str | None
    bound: float | None
    applicable: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"rule": self.rule, "bound": self.bound,
                "applicable": self.applicable, "reason": self.reason}


def evaluate_qei(spec: QuantumSpecification,
                  parameters: Mapping[str, float]) -> QEIResult:
    """Apply a registered QEI, or refuse."""
    for rule in QEI_REGISTRY.values():
        if rule.applies_to(spec):
            try:
                 return QEIResult(rule.key, float(rule.bound(parameters)), True,
                                  f"{rule.reference}. {rule.note}")
            except Exception as exc:                      # noqa: BLE001
                 return QEIResult(rule.key, None, False,
                                  f"rule matched but evaluation failed: {exc}")
    return QEIResult(None, None, False,
                      "no registered quantum energy inequality matches this "
                      "field content, dimension, state class, sampling function "
                      "and trajectory. Applying a QEI outside its assumptions "
                      "produces an invalid bound (v1.0 FM-012), so none is "
                      "applied.")


def semiclassical_residual(model: MetricModel, T_classical: sp.Matrix,
                           T_ren: sp.Matrix | None,
                           Lambda: sp.Expr = sp.S.Zero) -> sp.Matrix | str:
    """G + Lambda g - 8 pi (T_classical + <T>_ren), or a refusal."""
    if T_ren is None:
        return ("refused: a semiclassical residual requires an explicitly "
                "supplied <T_mu_nu>_ren from a valid state-specific "
                "calculation. None is inferred from the metric.")
    from .energy import symbolic_einstein
    g = model.require_metric()
    G = symbolic_einstein(model)
    return sp.simplify(G + Lambda * g - 8 * sp.pi * (T_classical + T_ren))


__all__ = ["QuantumStatus", "QuantumSpecification", "QuantumAssessment",
           "krw_hypotheses", "assess", "QEIRule", "QEI_REGISTRY", "QEIResult",
           "evaluate_qei", "semiclassical_residual"]
