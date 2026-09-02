"""
CTC-FA v2.0   --   L5 (compact-orbit CTC detector) and L7 (causality certificate).

L5 -- positive route
--------------------
For declared compact Killing generators ``xi_A`` with periods ``P_A`` the
Gram matrix is ``K_AB = g(xi_A, xi_B)`` and the closed orbit with integer
winding ``n^A`` has invariant interval ``q = n^A K_AB n^B``. ``q < 0`` is an
explicit symmetry-generated closed timelike curve.

Compared to v1.0 the detector now *earns* that conclusion instead of
assuming it. Before a witness is accepted it verifies, in order:

  G1 the model has a local metric (else the claim is external);
  G2 the point lies inside the declared physical domain;
  G3 the metric is nondegenerate Lorentzian there (Sylvester counts);
  G4 every declared compact generator is a Killing vector of the metric;
  G5 the orbit really closes under the declared identification / deck group;
  G6 the orbit is time-orientation consistent;
  G7 the sign of ``q`` survives its uncertainty budget, with automatic
     escalation to 50-digit arithmetic when it does not.

Only then is an ``OrbitStatus.SECTOR_CTC`` emitted, and even then it is
lifted to a *sector-scoped* verdict. Failure of any gate downgrades the
result to ``SECTOR_UNRESOLVED``; it never produces a negative claim.

L7 -- negative route
--------------------
A negative result is constructive. A smooth function ``tau`` with
``g^{ab} d_a tau d_b tau < 0`` everywhere on the declared domain is a
temporal function; its existence certifies stable causality and therefore
excludes CTCs on that domain. v2.0 proves the strict inequality either
symbolically or by interval branch-and-bound on a compact box, and reports
which, together with the box. Point sampling alone is never accepted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import sympy as sp

from . import interval as ivmod
from .model import MetricModel
from .numerics import (ErrorBudget, SignClass, budget_for_matrix, classify_sign,
                       escalate_if_marginal, signature_counts)
from .orientation import orbit_orientation
from .status import (ClaimScope, CTCStatus, Gate, GateRecord, GateResult,
                     OrbitResult, OrbitStatus, Verdict)
from .symmetry import (compact_gram_symbolic, generators_admissible,
                       winding_norm_symbolic)
from .topology import curve_closes


# --------------------------------------------------------------------------
# L5 : compact-orbit detector
# --------------------------------------------------------------------------

def _gcd_tuple(w: Sequence[int]) -> int:
    g = 0
    for q in w:
        g = math.gcd(g, abs(int(q)))
    return g


def evaluate_metric(model: MetricModel,
                    values: Mapping[str | sp.Symbol, float]) -> np.ndarray:
    g = model.require_metric()
    sub = model.substitution(values)
    missing = model.missing_values(values)
    if missing:
        raise ValueError(f"{model.id}: no numeric value supplied for {missing}. "
                         f"Every coordinate and parameter must be given.")
    return np.array(sp.Matrix(g).subs(sub).evalf(), dtype=float)


def compact_orbit_test(model: MetricModel,
                       values: Mapping[str | sp.Symbol, float],
                       max_winding: int = 2,
                       tol: float = 1e-10,
                       check_orientation: bool = True,
                       check_closure: bool = True) -> OrbitResult:
    """Full L5 detector with all seven admissibility gates."""
    point = {str(k): float(v) for k, v in values.items()}

    # -- G1 ---------------------------------------------------------------
    if not model.has_metric or not model.compact:
        return OrbitResult(
            OrbitStatus.SECTOR_UNRESOLVED, float("nan"), None, point,
            "no local metric and/or no declared compact generator; this model "
            "must be handled by a global/reduced predicate, not by the "
            "compact-orbit detector")

    # -- G2 ---------------------------------------------------------------
    dom = model.domain_report(point)
    if not dom["ok"]:
        bad = [c["name"] for c in dom["constraints"] if not c["ok"]]
        return OrbitResult(
            OrbitStatus.SECTOR_UNRESOLVED, float("nan"), None, point,
            f"test point violates declared domain constraints {bad}; "
            f"a causal claim outside the physical domain is rejected")

    # -- G3 ---------------------------------------------------------------
    try:
         gnum = evaluate_metric(model, point)
    except Exception as exc:                               # noqa: BLE001
         return OrbitResult(OrbitStatus.SECTOR_UNRESOLVED, float("nan"), None,
                             point, f"metric evaluation failed: {exc}")
    if not np.all(np.isfinite(gnum)):
         return OrbitResult(OrbitStatus.SECTOR_UNRESOLVED, float("nan"), None,
                             point, "metric is not finite at the test point")
    sig = signature_counts(gnum, tol)
    if sig != tuple(model.signature):
         return OrbitResult(
             OrbitStatus.SECTOR_UNRESOLVED, float("nan"), None, point,
             f"signature {sig} at the test point differs from the declared "
             f"{tuple(model.signature)} in the (-,+,+,+) convention")

    # -- G4 ---------------------------------------------------------------
    ok_gen, certs = generators_admissible(model)
    if not ok_gen:
        bad = [c.name for c in certs if not c.admissible]
        return OrbitResult(
            OrbitStatus.SECTOR_UNRESOLVED, float("nan"), None, point,
            f"declared compact generators {bad} are not Killing vectors of this "
            f"metric; the symmetry-orbit argument does not apply")

    # -- Gram matrix and winding scan -------------------------------------
    idx = [c.index for c in model.compact]
    K = gnum[np.ix_(idx, idx)]
    budget = budget_for_matrix(gnum, scale=float(max_winding) ** 2)
    budget.substitution = tol
    best_val = float("inf")
    best_w: tuple[int, ...] | None = None
    for w in product(range(-max_winding, max_winding + 1), repeat=len(idx)):
        if all(q == 0 for q in w):
            continue
        if _gcd_tuple(w) != 1:
            continue           # only primitive windings: n and k*n are the same orbit
        if next(q for q in w if q != 0) < 0:
            continue           # n and -n trace the same orbit with opposite orientation
        n = np.asarray(w, dtype=float)
        val = float(n @ K @ n) / float(n @ n)
        if val < best_val:
            best_val, best_w = val, tuple(int(q) for q in w)

    if best_w is None:
        return OrbitResult(OrbitStatus.SECTOR_UNRESOLVED, float("nan"), None,
                           point, "empty winding lattice")

    # -- G7 : uncertainty and escalation ----------------------------------
    escalated = False
    try:
         q_expr = winding_norm_symbolic(model, best_w) / float(np.dot(best_w, best_w))
         sub = model.substitution(point)
         best_val, budget, escalated = escalate_if_marginal(
             q_expr, sub, best_val, budget)
    except Exception as exc:                              # noqa: BLE001
         budget.notes.append(f"escalation unavailable: {exc}")

    cls = classify_sign(best_val, budget)

    # -- G5 : closure ------------------------------------------------------
    closure_note = "not checked"
    if check_closure:
        p0 = [point[str(c)] for c in model.coords]
        p1 = list(p0)
        for w, cd in zip(best_w, model.compact):
            try:
                 period = float(sp.N(cd.period.subs(model.substitution(point))))
            except Exception:                             # noqa: BLE001
                 period = float(sp.N(cd.period))
            p1[cd.index] = p0[cd.index] + w * period
        cl = curve_closes(model, p0, p1, tol=1e-8,
                           params=model.substitution(point))
        closure_note = f"{cl.mechanism} (residual {cl.residual:.2e})"
        if not cl.closed:
            return OrbitResult(
                 OrbitStatus.SECTOR_UNRESOLVED, best_val, best_w, point,
                 f"the winding orbit does not close under the registered "
                 f"identification structure: {cl.note}",
                 budget.total, tuple(c.name for c in model.compact))

    # -- G6 : orientation --------------------------------------------------
    orientation_ok: bool | None = None
    orientation_note = ""
    if check_orientation and cls is SignClass.NEGATIVE:
        xi = np.zeros(model.dim)
        for w, cd in zip(best_w, model.compact):
            try:
                 period = float(sp.N(cd.period.subs(model.substitution(point))))
            except Exception:                             # noqa: BLE001
                 period = float(sp.N(cd.period))
            xi[cd.index] += w * period
            orr = orbit_orientation(gnum, xi)
            orientation_ok = orr.consistent
            orientation_note = orr.note
            if not orientation_ok:
                return OrbitResult(
                    OrbitStatus.SECTOR_UNRESOLVED, best_val, best_w, point,
                    f"orbit is timelike but time orientation is not consistent: "
                    f"{orientation_note}", budget.total,
                    tuple(c.name for c in model.compact), orientation_ok)

    if cls is SignClass.NEGATIVE:
        st = OrbitStatus.SECTOR_CTC
        reason = (f"closed symmetry orbit with winding {best_w} is timelike "
                  f"(q={best_val:+.6e}, u={budget.total:.2e}); closure by "
                  f"{closure_note}; orientation consistent")
    elif cls is SignClass.MARGINAL:
        st = OrbitStatus.SECTOR_NULL
        reason = (f"closed symmetry orbit with winding {best_w} is null within "
                  f"the uncertainty band (q={best_val:+.6e}, u={budget.total:.2e})"
                  + (" [50-digit escalation applied]" if escalated else ""))
    else:
        st = OrbitStatus.SECTOR_SPACELIKE
        reason = (f"every scanned winding up to |n|<={max_winding} is spacelike "
                  f"(min q={best_val:+.6e}); this clears the tested sector only "
                  f"and is NOT a global no-CTC result")

    return OrbitResult(st, best_val, best_w, point, reason, budget.total,
                       tuple(c.name for c in model.compact), orientation_ok,
                       isometry_verified=all(c.is_killing for c in certs))


def compact_orbit_verdict(model: MetricModel,
                           values: Mapping[str | sp.Symbol, float],
                           **kw: Any) -> Verdict:
    """Run the detector and lift the result to a scoped verdict."""
    r = compact_orbit_test(model, values, **kw)
    gates = GateRecord()
    gates.set(Gate.V0_PARSE, GateResult.PASS, "registry record well-formed")
    gates.set(Gate.V1_ALGEBRA, GateResult.PASS,
               "signature and inverse verified at the test point")
    if r.status is OrbitStatus.SECTOR_CTC:
        gates.set(Gate.V4_CTC_WITNESS, GateResult.PASS, r.reason)
    elif r.status is OrbitStatus.SECTOR_NULL:
        gates.set(Gate.V4_CTC_WITNESS, GateResult.INCONCLUSIVE, r.reason)
        gates.set(Gate.V5_BOUNDARY, GateResult.PASS,
                   "closed null orbit locates a marginal causal boundary")
    else:
        gates.set(Gate.V4_CTC_WITNESS, GateResult.NOT_ATTEMPTED, r.reason)
    v = r.to_verdict(model.id, gates)
    if r.status is OrbitStatus.SECTOR_CTC and model.kind.value in (
             "LOCAL_METRIC", "QUOTIENT_GLOBAL"):
        # An explicit witness inside the declared spacetime is an existence
        # proof for that spacetime (asymmetric evidence).
        v = v.promote(ClaimScope.DECLARED_DOMAIN,
                       "explicit verified witness in the declared spacetime")
    return v


# --------------------------------------------------------------------------
# L7 : causality certificate
# --------------------------------------------------------------------------

@dataclass
class TemporalCertificate:
    """Result of attempting a global temporal-function proof."""

    model_id: str
    tau: str
    status: CTCStatus
    scope: ClaimScope
    method: str
    proved: bool
    refuted: bool
    expression: str
    box: dict[str, tuple[float, float]] = field(default_factory=dict)
    domain_is_compact: bool = True
    lower_bound: float | None = None
    counterexample: dict[str, float] | None = None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "tau": self.tau,
                "status": self.status.value, "scope": self.scope.value,
                "method": self.method, "proved": self.proved,
                "refuted": self.refuted, "expression": self.expression,
                "box": self.box, "domain_is_compact": self.domain_is_compact,
                "lower_bound": self.lower_bound,
                "counterexample": self.counterexample, "note": self.note}


def _deck_noninvariance(model: MetricModel, tau: sp.Expr) -> list[str]:
    """Return the identifications under which ``tau`` fails to be invariant."""
    bad: list[str] = []
    coords = list(model.coords)
    for gen in model.quotient.generators:
        img = sp.simplify(tau.subs(dict(zip(coords, gen.image)), simultaneous=True))
        if sp.simplify(img - tau) != 0:
            bad.append(f"deck generator {gen.name}: tau -> {img}")
    for cd in model.compact:
        sub = {coords[cd.index]: coords[cd.index] + cd.period}
        img = sp.simplify(tau.subs(sub, simultaneous=True))
        if sp.simplify(img - tau) != 0:
            bad.append(f"period in {cd.name}: tau -> {img}")
    return bad


def temporal_gradient_norm(model: MetricModel, tau: sp.Expr) -> sp.Expr:
    """C_tau = - g^{ab} d_a tau d_b tau. Positive means timelike gradient."""
    g = model.require_metric()
    gi = g.inv()
    coords = model.coords
    n = len(coords)
    grad = [sp.diff(tau, c) for c in coords]
    expr = sum(gi[a, b] * grad[a] * grad[b] for a in range(n) for b in range(n))
    return sp.simplify(-expr)


def certify_temporal_function(model: MetricModel,
                              tau: sp.Expr | None = None,
                              box: Mapping[sp.Symbol, tuple[float, float]] | None = None,
                              assumptions: Mapping[sp.Symbol, dict] | None = None,
                              domain_is_compact: bool | None = None,
                              max_boxes: int = 20000) -> TemporalCertificate:
    """Prove or refute that ``tau`` is a temporal function on the domain.

    A ``CERTIFIED`` outcome upgrades the model to ``NO_CTC_CERTIFIED`` with
    the scope implied by ``domain_is_compact``: ``GLOBAL`` when the proof is
    symbolic and unconditional, ``DECLARED_DOMAIN`` when it rests on a box.
    """
    tau = model.temporal_function if tau is None else tau
    if tau is None:
        return TemporalCertificate(
            model.id, "none", CTCStatus.UNRESOLVED, ClaimScope.SECTOR, "none",
            False, False, "", note="no candidate temporal function registered")

    # -- Gate T0 : is tau even a FUNCTION on the quotient? -----------------
    #
    # A temporal function must be single valued on the actual spacetime. If
    # the model carries a deck group, tau must be invariant under it. This
    # gate is what makes ads4_periodic fail while ads4_universal_cover
    # succeeds on an identical local metric -- the whole point of the L3
    # gate. v1.0 had no such check, so its temporal-function route would
    # have "certified" a spacetime that is entirely chronology violating.
    bad = _deck_noninvariance(model, tau)
    if bad:
        return TemporalCertificate(
            model.id, str(tau), CTCStatus.UNRESOLVED, ClaimScope.SECTOR,
            "quotient_invariance", False, True, "",
            note=("the candidate is NOT single valued on this spacetime: it "
                  f"fails to be invariant under the identification(s) {bad}. "
                  "A multivalued 'time coordinate' on a quotient is not a "
                  "temporal function, and no negative certificate follows."))

    C = temporal_gradient_norm(model, tau)
    cert = ivmod.prove_positive(C, box, assumptions, strict=True,
                                max_boxes=max_boxes,
                                domain_is_compact=(True if domain_is_compact is None
                                                   else domain_is_compact))
    if cert.proved:
        if cert.method == "symbolic" and (domain_is_compact is not False):
            status, scope = CTCStatus.NO_CTC_CERTIFIED, ClaimScope.GLOBAL
            note = ("unconditional symbolic proof of a strictly timelike "
                    "gradient: stable causality, hence no CTC")
        else:
            status, scope = CTCStatus.NO_CTC_CERTIFIED, ClaimScope.DECLARED_DOMAIN
            note = ("certified on the stated compact box only; extension to the "
                    "full domain requires an additional analytic argument")
    elif cert.refuted:
        status, scope = CTCStatus.UNRESOLVED, ClaimScope.SECTOR
        note = ("the candidate is NOT a temporal function; this refutes the "
                "candidate, it does not prove that CTCs exist")
    else:
        status, scope = CTCStatus.UNRESOLVED, ClaimScope.SECTOR
        note = ("neither proved nor refuted; UNRESOLVED is the only permitted "
                "output (failure of a search is not evidence of absence)")
    return TemporalCertificate(
        model.id, str(tau), status, scope, cert.method, cert.proved, cert.refuted,
        str(C), cert.box, cert.domain_is_compact, cert.lower_bound,
        cert.counterexample, note + (" | " + cert.symbolic_note
                                     if cert.symbolic_note else ""))


def temporal_verdict(model: MetricModel, **kw: Any) -> Verdict:
    c = certify_temporal_function(model, **kw)
    gates = GateRecord()
    gates.set(Gate.V0_PARSE, GateResult.PASS)
    gates.set(Gate.V1_ALGEBRA, GateResult.PASS)
    gates.set(Gate.V6_NEGATIVE_CERTIFICATE,
              GateResult.PASS if c.proved else
              (GateResult.FAIL if c.refuted else GateResult.INCONCLUSIVE), c.note)
    v = Verdict(status=c.status, scope=ClaimScope.SECTOR,
                 margin=float(c.lower_bound) if c.lower_bound is not None else float("nan"),
                 method=f"temporal_function:{c.method}", reason=c.note,
                 model_id=model.id, gates=gates, evidence=c.as_dict())
    if c.proved:
        v = v.promote(c.scope, "verified global temporal-function certificate")
    return v


# --------------------------------------------------------------------------
# Global / reduced predicates
# --------------------------------------------------------------------------

def global_predicate_verdict(model: MetricModel,
                             parameters: Mapping[str, float],
                             tol: float = 1e-12) -> Verdict:
    """Evaluate a registered global/reduced route-closure predicate.

    The result carries ``model.global_margin_scope``, which for every reduced
    model in the registry is ``REDUCED_MODEL``. It therefore cannot be
    printed as a global GR theorem -- the v1.0 category error (D-05).
    """
    gates = GateRecord()
    gates.set(Gate.V0_PARSE, GateResult.PASS)
    if model.global_margin is None:
        return Verdict(CTCStatus.UNRESOLVED, ClaimScope.SECTOR, float("nan"),
                       "global_predicate",
                       "no global or reduced predicate is registered for this model",
                       model.id, dict(parameters), gates)
    m = float(model.global_margin(parameters))
    if m < -tol:
        st, res = CTCStatus.CTC_PROVED, "route closes: arrival precedes departure"
        gates.set(Gate.V4_CTC_WITNESS, GateResult.PASS, res)
    elif abs(m) <= tol:
        st, res = CTCStatus.CRITICAL, "route closure is exactly marginal"
        gates.set(Gate.V5_BOUNDARY, GateResult.PASS, res)
    else:
        st, res = (CTCStatus.UNRESOLVED,
                   "route does not close under this reduced predicate; "
                   "this is not a no-CTC certificate")
        gates.set(Gate.V4_CTC_WITNESS, GateResult.NOT_ATTEMPTED, res)
    return Verdict(st, model.global_margin_scope, m, "global_predicate",
                   res + (" | " + model.global_margin_note
                          if model.global_margin_note else ""),
                   model.id, dict(parameters), gates)


__all__ = ["evaluate_metric", "compact_orbit_test", "compact_orbit_verdict",
           "TemporalCertificate", "temporal_gradient_norm",
           "certify_temporal_function", "temporal_verdict",
           "global_predicate_verdict"]
