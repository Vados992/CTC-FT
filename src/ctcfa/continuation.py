"""
CTC-FA v2.0    --   L16 : causal phase-boundary continuation.

The defect this module removes
------------------------------
v1.0 shipped

    def godel_family_margin(m, omega=1.0) -> float:
        return m**2 - 4*omega*omega

and then "recovered" the Godel-type causal boundary by running a root
finder on it, reporting ``m* = 1.9999999999999987`` as a regression result.
That is a root of a hard-coded literature answer, not a derivation from the
geometry. The v1.0 risk register names the pattern exactly:

    FM-016    "Known literature threshold encoded into detector and later
               claimed rediscovered" -> CIRCULAR VALIDATION.

v2.0 deletes the closed form from the solver path. The boundary is obtained
from the metric tensor alone and the analytic value is used **only
afterwards**, as an accuracy check that the solver never sees.
``NonCircularityError`` is raised if a caller tries to pass the analytic
answer into the search.

The reduced causal margin
-------------------------
Let ``xi`` be a compact generator and ``u`` a timelike reference direction.
Decompose ``xi`` orthogonally with respect to ``u``:

    xi_perp = xi - [ g(xi,u) / g(u,u) ] u,
    h(xi)   = g(xi_perp, xi_perp) = g(xi,xi) - g(xi,u)^2 / g(u,u) > 0,

which is strictly positive because the orthogonal complement of a timelike
vector is spacelike. The **reduced causal margin** is the dimensionless

    mu(x) = g(xi,xi) / h(xi).

It has the same sign as ``g(xi,xi)`` -- so it decides causality identically
-- but it is normalised by the geometry itself, is invariant under a
rescaling of the generator, and stays finite where the raw component
diverges. For the standard families it collapses to exactly the quantity
the literature uses:

    Godel-type               mu = 1 - (4 omega^2/m^2) tanh^2(m r / 2)
                             inf over r = 1 - 4 omega^2/m^2     -> root m = 2 omega
    van Stockum              mu = 1 - a^2 r^2                   -> root a r = 1
    spinning string          mu = 1 - 16 J^2/(alpha^2 r^2)      -> root alpha r = 4 J
    Ori 2007 core            mu = 1 - 2 mu_mass / r             -> root r = 2 mu_mass

None of those closed forms is used by the solver; they are what the solver
*produces*, and the test suite checks the agreement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import sympy as sp

from . import interval as ivmod
from .model import MetricModel
from .numerics import (ThresholdResult, bracketed_bisection, coarse_sign_scan)


class NonCircularityError(RuntimeError):
    """Raised when a known analytic answer would enter the search path."""


# --------------------------------------------------------------------------
# Reduced causal margin
# --------------------------------------------------------------------------

def reference_direction(model: MetricModel,
                        reference_index: int | None = None,
                        generator_index: int | None = None) -> sp.Matrix:
    """Choose a timelike reference direction ``u^a`` for the normalisation.

    Order of preference: an explicit ``reference_index``; the model's
    declared ``reference_timelike`` field; the first coordinate direction
    other than the generator whose diagonal metric component is manifestly
    negative. Charts such as Misner (``g_TT = 0``) and the null-adapted Ori
    cores (``g_rr = 0``) have no timelike *coordinate* direction at all,
    which is why the registry declares one explicitly for them.
    """
    g = model.require_metric()
    n = model.dim
    if reference_index is not None:
        v = [sp.S.Zero] * n
        v[reference_index] = sp.S.One
        return sp.Matrix(v)
    if model.reference_timelike is not None:
        return sp.Matrix(list(model.reference_timelike))
    for k in range(n):
        if k == generator_index:
             continue
        if sp.simplify(g[k, k]).is_negative:
             v = [sp.S.Zero] * n
             v[k] = sp.S.One
             return sp.Matrix(v)
    # Fall back to the model's own default point: symbolic negativity is often
    # undecidable (Taub-NUT's g_tt = -f has no fixed sign over the whole
    # parameter space) even though a timelike direction plainly exists on the
    # domain of interest.
    if model.default_point:
        try:
             sub = model.substitution(model.default_point)
             for k in range(n):
                  if k == generator_index:
                      continue
                  if float(sp.N(g[k, k].subs(sub))) < 0:
                      v = [sp.S.Zero] * n
                      v[k] = sp.S.One
                      return sp.Matrix(v)
        except Exception:                                  # noqa: BLE001
             pass
    raise ValueError(
        f"{model.id}: no timelike reference direction is available for the "
        f"reduced-margin normalisation (the compact generator itself may be "
        f"the only timelike direction, in which case the sector has no causal "
        f"phase boundary). Declare MetricModel.reference_timelike to override.")


def reduced_causal_margin_symbolic(model: MetricModel,
                                   generator_index: int | None = None,
                                   reference_index: int | None = None
                                   ) -> sp.Expr:
    """Symbolic ``mu = g(xi,xi) / h(xi)`` for a coordinate generator."""
    g = model.require_metric()
    if generator_index is None:
        if not model.compact:
            raise ValueError(f"{model.id}: no compact generator declared")
        generator_index = model.compact[0].index
    i = generator_index
    u = reference_direction(model, reference_index, i)
    n = model.dim
    xi = sp.Matrix([sp.S.One if k == i else sp.S.Zero for k in range(n)])
    q = sp.simplify((xi.T * g * xi)[0])
    guu = sp.simplify((u.T * g * u)[0])
    giu = sp.simplify((xi.T * g * u)[0])
    if guu == 0:
        raise ValueError(f"{model.id}: the reference direction is null; the "
                         f"orthogonal decomposition is undefined")
    h = sp.simplify(q - giu ** 2 / guu)
    return sp.simplify(q / h)


def reference_validity(model: MetricModel,
                       generator_index: int | None = None,
                       reference_index: int | None = None) -> sp.Expr:
    """``-g(u,u)``: strictly positive exactly where the normalisation is valid.

    The reduced margin acquires a spurious zero wherever the reference
    direction itself becomes null, so every boundary the continuation engine
    reports is checked against this expression. Without the check the
    charged pseudo-RN core reports a root at ``r^2 + 2 mu r - q^2 = 0``, which
    is an artefact of the reference direction and not a causal boundary at
    all -- precisely failure mode FM-001 in a new disguise.
    """
    g = model.require_metric()
    if generator_index is None and model.compact:
        generator_index = model.compact[0].index
    u = reference_direction(model, reference_index, generator_index)
    return sp.simplify(-(u.T * g * u)[0])


def reduced_causal_margin(model: MetricModel, values: Mapping[str, float],
                          generator_index: int | None = None,
                          reference_index: int | None = None) -> float:
    """Numeric reduced causal margin at a point.

    If no reference index is given, the timelike coordinate direction is used
    when one exists at the point, and otherwise the timelike eigendirection
    of the metric -- so the quantity remains defined inside an ergoregion,
    where every coordinate direction can be spacelike.
    """
    g = np.array(sp.Matrix(model.require_metric())
                 .subs(model.substitution(values)).evalf(), dtype=float)
    if generator_index is None:
        generator_index = model.compact[0].index
    i = generator_index
    q = float(g[i, i])
    if reference_index is None:
        cand = [k for k in range(model.dim) if k != i and g[k, k] < 0]
        if cand:
            u = np.zeros(model.dim)
            u[cand[0]] = 1.0
        else:
            vals, vecs = np.linalg.eigh((g + g.T) / 2.0)
            u = vecs[:, int(np.argmin(vals))]
    else:
        u = np.zeros(model.dim)
        u[reference_index] = 1.0
    xi = np.zeros(model.dim)
    xi[i] = 1.0
    guu = float(u @ g @ u)
    giu = float(xi @ g @ u)
    if guu >= 0:
        raise ValueError("reference direction is not timelike at this point")
    h = q - giu * giu / guu
    if h <= 0:
        raise ValueError("degenerate projection: generator is parallel to the "
                         "reference direction")
    return q / h


# --------------------------------------------------------------------------
# Sector infimum over a coordinate box
# --------------------------------------------------------------------------

@dataclass
class SectorInfimum:
    value: float
    certified: bool
    method: str
    box: dict[str, tuple[float, float]]
    argmin: dict[str, float] = field(default_factory=dict)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"value": self.value, "certified": self.certified,
                "method": self.method, "box": self.box, "argmin": self.argmin,
                "note": self.note}


def sector_infimum(model: MetricModel, fixed: Mapping[str, float],
                   box: Mapping[str, tuple[float, float]],
                   generator_index: int | None = None,
                   reference_index: int | None = None,
                   samples: int = 2001,
                   certify: bool = True) -> SectorInfimum:
    """Infimum of the reduced causal margin over a coordinate box.

    A dense scan gives the value; interval branch-and-bound is then used to
    *certify* the sign of the infimum. Only the certified branch may feed a
    formal causal statement.
    """
    mu = reduced_causal_margin_symbolic(model, generator_index, reference_index)
    subs = {model.symbol(k): v for k, v in fixed.items()}
    expr = sp.simplify(mu.subs(subs))
    free = sorted(expr.free_symbols, key=str)
    boxd = {model.symbol(k): (float(v[0]), float(v[1])) for k, v in box.items()}
    boxd = {k: v for k, v in boxd.items() if k in free}
    if not boxd:
        val = float(sp.N(expr))
        return SectorInfimum(val, True, "constant", {}, {},
                             "margin is independent of the box variables")

    grids = [np.linspace(v[0], v[1], samples) for v in boxd.values()]
    f = sp.lambdify(list(boxd.keys()), expr, "numpy")
    if len(grids) == 1:
        with np.errstate(all="ignore"):
            vals = np.asarray(f(grids[0]), dtype=float)
        good = np.isfinite(vals)
        if not good.any():
            return SectorInfimum(float("nan"), False, "scan",
                                 {str(k): v for k, v in boxd.items()}, {},
                                 "margin is nowhere finite on the box")
        j = int(np.nanargmin(np.where(good, vals, np.inf)))
        best, arg = float(vals[j]), {str(list(boxd)[0]): float(grids[0][j])}
    else:
        mesh = np.meshgrid(*grids, indexing="ij")
        with np.errstate(all="ignore"):
            vals = np.asarray(f(*mesh), dtype=float)
        j = np.unravel_index(int(np.nanargmin(vals)), vals.shape)
        best = float(vals[j])
        arg = {str(k): float(grids[i][j[i]]) for i, k in enumerate(boxd)}

    certified, method, note = False, "dense_scan", "scan only: not a certificate"
    if certify:
        cert = ivmod.prove_positive(expr, boxd, strict=True, max_boxes=6000)
        if cert.proved:
            certified, method = True, "interval_branch_and_bound"
            note = "certified strictly positive on the box"
        elif cert.refuted:
            certified, method = True, "interval_branch_and_bound"
            note = "certified to become negative on the box"
    return SectorInfimum(best, certified, method,
                         {str(k): v for k, v in boxd.items()}, arg, note)


# --------------------------------------------------------------------------
# Phase boundary in a parameter
# --------------------------------------------------------------------------

def causal_phase_boundary(model: MetricModel,
                          parameter: str,
                          bracket: tuple[float, float],
                          fixed: Mapping[str, float],
                          box: Mapping[str, tuple[float, float]],
                          generator_index: int | None = None,
                          reference_index: int | None = None,
                          analytic_value: float | None = None,
                          analytic_form: str = "",
                          xtol: float = 1e-13,
                          samples: int = 4001) -> ThresholdResult:
    """Find the parameter value at which the sector infimum changes sign.

    The search evaluates only the metric. ``analytic_value`` is compared
    with the result afterwards and is *never* used to steer the search; a
    caller that tries to make the margin function itself depend on it gets a
    :class:`NonCircularityError`.
    """
    if analytic_value is not None and analytic_form and \
            parameter in analytic_form and "margin" in analytic_form.lower():
        raise NonCircularityError(
            "the analytic form appears to be wired into the margin function; "
            "the boundary must be derived from the metric alone")

    # Lambdify the reduced margin ONCE. Re-simplifying inside the root
    # finder turns a millisecond scan into a minute-long symbolic session and
    # was the single biggest cost in the first v2.0 prototype.
    mu = reduced_causal_margin_symbolic(model, generator_index, reference_index)
    box_syms = [model.symbol(k) for k in box]
    par_sym = model.symbol(parameter)
    fixed_syms = [model.symbol(k) for k in fixed if k != parameter]
    free = mu.free_symbols
    unresolved = free - set(box_syms) - {par_sym} - set(fixed_syms)
    if unresolved:
        raise ValueError(f"{model.id}: reduced margin still depends on "
                         f"{sorted(map(str, unresolved))}; give them values in "
                         f"'fixed' or ranges in 'box'")
    f = sp.lambdify(box_syms + [par_sym] + fixed_syms, mu, "numpy")
    grids = [np.linspace(box[str(s)][0], box[str(s)][1], samples) for s in box_syms]
    fixed_vals = [float(fixed[str(s)]) for s in fixed_syms]
    if len(grids) > 1:
        mesh = np.meshgrid(*grids, indexing="ij")
    else:
        mesh = grids

    def margin_of(p: float) -> float:
        with np.errstate(all="ignore"):
            if not box_syms:
                v = float(f(float(p), *fixed_vals))
                return v
            vals = np.asarray(f(*mesh, float(p), *fixed_vals), dtype=float)
        vals = np.where(np.isfinite(vals), vals, np.inf)
        return float(np.min(vals))

    lo, hi = float(bracket[0]), float(bracket[1])
    brackets = coarse_sign_scan(margin_of, lo, hi, n=48)
    if not brackets:
        raise ValueError(f"{model.id}: no sign change of the sector infimum for "
                         f"{parameter} in [{lo}, {hi}]")
    a, b = brackets[0]
    root, final_bracket, iters = bracketed_bisection(margin_of, a, b, xtol=xtol)

    eps = max(1e-9, 10 * xtol)
    left = margin_of(root - eps)
    right = margin_of(root + eps)

    fx = dict(fixed)
    fx[parameter] = root
    dom = model.domain_report({**fx, **{k: 0.5 * (v[0] + v[1])
                                        for k, v in box.items()}}) \
        if model.domain else {"ok": True}

    # Reject a root that is an artefact of the normalisation rather than a
    # causal transition (see reference_validity).
    ref_ok, ref_note = True, ""
    try:
         val = reference_validity(model, generator_index, reference_index)
         # (a) pointwise at the root: a root that sits where -g(u,u) vanishes is
         #     an artefact of the normalisation, not a causal transition.
         at_root = val.subs({model.symbol(k): v for k, v in fx.items()})
         rest = {model.symbol(k): 0.5 * (v[0] + v[1]) for k, v in box.items()}
         at_root = at_root.subs(rest)
         if not at_root.free_symbols:
             v0 = float(sp.N(at_root))
             if v0 <= 1e-9:
                 ref_ok = False
                 ref_note = (f" | REJECTED: the timelike reference direction is "
                              f"null at this root (-g(u,u) = {v0:.3e}), so the sign "
                              f"change is a normalisation artefact, not a causal "
                              f"boundary")
         # (b) over the whole box, when one was given.
         if ref_ok and box:
             cert = ivmod.prove_positive(
                 val.subs({model.symbol(k): v for k, v in fx.items()}),
                 {model.symbol(k): (float(v[0]), float(v[1]))
                  for k, v in box.items()},
                 strict=True, max_boxes=4000)
             if cert.refuted:
                 ref_ok = False
                 ref_note = (" | WARNING: the timelike reference direction becomes "
                              "null somewhere on this box, so the reported root may "
                             "be a normalisation artefact")
    except Exception as exc:                              # noqa: BLE001
        ref_note = f" | reference-validity check unavailable: {exc}"

    return ThresholdResult(
        value=root, bracket=final_bracket, left_margin=left, right_margin=right,
        residual=abs(margin_of(root)), iterations=iters,
        method="reduced-causal-margin infimum + bracketed bisection",
        analytic_value=analytic_value, analytic_form=analytic_form,
        domain_ok=bool(dom.get("ok", True)) and ref_ok,
        note=("boundary derived from the metric tensor only; the analytic value "
              "is reported for comparison and was not used in the search")
             + ref_note)


def convergence_study(model: MetricModel, parameter: str,
                      bracket: tuple[float, float], fixed: Mapping[str, float],
                      box_variable: str, radii: Sequence[float],
                      generator_index: int | None = None,
                      analytic_value: float | None = None) -> list[dict[str, Any]]:
    """Boundary value as the coordinate box is enlarged.

    For families whose exact boundary is an asymptotic statement -- the
    Godel-type critical radius diverges as ``m -> 2 omega`` -- the boundary
    at finite box radius is only an approximation, and the *rate* at which it
    converges is itself evidence. Reporting this table instead of a single
    number is what distinguishes a derivation from a lucky root.
    """
    rows = []
    for Rmax in radii:
        box = {box_variable: (1e-6, float(Rmax))}
        try:
             th = causal_phase_boundary(model, parameter, bracket, fixed, box,
                                        generator_index,
                                        analytic_value=analytic_value)
        except ValueError as exc:
             rows.append({"box_radius": float(Rmax), "error": str(exc)})
             continue
        rows.append({"box_radius": float(Rmax), "boundary": th.value,
                      "bracket_width": th.bracket_width,
                      "analytic": analytic_value,
                      "error": (None if analytic_value is None
                                else abs(th.value - analytic_value))})
    return rows


__all__ = ["NonCircularityError", "reduced_causal_margin_symbolic",
           "reduced_causal_margin", "SectorInfimum", "sector_infimum",
           "causal_phase_boundary", "convergence_study"]
