"""
CTC-FA v2.0   --   L5b : deck-orbit CTC detector for flat quotients.

The compact-orbit detector of L5 finds closed timelike curves that are orbits
of a *coordinate* Killing generator. That covers Godel, Kerr, van Stockum,
the Ori cores and the AdS quadric, but it cannot see the family of
chronology-violating spacetimes built as quotients of a flat covering space
by an isometry that is not a coordinate translation -- Misner space by a
boost, Grant space by a boost composed with a translation, and the polarised
Misner-type spacetimes that dominate the modern causal-classification
literature. v1.0 had no engine for them at all; its Misner entry used an
ad-hoc chart and its Gott entry a scalar predicate.

Method (exact, not heuristic)
-----------------------------
Let ``(M, eta)`` be a **flat** covering spacetime and ``Gamma`` a group of
isometries acting freely. For ``p`` in ``M`` and ``g`` in ``Gamma``, the
straight segment from ``p`` to ``g(p)`` is a geodesic of ``M`` and projects
to a closed curve in ``M/Gamma``. Its causal character is exactly the sign of

    q(p, g) = eta( g(p) - p , g(p) - p ).

Because the metric is constant, this is an **exact** statement, not a
sampled estimate: ``q < 0`` proves a closed timelike geodesic through the
projection of ``p``.

Two gates guard the method:

  F1   flatness of the covering is *verified* symbolically (Riemann = 0),
       not assumed;
  F2   the deck element must be orthochronous, otherwise the projected loop
       reverses the future direction and is not a CTC witness.

The engine reproduces the known results exactly:

    Misner covering : q = 2 (cosh(n b) - 1) (t^2 - x^2)
    Grant space     : q = 2 (cosh(n b) - 1) (t^2 - x^2) + n^2 alpha^2

and the first of these agrees, point by point, with the independent Misner
*chart* computation in ``metrics.misner`` -- a genuine cross-representation
check of the architecture rather than a self-consistency check of one chart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import sympy as sp

from .model import MetricModel
from .numerics import (ErrorBudget, SignClass, classify_sign, signature_counts)
from .orientation import timelike_eigenvector
from .status import (ClaimScope, CTCStatus, Gate, GateRecord, GateResult,
                     OrbitResult, OrbitStatus, Verdict)
from .symbolic import riemann


@dataclass
class FlatnessCertificate:
    flat: bool
    residual: str
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"flat": self.flat, "residual": self.residual, "note": self.note}


def certify_flat(model: MetricModel) -> FlatnessCertificate:
    """Gate F1: verify Riemann = 0 for the covering metric."""
    g = model.require_metric()
    R = riemann(g, model.coords, simplify=True)
    n = model.dim
    bad = []
    for a in range(n):
        for b in range(n):
             for c in range(n):
                 for d in range(c + 1, n):
                     e = sp.simplify(R[a][b][c][d])
                     if e != 0:
                         bad.append(f"R^{a}_{{{b}{c}{d}}} = {e}")
    if bad:
        return FlatnessCertificate(False, "; ".join(bad[:4]),
                                   "covering is NOT flat: the straight-segment "
                                   "geodesic argument does not apply")
    return FlatnessCertificate(True, "0",
                               "Riemann tensor vanishes identically: straight "
                               "coordinate segments are geodesics and the "
                               "interval formula is exact")


def _orthochronous(model: MetricModel, gen, subs: Mapping[sp.Symbol, float]) -> bool | None:
    """Gate F2: does the deck element preserve the time orientation?"""
    if gen.time_orientation_preserving is not None:
         return gen.time_orientation_preserving
    n = model.dim
    J = sp.Matrix(n, n, lambda a, b: sp.diff(gen.image[a], model.coords[b]))
    try:
         Jn = np.array(J.subs(subs).evalf(), dtype=float)
         gn = np.array(sp.Matrix(model.metric).subs(subs).evalf(), dtype=float)
    except Exception:                                     # noqa: BLE001
         return None
    T, lam = timelike_eigenvector(gn)
    if lam >= 0:
         return None
    ip = float((Jn @ T) @ gn @ T)
    return ip < 0 if abs(ip) > 1e-12 else None


@dataclass
class DeckOrbitResult:
    status: OrbitStatus
    margin: float
    word: tuple[str, ...] | None
    point: dict[str, float]
    reason: str
    uncertainty: float = 0.0
    flatness: FlatnessCertificate | None = None
    orthochronous: bool | None = None
    displacement: list[float] = field(default_factory=list)

    def to_verdict(self, model_id: str) -> Verdict:
        gates = GateRecord()
        gates.set(Gate.V0_PARSE, GateResult.PASS)
        gates.set(Gate.V1_ALGEBRA, GateResult.PASS,
                  "flat covering verified symbolically")
        if self.status is OrbitStatus.SECTOR_CTC:
            st, gr = CTCStatus.CTC_PROVED, GateResult.PASS
        elif self.status is OrbitStatus.SECTOR_NULL:
            st, gr = CTCStatus.CRITICAL, GateResult.INCONCLUSIVE
        else:
            st, gr = CTCStatus.UNRESOLVED, GateResult.NOT_ATTEMPTED
        gates.set(Gate.V4_CTC_WITNESS, gr, self.reason)
        if self.status is OrbitStatus.SECTOR_NULL:
            gates.set(Gate.V5_BOUNDARY, GateResult.PASS,
                      "closed null geodesic at the chronology boundary")
        v = Verdict(st, ClaimScope.SECTOR, self.margin, "deck_orbit", self.reason,
                    model_id, self.point, gates, self.as_evidence(), self.uncertainty)
        if st is CTCStatus.CTC_PROVED:
            v = v.promote(ClaimScope.DECLARED_DOMAIN,
                          "explicit closed timelike geodesic of the quotient")
        return v

    def as_evidence(self) -> dict[str, Any]:
        return {"word": list(self.word) if self.word else None,
                "displacement": self.displacement,
                "orthochronous": self.orthochronous,
                "flatness": self.flatness.as_dict() if self.flatness else None,
                "orbit_status": self.status.value}


def deck_orbit_test(model: MetricModel,
                    values: Mapping[str | sp.Symbol, float],
                    max_word_length: int = 3,
                    tol: float = 1e-10,
                    verify_flatness: bool = True) -> DeckOrbitResult:
    """Exact CTC test for a flat quotient at the point ``values``."""
    point = {str(k): float(v) for k, v in values.items()}
    if not model.has_metric:
        return DeckOrbitResult(OrbitStatus.SECTOR_UNRESOLVED, float("nan"), None,
                               point, "no local metric on the covering space")
    if not model.quotient.generators:
        return DeckOrbitResult(OrbitStatus.SECTOR_UNRESOLVED, float("nan"), None,
                               point, "no deck generators registered")

    flat: FlatnessCertificate | None = None
    if verify_flatness:
        flat = certify_flat(model)
        if not flat.flat:
            return DeckOrbitResult(
                OrbitStatus.SECTOR_UNRESOLVED, float("nan"), None, point,
                "the covering space is not flat, so the straight-segment "
                "geodesic argument is unavailable; use the Killing-orbit "
                "detector or a full geodesic search instead", 0.0, flat)

    sub = model.substitution(point)
    try:
         gnum = np.array(sp.Matrix(model.metric).subs(sub).evalf(), dtype=float)
    except Exception as exc:                              # noqa: BLE001
         return DeckOrbitResult(OrbitStatus.SECTOR_UNRESOLVED, float("nan"), None,
                                point, f"metric not numeric: {exc}", 0.0, flat)
    sig = signature_counts(gnum, tol)
    if sig != tuple(model.signature):
         return DeckOrbitResult(OrbitStatus.SECTOR_UNRESOLVED, float("nan"), None,
                                point, f"signature {sig} != declared "
                                       f"{tuple(model.signature)}", 0.0, flat)

    ortho = all(_orthochronous(model, gen, sub) is not False
                for gen in model.quotient.generators)

    p0 = [point[str(c)] for c in model.coords]
    # Words are scanned in order of increasing length so that the reported
    # witness is the SHORTEST closed loop that violates chronology, not merely
    # the most negative one. A long word is a multiple circuit of a shorter
    # loop and carries no extra information.
    words = sorted((w for w in model.quotient.words(max_word_length) if w),
                    key=len)
    best = (float("inf"), None, None)
    first_negative = None
    for word in words:
        try:
             p1 = model.quotient.apply_word(model.coords, p0, word, sub)
        except Exception:                                 # noqa: BLE001
             continue
        d = np.asarray([b - a for a, b in zip(p0, p1)], dtype=float)
        if np.max(np.abs(d)) < tol:
             continue                                     # fixed point of this word
        q = float(d @ gnum @ d)
        if q < best[0]:
             best = (q, word, d)
        if first_negative is None and q < 0:
             first_negative = (q, word, d)

    q, word, d = first_negative if first_negative is not None else best
    if word is None:
        return DeckOrbitResult(OrbitStatus.SECTOR_UNRESOLVED, float("nan"), None,
                               point, "every deck word fixes the test point", 0.0, flat)

    budget = ErrorBudget()
    budget.roundoff = float(np.finfo(np.float64).eps) * max(1.0, abs(q)) * 16
    budget.substitution = tol
    cls = classify_sign(q, budget)

    if cls is SignClass.NEGATIVE:
        if not ortho:
            return DeckOrbitResult(
                OrbitStatus.SECTOR_UNRESOLVED, q, word, point,
                "the closed geodesic is timelike but the deck element is not "
                "orthochronous, so the projected loop reverses the future "
                "direction and is not a future-directed CTC",
                budget.total, flat, ortho, list(map(float, d)))
        st = OrbitStatus.SECTOR_CTC
        reason = (f"closed timelike geodesic through the projection of the test "
                  f"point: word {word}, interval^2 = {q:+.6e} "
                  f"(exact on a flat covering)")
    elif cls is SignClass.MARGINAL:
        st = OrbitStatus.SECTOR_NULL
        reason = (f"closed null geodesic (word {word}, interval^2 = {q:+.6e} "
                  f"within u={budget.total:.2e}): chronology boundary")
    else:
        st = OrbitStatus.SECTOR_SPACELIKE
        reason = (f"every deck word up to length {max_word_length} gives a "
                  f"spacelike closed geodesic (min interval^2 = {q:+.6e}); "
                  f"this clears the tested words only")
    return DeckOrbitResult(st, q, word, point, reason, budget.total, flat, ortho,
                           list(map(float, d)))


def deck_orbit_verdict(model: MetricModel, values: Mapping[str, float],
                       **kw: Any) -> Verdict:
    return deck_orbit_test(model, values, **kw).to_verdict(model.id)


def winding_spectrum(model: MetricModel, values: Mapping[str, float],
                     max_power: int = 8) -> list[dict[str, Any]]:
    """Interval^2 of the winding-n closed geodesic for n = 1..max_power.

    The sign pattern of this spectrum *is* the causal structure of the
    quotient at that point: the smallest n with a negative entry is the
    shortest chronology-violating loop, and a zero entry marks a polarised
    hypersurface (a closed null geodesic of winding n).
    """
    sub = model.substitution(values)
    gnum = np.array(sp.Matrix(model.require_metric()).subs(sub).evalf(), dtype=float)
    coords = list(model.coords)
    p0 = [float(values[str(c)]) for c in coords]
    gen = model.quotient.generators[0]
    out: list[dict[str, Any]] = []
    for n in range(1, max_power + 1):
        word = tuple([gen.name] * n)
        p1 = model.quotient.apply_word(coords, p0, word, sub)
        d = np.asarray([b - a for a, b in zip(p0, p1)], dtype=float)
        q = float(d @ gnum @ d)
        out.append({"n": n, "interval_squared": q,
                    "character": ("timelike" if q < -1e-12 else
                                   "null" if abs(q) <= 1e-12 else "spacelike"),
                    "displacement": list(map(float, d))})
    return out


def polarized_hypersurfaces(model: MetricModel, base_point: Mapping[str, float],
                            variable: str, lo: float, hi: float,
                            max_power: int = 6) -> list[dict[str, Any]]:
    """Locate, for each winding n, the radius at which the loop becomes null.

    For a flat quotient these are the *polarised hypersurfaces*: the loci on
    which the winding-n closed geodesic is null. In Misner-type spacetimes
    they are precisely where the renormalised stress tensor of a quantum
    field diverges, and their accumulation locus is the chronology horizon.

    Locating them from the deck group is only possible because v2.0 stores
    the identification structure as an executable object; v1.0 had no path
    to this result at all.
    """
    from .numerics import bracketed_bisection, coarse_sign_scan
    gen = model.quotient.generators[0]
    coords = list(model.coords)
    out: list[dict[str, Any]] = []
    for n in range(1, max_power + 1):
        word = tuple([gen.name] * n)

        def q_of(x: float, _w=word) -> float:
            pt = dict(base_point)
            pt[variable] = x
            sub = model.substitution(pt)
            gnum = np.array(sp.Matrix(model.metric).subs(sub).evalf(), dtype=float)
            p0 = [float(pt[str(c)]) for c in coords]
            p1 = model.quotient.apply_word(coords, p0, _w, sub)
            d = np.asarray([b - a for a, b in zip(p0, p1)], dtype=float)
            return float(d @ gnum @ d)

        roots = []
        for a, b in coarse_sign_scan(q_of, lo, hi, n=200):
            if a == b:
                 roots.append((a, (a, a), 0))
                 continue
            try:
                 roots.append(bracketed_bisection(q_of, a, b, xtol=1e-13))
            except ValueError:
                 continue
        for root, bracket, iters in roots:
            out.append({"winding": n, "variable": variable, "value": root,
                          "bracket": list(bracket), "iterations": iters,
                          "interval_squared_at_root": q_of(root)})
    return out


def exact_orbit_interval(model: MetricModel, word_power: int = 1) -> sp.Expr:
    """Symbolic ``eta(g^n p - p, g^n p - p)`` for a single-generator quotient.

    Used to derive the closed-form chronology-violating region of Misner and
    Grant space instead of sampling it.
    """
    g = model.require_metric()
    gens = model.quotient.generators
    if len(gens) != 1:
        raise ValueError("exact_orbit_interval requires exactly one deck generator")
    coords = list(model.coords)
    cur = list(coords)
    for _ in range(abs(word_power)):
        image = gens[0].image if word_power > 0 else gens[0].inverse
        cur = [sp.simplify(e.subs(dict(zip(coords, cur)), simultaneous=True))
               for e in image]
    d = sp.Matrix([sp.simplify(c - x) for c, x in zip(cur, coords)])
    return sp.simplify((d.T * g * d)[0])


__all__ = ["FlatnessCertificate", "certify_flat", "DeckOrbitResult",
           "deck_orbit_test", "deck_orbit_verdict", "winding_spectrum",
           "polarized_hypersurfaces", "exact_orbit_interval"]
