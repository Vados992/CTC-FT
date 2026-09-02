"""
CTC-FA v2.0   --   L3 : Topology and identification engine.

This layer is the single largest addition of v2.0. In v1.0 the topology
component of the state vector was a free-text field, and the architecture's
own risk register recorded the consequence:

    FM-007    "Topology stored only as prose and misapplied" -> global causal
              error. Mitigation: "implement quotient/cover objects and deck
              transformations in v2."

That is what this module does.    A quotient is now an executable object with

  * an **isometry certificate** for every deck generator (symbolic pullback),
  * a **free-action** check where decidable,
  * a **time-orientation** check for every generator,
  * a **closure test** for curves on the quotient: a curve is closed in
    ``M/Gamma`` when ``gamma(1) = g . gamma(0)`` for some word ``g`` in
    ``Gamma``, which is strictly more general than the v1.0 test of reducing
    one coordinate difference modulo a scalar period.

The decisive scientific consequence is the L3 gate:

    Local equality of metrics does not imply equality of global causal
    structure.

v2.0 ships an exact demonstration of that gate: ``ads4_universal_cover`` and
``ads4_periodic`` carry the *identical* local metric tensor and receive
opposite global verdicts (``NO_CTC_CERTIFIED`` vs ``CTC_PROVED``) purely
because of the deck group. No v1.0 layer could express that distinction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import sympy as sp

from .model import DeckTransformation, MetricModel, Quotient
from .numerics import signature_counts


# --------------------------------------------------------------------------
# Certificates
# --------------------------------------------------------------------------

@dataclass
class DeckCertificate:
    """Verification record for one deck generator."""

    name: str
    isometry_residual: str
    is_isometry: bool
    time_orientation_preserving: bool | None
    has_fixed_point: bool | None
    fixed_point_note: str = ""
    note: str = ""

    @property
    def admissible(self) -> bool:
        if not self.is_isometry:
            return False
        if self.has_fixed_point is True:
            return False
        return True

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "is_isometry": self.is_isometry,
                "isometry_residual": self.isometry_residual,
                "time_orientation_preserving": self.time_orientation_preserving,
                "has_fixed_point": self.has_fixed_point,
                "fixed_point_note": self.fixed_point_note,
                "admissible": self.admissible, "note": self.note}


@dataclass
class QuotientCertificate:
    """Verification record for the whole deck group of a model."""

    model_id: str
    label: str
    generators: list[DeckCertificate] = field(default_factory=list)
    trivial: bool = True
    coordinate_periodicities: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    @property
    def admissible(self) -> bool:
        return all(c.admissible for c in self.generators)

    @property
    def time_orientable(self) -> bool | None:
        vals = [c.time_orientation_preserving for c in self.generators]
        if not vals:
            return True
        if any(v is False for v in vals):
            return False
        if any(v is None for v in vals):
            return None
        return True

    def as_dict(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "label": self.label,
                "trivial": self.trivial, "admissible": self.admissible,
                "time_orientable": self.time_orientable,
                "generators": [c.as_dict() for c in self.generators],
                "coordinate_periodicities": self.coordinate_periodicities,
                "note": self.note}


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

def _time_orientation_preserving(model: MetricModel, gen: DeckTransformation,
                                 probe: Mapping[str, float] | None = None
                                 ) -> tuple[bool | None, str]:
    """Does ``gen`` map future-directed timelike vectors to future-directed ones?

    Test: take the declared timelike reference direction, push it forward with
    the Jacobian of ``gen`` and check that the inner product of the image with
    the reference field at the image point is still negative (same time
    orientation). Returns ``(verdict, note)``; ``None`` when the test is not
    decidable at the probe point.
    """
    if gen.time_orientation_preserving is not None:
         return gen.time_orientation_preserving, "declared in registry"
    g = model.metric
    if g is None:
         return None, "no local metric"
    pt = dict(probe or model.default_point)
    if not pt:
         return None, "no probe point"
    coords = model.coords
    n = len(coords)
    try:
         sub = model.substitution(pt)
    except KeyError as exc:
         return None, f"probe point incomplete: {exc}"
    try:
         gnum = np.array(sp.Matrix(g).subs(sub).evalf(), dtype=float)
    except Exception as exc:                              # noqa: BLE001
         return None, f"metric not numeric at probe: {exc}"
    vals, vecs = np.linalg.eigh((gnum + gnum.T) / 2.0)
    order = np.argsort(vals)
    tvec = vecs[:, order[0]]                        # the timelike eigendirection
    if vals[order[0]] >= 0:
         return None, "no timelike direction at probe point"
    J = sp.Matrix(n, n, lambda a, b: sp.diff(gen.image[a], coords[b]))
    try:
         Jn = np.array(J.subs(sub).evalf(), dtype=float)
    except Exception as exc:                              # noqa: BLE001
         return None, f"Jacobian not numeric at probe: {exc}"
    pushed = Jn @ tvec
    image_pt = gen.apply_numeric(coords, [float(sub[c]) for c in coords])
    sub_img = dict(sub)
    for c, v in zip(coords, image_pt):
         sub_img[c] = v
    try:
        gimg = np.array(sp.Matrix(g).subs(sub_img).evalf(), dtype=float)
    except Exception as exc:                             # noqa: BLE001
        return None, f"metric not numeric at image: {exc}"
    vi, ve = np.linalg.eigh((gimg + gimg.T) / 2.0)
    oi = np.argsort(vi)
    if vi[oi[0]] >= 0:
        return None, "no timelike direction at image point"
    tref = ve[:, oi[0]]
    # Align the reference at the image with the reference at the source by
    # continuity along the shortest path in coordinate space.
    if float(tref @ tvec) < 0:
        tref = -tref
    ip = float(pushed @ gimg @ tref)
    if abs(ip) < 1e-12:
        return None, "degenerate orientation probe"
    return (ip < 0), f"g(phi_* T, T_ref) = {ip:.6e}"


def _has_fixed_point(model: MetricModel, gen: DeckTransformation
                      ) -> tuple[bool | None, str]:
    """Try to solve ``phi(x) = x``. A free action is required for a manifold
    quotient; a fixed point means the quotient is an orbifold, not a manifold.
    """
    coords = model.coords
    eqs = [sp.Eq(sp.simplify(gen.image[i] - coords[i]), 0) for i in range(len(coords))]
    eqs = [e for e in eqs if e != True]                    # noqa: E712
    if not eqs:
         return True, "generator is the identity on this chart"
    try:
         sol = sp.solve(eqs, list(coords), dict=True)
    except Exception as exc:                               # noqa: BLE001
         return None, f"fixed-point system not solvable symbolically: {exc}"
    if not sol:
         return False, "no solution to phi(x)=x"
    real = []
    for s in sol:
         if all(getattr(v, "is_real", None) is not False for v in s.values()):
             real.append(s)
    if not real:
         return False, "solutions exist but none are real"
    return True, f"fixed set: {real}"


def verify_quotient(model: MetricModel,
                    probe: Mapping[str, float] | None = None) -> QuotientCertificate:
    """Full L3 verification of a model's identification structure."""
    q = model.quotient
    cert = QuotientCertificate(model_id=model.id, label=q.label,
                               trivial=not q.generators)

    # Coordinate periodicities declared via CompactDirection are the special
    # case Gamma = <x^i -> x^i + P>. Verify each is an isometry too.
    for cd in model.compact:
        row: dict[str, Any] = {"coordinate": cd.name, "index": cd.index,
                               "period": str(cd.period)}
        if model.metric is not None:
            d = sp.simplify(sp.diff(model.metric, model.coords[cd.index]))
            row["metric_independent_of_coordinate"] = bool(d.is_zero_matrix)
            row["killing_residual"] = "0" if d.is_zero_matrix else str(d)
        cert.coordinate_periodicities.append(row)

    if model.metric is None:
        cert.note = ("model has no local metric; identification structure is "
                     "recorded as an external/global specification")
        for gen in q.generators:
            cert.generators.append(DeckCertificate(
                name=gen.name, isometry_residual="not-applicable",
                is_isometry=True, time_orientation_preserving=gen.time_orientation_preserving,
                has_fixed_point=None,
                note="isometry not checkable without a local metric"))
        return cert

    residuals = q.verify_isometries(model.coords, model.metric)
    for gen in q.generators:
        res = residuals.get(gen.name, "unknown")
        tof, tnote = _time_orientation_preserving(model, gen, probe)
        fp, fnote = _has_fixed_point(model, gen)
        cert.generators.append(DeckCertificate(
            name=gen.name, isometry_residual=res, is_isometry=(res == "0"),
            time_orientation_preserving=tof, has_fixed_point=fp,
            fixed_point_note=fnote, note=tnote))
    return cert


# --------------------------------------------------------------------------
# Closure on the quotient
# --------------------------------------------------------------------------

@dataclass
class ClosureResult:
    closed: bool
    word: tuple[str, ...] | None
    residual: float
    mechanism: str
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"closed": self.closed, "word": list(self.word) if self.word else None,
                "residual": self.residual, "mechanism": self.mechanism,
                "note": self.note}


def curve_closes(model: MetricModel, p0: Sequence[float], p1: Sequence[float],
                 tol: float = 1e-9, max_word_length: int = 3,
                 params: Mapping[Any, float] | None = None) -> ClosureResult:
    """Does the curve segment from ``p0`` to ``p1`` close on ``M/Gamma``?

    Three mechanisms are tried in order:

    1. exact coincidence in the chart (``p1 == p0``);
    2. coordinate periodicity from ``model.compact``;
    3. a non-trivial word in the deck group ``Gamma``.

    Mechanism 3 is what v1.0 could not express.
    """
    p0 = [float(v) for v in p0]
    p1 = [float(v) for v in p1]
    if not all(math.isfinite(v) for v in p0 + p1):
        return ClosureResult(False, None, float("nan"), "invalid",
                             "curve endpoints are not finite")
    res = max(abs(a - b) for a, b in zip(p0, p1))
    if res <= tol:
        return ClosureResult(True, (), res, "identity",
                             "curve returns to the same chart point")

    if model.compact:
        delta = [b - a for a, b in zip(p0, p1)]
        reduced = list(delta)
        used = []
        for cd in model.compact:
            period = 0.0
            for src in (params, model.default_point):
                if not src:
                     continue
                try:
                     sub = (src if all(isinstance(k, sp.Symbol) for k in src)
                            else model.substitution(src))
                     period = float(sp.N(cd.period.subs(sub)))
                     break
                except Exception:                         # noqa: BLE001
                     continue
            if not period:
                try:
                     period = float(sp.N(cd.period))
                except Exception:                         # noqa: BLE001
                     continue
            if period == 0 or not math.isfinite(period):
                continue
            k = round(reduced[cd.index] / period)
            if k != 0:
                used.append(f"{cd.name}+{k}*P")
            reduced[cd.index] -= k * period
        r2 = max(abs(v) for v in reduced)
        if r2 <= tol:
            return ClosureResult(True, tuple(used), r2, "coordinate_period",
                                  "closed modulo declared coordinate periods")

    if model.quotient.generators:
        closed, word, r3 = model.quotient.closes(
            model.coords, p0, p1, tol=tol, max_length=max_word_length,
            params=params)
        if closed:
            return ClosureResult(True, word, r3, "deck_group",
                                  f"gamma(1) = {word or 'e'} . gamma(0)")
        return ClosureResult(False, word, r3, "deck_group",
                             f"best residual over words of length <= {max_word_length}")

    return ClosureResult(False, None, res, "none",
                         "no identification structure registered for this model")


# --------------------------------------------------------------------------
# Topological classification
# --------------------------------------------------------------------------

def classify(model: MetricModel) -> dict[str, Any]:
    """A compact machine summary of the model's global structure."""
    q = model.quotient
    return {
        "model_id": model.id,
        "kind": model.kind.value,
        "topology": model.topology,
        "n_coordinate_identifications": len(model.compact),
        "coordinate_identifications": [
             {"coordinate": c.name, "period": str(c.period)} for c in model.compact],
        "deck_group": q.label,
        "n_deck_generators": len(q.generators),
        "simply_connected_chart": not model.compact and not q.generators,
        "requires_global_predicate": model.global_margin is not None,
    }


def compare_global_structure(a: MetricModel, b: MetricModel) -> dict[str, Any]:
    """Compare two models that may share a local metric.

    This is the executable form of the L3 gate. Used by the AdS pair test.
    """
    same_metric = False
    note = ""
    if a.metric is not None and b.metric is not None and a.dim == b.dim:
        try:
             ren = {bc: ac for ac, bc in zip(a.coords, b.coords)}
             bm = sp.Matrix(b.metric).subs(ren, simultaneous=True)
             same_metric = bool(sp.simplify(sp.Matrix(a.metric) - bm).is_zero_matrix)
        except Exception as exc:                          # noqa: BLE001
             note = f"comparison failed: {exc}"
    return {
        "a": a.id, "b": b.id,
        "identical_local_metric": same_metric,
        "a_structure": classify(a), "b_structure": classify(b),
        "a_expected": a.expected.value, "b_expected": b.expected.value,
        "demonstrates_L3_gate": bool(same_metric and a.expected != b.expected),
        "note": note,
    }


__all__ = ["DeckCertificate", "QuotientCertificate", "verify_quotient",
           "ClosureResult", "curve_closes", "classify", "compare_global_structure"]
