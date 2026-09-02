"""
CTC-FA v2.0   --   Time-orientation engine.

Section 3.2 of v1.0 states the requirement plainly:

    "A complete witness must also respect a consistent time orientation.   A
    curve that changes the chosen future direction is not accepted as a
    physical future-directed CTC witness."

The v1.0 kernel never implemented it. ``validate_sampled_loop`` returned
``"orientation_status": "NOT_CHECKED"`` and ``compact_orbit_test`` did not
look at orientation at all, so a closed *timelike* curve that reverses the
future direction -- which is not a causality violation, merely a
non-time-orientable circuit -- would have been reported as a CTC. That is
v2.0 defect **D-03**.

Method
------
1. At every sample point the metric matrix ``g`` is diagonalised. Its unique
    negative-eigenvalue eigenvector ``T`` is timelike, because
    ``T^t g T = lambda |T|^2 < 0``.
2. The sign of ``T`` is fixed by *continuity in the cone*: two timelike
    vectors lie in the same cone iff their inner product is negative, so the
    sign of ``T_i`` is chosen such that ``g(T_i, T_{i-1}) < 0``. This
    produces a continuous future-directed reference field along the curve
    without needing a global time function.
3. A curve is orientation-consistent when ``g(gamma_dot, T)`` never changes
    sign along the curve.
4. For a closed curve on a quotient the field must also close: the reference
    at the endpoint has to lie in the same cone as the deck-transported
    reference from the start point. A sign flip there is a *time-orientation
    holonomy* and disqualifies the loop as a future-directed CTC witness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np


@dataclass
class OrientationResult:
    consistent: bool
    reversals: int
    holonomy_ok: bool | None
    min_abs_inner: float
    reference_field: list[list[float]] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"consistent": self.consistent, "reversals": self.reversals,
                "holonomy_ok": self.holonomy_ok,
                "min_abs_inner": self.min_abs_inner, "note": self.note}


def timelike_eigenvector(g: np.ndarray) -> tuple[np.ndarray, float]:
    """Return a timelike vector at a point plus its eigenvalue."""
    gs = (g + g.T) / 2.0
    vals, vecs = np.linalg.eigh(gs)
    i = int(np.argmin(vals))
    return np.asarray(vecs[:, i], dtype=float), float(vals[i])


def continuous_reference_field(metrics: Sequence[np.ndarray],
                               seed: np.ndarray | None = None
                               ) -> tuple[list[np.ndarray], str]:
    """Build a cone-continuous timelike reference field along a sample path."""
    field_: list[np.ndarray] = []
    note = ""
    prev: np.ndarray | None = None
    prev_g: np.ndarray | None = None
    for g in metrics:
        T, lam = timelike_eigenvector(np.asarray(g, dtype=float))
        if lam >= 0:
            note = "metric has no timelike eigendirection at some sample point"
        if prev is None:
            if seed is not None and float(seed @ np.asarray(g) @ T) > 0:
                T = -T
        else:
            ip = float(prev @ np.asarray(g, dtype=float) @ T)
            if ip > 0:
                T = -T
        field_.append(T)
        prev, prev_g = T, np.asarray(g, dtype=float)
    return field_, note


def check_curve_orientation(metrics: Sequence[np.ndarray],
                            tangents: Sequence[np.ndarray],
                            closed: bool = True,
                            deck_jacobian: np.ndarray | None = None,
                            tol: float = 1e-12) -> OrientationResult:
    """Full orientation test for a sampled curve.

    Parameters
    ----------
    metrics
        ``g`` at each sample point, in coordinate components.
    tangents
        ``gamma_dot`` at each sample point, same components.
    closed
        Whether the curve is closed (then the holonomy test applies).
    deck_jacobian
        Jacobian ``d phi / d x`` of the deck element that identifies the
        endpoint with the start point, if the closure used one. ``None``
        means the identity.
    """
    metrics = [np.asarray(g, dtype=float) for g in metrics]
    tangents = [np.asarray(v, dtype=float) for v in tangents]
    ref, note = continuous_reference_field(metrics)

    inners = [float(v @ g @ T) for v, g, T in zip(tangents, metrics, ref)]
    finite = [x for x in inners if np.isfinite(x)]
    if not finite:
        return OrientationResult(False, 0, None, 0.0, note="no finite inner products")
    signs = [1 if x > 0 else -1 for x in finite]
    reversals = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
    min_abs = float(min(abs(x) for x in finite))

    holonomy_ok: bool | None = None
    if closed and len(ref) >= 2:
        J = np.eye(metrics[0].shape[0]) if deck_jacobian is None \
            else np.asarray(deck_jacobian, dtype=float)
        start_pushed = J @ ref[0]
        ip = float(start_pushed @ metrics[-1] @ ref[-1])
        if abs(ip) <= tol:
            holonomy_ok = None
            note = (note + " | " if note else "") + "degenerate holonomy probe"
        else:
            holonomy_ok = ip < 0
            if not holonomy_ok:
                note = (note + " | " if note else "") + \
                       "time-orientation holonomy reverses around the loop"

    consistent = (reversals == 0) and (holonomy_ok is not False) and min_abs > tol
    if min_abs <= tol:
        note = (note + " | " if note else "") + \
               "tangent is (numerically) orthogonal to the reference field"
    return OrientationResult(consistent, reversals, holonomy_ok, min_abs,
                             [list(map(float, T)) for T in ref], note)


def orbit_orientation(g_at_point: np.ndarray, generator: np.ndarray,
                      tol: float = 1e-12) -> OrientationResult:
    """Orientation test for a *symmetry orbit* of a Killing generator.

    A closed orbit of a Killing vector has constant tangent ``xi`` in the
    adapted chart and the metric is constant along it, so orientation is
    consistent by construction whenever ``xi`` is timelike; the test reduces
    to checking that ``xi`` is genuinely timelike and non-degenerate. This
    is stated explicitly so that the compact-orbit detector's orientation
    claim is derived, not assumed.
    """
    g = np.asarray(g_at_point, dtype=float)
    xi = np.asarray(generator, dtype=float)
    q = float(xi @ g @ xi)
    if not np.isfinite(q):
        return OrientationResult(False, 0, None, 0.0, note="non-finite orbit norm")
    if q >= -tol:
        return OrientationResult(False, 0, None, abs(q),
                                 note="orbit tangent is not timelike; orientation "
                                      "consistency is vacuous")
    T, _ = timelike_eigenvector(g)
    ip = float(xi @ g @ T)
    if ip > 0:
        T = -T
        ip = -ip
    return OrientationResult(True, 0, True, abs(ip), [list(map(float, T))],
                             "Killing orbit: metric and tangent are invariant along "
                             "the orbit, so the future direction cannot rotate")


__all__ = ["OrientationResult", "timelike_eigenvector", "continuous_reference_field",
           "check_curve_orientation", "orbit_orientation"]
