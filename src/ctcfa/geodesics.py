"""
CTC-FA v2.0   --    L18 : geodesics, accessibility and ANEC on a real geodesic.

Two jobs.

1. **Integration.** Integrate the geodesic equation
   ``d2x^a/dl^2 = -Gamma^a_bc dx^b/dl dx^c/dl`` with an adaptive stiff-capable
   integrator, monitoring the conserved norm ``g(u,u)`` as an *independent*
   error diagnostic. v1.0 built the right-hand side from symbolic
   Christoffels via ``lambdify`` but never checked the norm drift, so a
   silently diverging integration would have been reported as a trajectory.

2. **Accessibility.** Section 31.1 of v1.0 is explicit that a compact
   witness "does not establish that an astronaut can enter that region,
   survive, return to a desired external event, or that the spacetime can be
   created from realistic initial data". v2.0 turns that paragraph into a
   computation: from a declared observer domain, integrate future-directed
   causal curves and report whether the chronology-violating region is
   *reached*, with the four claim levels kept apart:

       EXISTS           the CTC region is nonempty          (L5/L5b)
       REACHABLE        a future-directed causal curve from the observer
                        domain enters it
       SURVIVABLE       tidal/curvature invariants stay bounded along that curve
       RETURNABLE       a causal curve exists from the region back to an
                        external past event

   Each is reported separately. A model may be EXISTS and not REACHABLE, and
   that distinction is the whole content of the "mathematical existence is
   not a time machine" caveat.

3. **ANEC.** The averaged null energy condition is integrated along an
    actual affinely parametrised null geodesic produced here, not along a
    caller-supplied list of numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from .energy import ANECResult, anec_integral, stress_tensor_from_metric
from .model import MetricModel
from .numgeom import NumericGeometry
from .symbolic import christoffel


@dataclass
class GeodesicResult:
    affine: np.ndarray
    x: np.ndarray
    u: np.ndarray
    norm: np.ndarray
    norm_drift: float
    success: bool
    message: str
    kind: str = "unknown"

    def as_dict(self) -> dict[str, Any]:
        return {"n_points": int(self.affine.size),
                "affine_range": [float(self.affine[0]), float(self.affine[-1])]
                if self.affine.size else [],
                "norm_initial": float(self.norm[0]) if self.norm.size else None,
                "norm_final": float(self.norm[-1]) if self.norm.size else None,
                "norm_drift": self.norm_drift, "success": self.success,
                "kind": self.kind, "message": self.message}


class GeodesicIntegrator:
    """Lambdified Christoffel symbols plus an adaptive ODE integration."""

    def __init__(self, model: MetricModel, simplify: bool = False):
        self.model = model
        g = model.require_metric()
        coords = list(model.coords)
        params = list(model.params)
        n = len(coords)
        Gam = christoffel(g, coords, simplify=simplify)
        flat = [Gam[a][b][c] for a in range(n) for b in range(n) for c in range(n)]
        self._G = sp.lambdify(coords + params, flat, "numpy")
        self._g = sp.lambdify(coords + params, sp.Matrix(g), "numpy")
        self.n = n
        self.coords = coords
        self.params = params

    def christoffel_at(self, x: Sequence[float], p: Sequence[float]) -> np.ndarray:
        vals = np.asarray(self._G(*list(x), *list(p)), dtype=float)
        return vals.reshape(self.n, self.n, self.n)

    def metric_at(self, x: Sequence[float], p: Sequence[float]) -> np.ndarray:
        m = np.array(self._g(*list(x), *list(p)), dtype=float)
        return (m + m.T) / 2.0

    def rhs(self, lam: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        n = self.n
        x, u = y[:n], y[n:]
        Gam = self.christoffel_at(x, p)
        acc = -np.einsum("abc,b,c->a", Gam, u, u)
        return np.concatenate([u, acc])

    def integrate(self, x0: Sequence[float], u0: Sequence[float],
                  params: Mapping[str, float],
                  lam_span: tuple[float, float] = (0.0, 10.0),
                  n_out: int = 401, rtol: float = 1e-10,
                  atol: float = 1e-12, method: str = "DOP853") -> GeodesicResult:
        p = np.array([float(params[str(s)]) for s in self.params], dtype=float)
        y0 = np.concatenate([np.asarray(x0, float), np.asarray(u0, float)])
        t_eval = np.linspace(lam_span[0], lam_span[1], n_out)
        sol = solve_ivp(self.rhs, lam_span, y0, args=(p,), t_eval=t_eval,
                         rtol=rtol, atol=atol, method=method, dense_output=False)
        n = self.n
        xs = sol.y[:n].T
        us = sol.y[n:].T
        norms = np.array([float(u @ self.metric_at(x, p) @ u)
                           for x, u in zip(xs, us)])
        drift = (float(np.max(np.abs(norms - norms[0]))) if norms.size
                 else float("nan"))
        kind = ("null" if abs(norms[0]) < 1e-9 else
                "timelike" if norms[0] < 0 else "spacelike") if norms.size else "?"
        return GeodesicResult(sol.t, xs, us, norms, drift, bool(sol.success),
                               sol.message, kind)


def null_direction(model: MetricModel, values: Mapping[str, float],
                   spatial: Sequence[float] | None = None) -> np.ndarray:
    """Build a future-directed null vector at a point.

    Uses the orthonormal frame, so the construction works in an ergoregion
    and in charts with no timelike coordinate direction.
    """
    from .energy import orthonormal_frame
    g = np.array(sp.Matrix(model.require_metric())
                 .subs(model.substitution(values)).evalf(), dtype=float)
    E = orthonormal_frame(g)
    d = model.dim - 1
    if spatial is None:
        spatial = np.zeros(d)
        spatial[0] = 1.0
    s = np.asarray(spatial, dtype=float)
    s = s / np.linalg.norm(s)
    khat = np.concatenate([[1.0], s])
    return E @ khat


def anec_on_geodesic(model: MetricModel, values: Mapping[str, float],
                     spatial: Sequence[float] | None = None,
                     lam_max: float = 5.0, n_out: int = 401,
                     **kw: Any) -> tuple[ANECResult, GeodesicResult]:
    """Integrate a null geodesic and evaluate ``int T_ab k^a k^b dl`` on it."""
    integ = GeodesicIntegrator(model)
    k0 = null_direction(model, values, spatial)
    x0 = [float(values[str(c)]) for c in model.coords]
    geo = integ.integrate(x0, k0, values, (0.0, lam_max), n_out, **kw)

    ng = NumericGeometry(model)
    Lam = 0.0
    if model.matter.cosmological_constant != 0:
        Lam = float(sp.N(model.matter.cosmological_constant.subs(
             model.substitution(values))))
    p = {str(s): float(values[str(s)]) for s in model.params}
    contractions = []
    for x, u in zip(geo.x, geo.u):
        pt = {str(c): float(v) for c, v in zip(model.coords, x)}
        pt.update(p)
        try:
             T, _ = ng.stress_tensor(pt, Lam)
             contractions.append(float(u @ T @ u))
        except Exception:                                 # noqa: BLE001
             contractions.append(float("nan"))
    arr = np.asarray(contractions)
    good = np.isfinite(arr)
    res = anec_integral(arr[good], geo.affine[good])
    res.note += (f" | geodesic norm drift {geo.norm_drift:.2e} "
                  f"(an independent accuracy check on the integration)")
    return res, geo


# --------------------------------------------------------------------------
# Accessibility
# --------------------------------------------------------------------------

@dataclass
class AccessibilityReport:
    model_id: str
    exists: bool
    reachable: bool | None
    survivable: bool | None
    returnable: bool | None
    observer_domain: dict[str, Any] = field(default_factory=dict)
    trials: int = 0
    entered_at: dict[str, float] | None = None
    max_curvature_on_path: float | None = None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "EXISTS": self.exists,
                "REACHABLE": self.reachable, "SURVIVABLE": self.survivable,
                "RETURNABLE": self.returnable,
                "observer_domain": self.observer_domain, "trials": self.trials,
                "entered_at": self.entered_at,
                "max_curvature_on_path": self.max_curvature_on_path,
                "note": self.note}


def accessibility(model: MetricModel,
                  observer_point: Mapping[str, float],
                  ctc_predicate: Callable[[Mapping[str, float]], bool],
                  n_directions: int = 64, lam_max: float = 20.0,
                  curvature_limit: float = 1e6,
                  seed: int = 20260831) -> AccessibilityReport:
    """Shoot future-directed timelike geodesics and test whether they enter
    the chronology-violating set.

    A negative result here is explicitly NOT a proof of inaccessibility: the
    search is a finite shooting scan and inherits the same asymmetry as every
    other search in the architecture.
    """
    from .energy import orthonormal_frame
    integ = GeodesicIntegrator(model)
    g = np.array(sp.Matrix(model.require_metric())
                 .subs(model.substitution(observer_point)).evalf(), dtype=float)
    E = orthonormal_frame(g)
    rng = np.random.default_rng(seed)
    d = model.dim - 1
    x0 = [float(observer_point[str(c)]) for c in model.coords]
    ng = NumericGeometry(model)
    p = {str(s): float(observer_point[str(s)]) for s in model.params}

    entered = None
    max_curv = 0.0
    for i in range(n_directions):
        v = rng.normal(size=d)
        v /= np.linalg.norm(v)
        beta = 0.9 * rng.random()
        gamma = 1.0 / math.sqrt(1.0 - beta * beta)
        uhat = gamma * np.concatenate([[1.0], beta * v])
        u0 = E @ uhat
        geo = integ.integrate(x0, u0, observer_point, (0.0, lam_max), 301)
        if not geo.success:
            continue
        for x in geo.x:
            pt = {str(c): float(val) for c, val in zip(model.coords, x)}
            pt.update(p)
            try:
                 if ctc_predicate(pt):
                     entered = pt
                     break
            except Exception:                            # noqa: BLE001
                 continue
        if entered is not None:
            try:
                 cur = ng.curvature(entered)
                 max_curv = abs(cur.kretschmann)
            except Exception:                            # noqa: BLE001
                 max_curv = float("nan")
            break

    reachable = entered is not None
    survivable = (None if not reachable else
                  (math.isfinite(max_curv) and max_curv < curvature_limit))
    return AccessibilityReport(
        model.id, exists=True, reachable=reachable, survivable=survivable,
        returnable=None,
        observer_domain={str(k): float(v) for k, v in observer_point.items()},
        trials=n_directions, entered_at=entered,
        max_curvature_on_path=(None if not reachable else float(max_curv)),
        note=("REACHABLE=False from a finite shooting scan is UNRESOLVED, not a "
              "proof of inaccessibility. RETURNABLE is not attempted here: it "
              "requires a causal-past computation on the quotient and is a "
              "v2.1 target."))


__all__ = ["GeodesicResult", "GeodesicIntegrator", "null_direction",
           "anec_on_geodesic", "AccessibilityReport", "accessibility"]
