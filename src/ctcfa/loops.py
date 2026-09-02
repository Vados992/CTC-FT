"""
CTC-FA v2.0    --   L6 : general closed-curve search.

Scientific status
-----------------
This engine is **one sided by construction**. Finding a closed curve whose
tangent is everywhere timelike is a positive witness; failing to find one is
never evidence of absence. Section 6.2 of v1.0 said so, and REQ-013 required
that non-detection stay ``UNRESOLVED``, but the v1.0 optimizer had three
gaps that could turn a numerical artefact into a "witness":

  D-10a   closure was enforced only for the winding term; a general curve was
          periodic by construction but its *time orientation* was never
          checked and the returned object recorded
          ``"orientation_status": "NOT_CHECKED"``;
  D-10b   the objective used a smooth maximum (log-sum-exp), which can be
          negative while the true maximum is positive, so the reported
          "max norm" was not the quantity the criterion needs;
  D-10c   a candidate found on the optimizer's own sample grid was never
          re-validated on a denser grid or in an alternative parametrisation.

v2.0 keeps the Fourier ansatz but separates *search* from *validation*:

    search      -- any optimiser, any smoothing, any heuristic; produces
                   candidates only;
    validate    -- exact maximum of ``g(gamma_dot, gamma_dot)`` on a refined
                   grid, closure test on the quotient, time-orientation test,
                   non-degeneracy of the tangent, and a repeat at doubled
                   resolution.

Only a candidate that survives *validation* is reported, and even then it is
reported as ``CTC_PROVED`` at ``SECTOR`` scope pending an independent
symbolic or interval confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import sympy as sp
from scipy.optimize import minimize

from .model import MetricModel
from .orientation import check_curve_orientation
from .status import (ClaimScope, CTCStatus, Gate, GateRecord, GateResult, Verdict)
from .topology import curve_closes


@dataclass
class LoopCandidate:
    max_norm: float
    mean_norm: float
    min_speed: float
    points: np.ndarray
    tangents: np.ndarray
    coefficients: np.ndarray
    n_samples: int
    winding: tuple[int, ...] = ()
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"max_norm": self.max_norm, "mean_norm": self.mean_norm,
                "min_speed": self.min_speed, "n_samples": self.n_samples,
                "winding": list(self.winding), "note": self.note}


@dataclass
class LoopValidation:
    timelike_everywhere: bool
    max_norm: float
    max_norm_normalised: float = 0.0
    coordinate_extent: float = 0.0
    closed: bool = False
    closure_mechanism: str = ""
    closure_residual: float = float("nan")
    orientation_consistent: bool = False
    degenerate_tangent: bool = True
    resolution_stable: bool = False
    n_samples: int = 0
    bounded: bool = True
    note: str = ""

    @property
    def accepted(self) -> bool:
        return (self.timelike_everywhere and self.closed
                and self.orientation_consistent and not self.degenerate_tangent
                and self.resolution_stable and self.bounded)

    def as_dict(self) -> dict[str, Any]:
        return {"timelike_everywhere": self.timelike_everywhere,
                "max_norm": self.max_norm,
                "max_norm_normalised": self.max_norm_normalised,
                "coordinate_extent": self.coordinate_extent,
                "bounded": self.bounded, "closed": self.closed,
                "closure_mechanism": self.closure_mechanism,
                "closure_residual": self.closure_residual,
                "orientation_consistent": self.orientation_consistent,
                "degenerate_tangent": self.degenerate_tangent,
                "resolution_stable": self.resolution_stable,
                "n_samples": self.n_samples, "accepted": self.accepted,
                "note": self.note}


class LoopSearch:
    """Fourier-series closed-curve search with an explicit validation stage."""

    def __init__(self, model: MetricModel, params: Mapping[str, float],
                  modes: int = 3, samples: int = 256):
        self.model = model
        self.modes = modes
        self.samples = samples
        g = model.require_metric()
        self.coords = list(model.coords)
        self.n = len(self.coords)
        self.pvals = [float(params[str(s)]) for s in model.params]
        # Component-wise lambdification so the metric can be evaluated on a
        # whole sampled curve in one vectorised call. Evaluating a lambdified
        # Matrix point by point was the dominant cost of the v1.0 search.
        self._gc = [[sp.lambdify(self.coords + list(model.params), g[i, j],
                                   "numpy")
                      for j in range(self.n)] for i in range(self.n)]
        self.params = dict(params)
        self.compact_index = (model.compact[0].index if model.compact else None)
        self.period = 0.0
        if model.compact:
            try:
                 self.period = float(sp.N(model.compact[0].period.subs(
                     model.substitution(params))))
            except Exception:                             # noqa: BLE001
                 self.period = 0.0

    # -- curve construction ----------------------------------------------

    def unpack(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, K = self.n, self.modes
        centre = theta[:n]
        A = theta[n:n + n * K].reshape(n, K)
        B = theta[n + n * K:n + 2 * n * K].reshape(n, K)
        return centre, A, B

    def curve(self, theta: np.ndarray, samples: int | None = None,
              winding: int = 0) -> tuple[np.ndarray, np.ndarray]:
        S = samples or self.samples
        centre, A, B = self.unpack(theta)
        lam = np.linspace(0.0, 1.0, S, endpoint=False)
        ks = np.arange(1, self.modes + 1)
        ang = 2 * np.pi * np.outer(lam, ks)
        x = centre[None, :] + np.cos(ang) @ A.T + np.sin(ang) @ B.T
        dx = ((-np.sin(ang) * (2 * np.pi * ks)) @ A.T
              + (np.cos(ang) * (2 * np.pi * ks)) @ B.T)
        if winding and self.compact_index is not None and self.period:
            x[:, self.compact_index] += self.period * winding * lam
            dx[:, self.compact_index] += self.period * winding
        return x, dx

    def metrics_along(self, x: np.ndarray) -> np.ndarray:
        S = x.shape[0]
        cols = [x[:, k] for k in range(self.n)] + [
            np.full(S, v) for v in self.pvals]
        out = np.empty((S, self.n, self.n))
        with np.errstate(all="ignore"):
            for i in range(self.n):
                for j in range(self.n):
                    try:
                        v = self._gc[i][j](*cols)
                    except Exception:                     # noqa: BLE001
                        v = np.nan
                    out[:, i, j] = np.broadcast_to(np.asarray(v, dtype=float), (S,))
        return (out + np.transpose(out, (0, 2, 1))) / 2.0

    def norms(self, theta: np.ndarray, samples: int | None = None,
              winding: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, dx = self.curve(theta, samples, winding)
        G = self.metrics_along(x)
        q = np.einsum("ij,ijk,ik->i", dx, G, dx)
        return q, x, dx

    # -- search ------------------------------------------------------------

    def objective(self, theta: np.ndarray, winding: int, alpha: float = 40.0,
                  amplitude_cap: float = 5.0) -> float:
        """Smoothed maximum of g(gamma_dot, gamma_dot), with two guards.

        The amplitude cap keeps the curve in a bounded coordinate region: an
        unconstrained optimiser drives the Fourier coefficients to infinity,
        because a curve of unbounded size trivially attains an unbounded
        negative norm. That runaway is a parametrisation artefact, not a
        CTC, and it is exactly the class of false positive FM-004 describes.
        The speed floor rejects degenerate (zero-tangent) curves.
        """
        q, _, dx = self.norms(theta, winding=winding)
        if not np.all(np.isfinite(q)):
            return 1e6
        smooth_max = float(np.log(np.sum(np.exp(alpha * (q - q.max()))))
                           / alpha + q.max())
        speed = np.einsum("ij,ij->i", dx, dx)
        penalty = float(np.sum(np.maximum(0.0, 1e-3 - speed) ** 2)) * 1e3
        amp = float(np.sum(theta[self.n:] ** 2))
        penalty += 1e3 * max(0.0, amp - amplitude_cap ** 2) ** 2
        # Normalise by the curve speed so that the reported number is a
        # scale-free causal statement rather than a proxy for curve length.
        norm = max(1e-12, float(np.mean(speed)))
        return smooth_max / norm + penalty

    def search(self, centre: Mapping[str, float], winding: int = 1,
               restarts: int = 8, scale: float = 0.4, seed: int = 20260831,
               maxiter: int = 150) -> LoopCandidate | None:
        cands = self.search_all(centre, winding, restarts, scale, seed, maxiter)
        return cands[0] if cands else None

    def search_all(self, centre: Mapping[str, float], winding: int = 1,
                   restarts: int = 8, scale: float = 0.4, seed: int = 20260831,
                   maxiter: int = 150) -> list[LoopCandidate]:
        """Return every restart's candidate, best objective first.

        The verdict layer validates candidates in order and accepts the first
        that survives, instead of validating only the single best objective
        value. A runaway curve can win on the smoothed objective while
        failing validation, and discarding the runner-up would then lose a
        perfectly good witness.
        """
        rng = np.random.default_rng(seed)
        c0 = np.array([float(centre[str(s)]) for s in self.coords])
        dim = self.n * (1 + 2 * self.modes)
        out: list[tuple[float, np.ndarray]] = []
        for attempt in range(restarts):
            th = np.zeros(dim)
            th[:self.n] = c0
            # Restart 0 is the pure symmetry orbit (all Fourier coefficients
            # zero, closure supplied entirely by the winding term). Seeding
            # the search with the object the L5 detector already understands
            # makes the general search a strict extension of it rather than an
            # unrelated heuristic.
            if attempt > 0:
                 th[self.n:] = rng.normal(scale=scale, size=dim - self.n)
            try:
                 res = minimize(self.objective, th, args=(winding,),
                                method="Powell",
                                options={"maxiter": maxiter, "xtol": 1e-8,
                                         "ftol": 1e-10})
            except Exception:                             # noqa: BLE001
                 continue
            out.append((float(res.fun), res.x))
        # Always include the untouched symmetry orbit as a candidate.
        seed_theta = np.zeros(dim)
        seed_theta[:self.n] = c0
        out.append((self.objective(seed_theta, winding), seed_theta))
        out.sort(key=lambda t: t[0])
        cands: list[LoopCandidate] = []
        for fval, th in out:
            q, x, dx = self.norms(th, winding=winding)
            if not np.all(np.isfinite(q)):
                 continue
            speed = np.einsum("ij,ij->i", dx, dx)
            cands.append(LoopCandidate(
                 float(np.max(q)), float(np.mean(q)), float(np.min(speed)),
                 x, dx, th, self.samples, (winding,),
                 "candidate only: not a witness until validated"))
        return cands

    # -- validation --------------------------------------------------------

    def validate(self, cand: LoopCandidate, winding: int = 1,
                 refine: int = 4, tol: float = 1e-10,
                 extent_limit: float = 1e4) -> LoopValidation:
        """Exact re-check of a candidate at higher resolution."""
        S = cand.n_samples * refine
        q, x, dx = self.norms(cand.coefficients, samples=S, winding=winding)
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(x))):
            return LoopValidation(False, float("nan"), 0.0, float("nan"),
                                  n_samples=S, bounded=False,
                                  note="candidate leaves the chart: the metric "
                                       "is not finite somewhere on the curve")
        extent = float(np.max(np.abs(x)))
        if extent > extent_limit or abs(float(np.max(np.abs(q)))) > 1e100:
            return LoopValidation(False, float(np.max(q)), 0.0, extent,
                                  n_samples=S, bounded=False,
                                  note=f"runaway candidate: coordinate extent "
                                       f"{extent:.3e} exceeds the limit "
                                       f"{extent_limit:.3e}. An unbounded curve "
                                       f"attains an unbounded negative norm by "
                                       f"parametrisation alone and is not a CTC.")
        maxq = float(np.max(q))
        speed = np.einsum("ij,ij->i", dx, dx)
        maxq_norm = float(np.max(q / np.maximum(speed, 1e-300)))
        timelike = maxq < -tol

        q2, _, _ = self.norms(cand.coefficients, samples=2 * S, winding=winding)
        stable = bool(np.sign(np.max(q2)) == np.sign(maxq)
                      and abs(np.max(q2) - maxq) <= 0.05 * max(1.0, abs(maxq)))

        p0 = list(x[0])
        p1 = list(x[-1] + (x[1] - x[0]))           # close the sample ring
        cl = curve_closes(self.model, p0, p1, tol=1e-6,
                          params=self.model.substitution(self.params))

        G = self.metrics_along(x)
        orr = check_curve_orientation(list(G), list(dx), closed=True)

        degenerate = bool(np.min(np.abs(speed)) < 1e-8)

        return LoopValidation(
            timelike, maxq, maxq_norm, extent, cl.closed, cl.mechanism,
            cl.residual, orr.consistent, degenerate, stable, S, True,
            note=("validation uses the EXACT maximum of g(gamma_dot, gamma_dot) "
                  "on a refined grid, not the smoothed objective the optimiser "
                  "minimised; the reported normalised margin is "
                  "max g(u,u)/|u|^2, which for a pure symmetry orbit reduces "
                  "exactly to the L5 compact-orbit margin"))


def loop_search_verdict(model: MetricModel, centre: Mapping[str, float],
                         winding: int = 1, **kw: Any) -> Verdict:
    """Run search + validation and produce a correctly scoped verdict."""
    gates = GateRecord()
    gates.set(Gate.V0_PARSE, GateResult.PASS)
    gates.set(Gate.V1_ALGEBRA, GateResult.PASS)
    ls = LoopSearch(model, centre, modes=kw.pop("modes", 3),
                     samples=kw.pop("samples", 256))
    cands = ls.search_all(centre, winding=winding, **kw)
    cand = None
    val = None
    for c in cands[:5]:
        v = ls.validate(c, winding=winding)
        if val is None or (v.accepted and not val.accepted):
            cand, val = c, v
        if v.accepted:
            break
    if cand is None:
        gates.set(Gate.V4_CTC_WITNESS, GateResult.NOT_ATTEMPTED, "optimiser failed")
        return Verdict(CTCStatus.UNRESOLVED, ClaimScope.SECTOR, float("nan"),
                        "loop_search", "optimiser produced no candidate; this is "
                                       "NOT evidence of absence",
                        model.id, dict(centre), gates)
    if val.accepted:
        gates.set(Gate.V4_CTC_WITNESS, GateResult.PASS,
                   "validated closed timelike curve")
        v = Verdict(CTCStatus.CTC_PROVED, ClaimScope.SECTOR,
                     val.max_norm_normalised, "loop_search",
                     "closed curve with max g(u,u)/|u|^2 = "
                     f"{val.max_norm_normalised:+.6e} < 0, "
                     f"closure via {val.closure_mechanism}, "
                     "time orientation consistent, stable under grid refinement",
                     model.id, dict(centre), gates, val.as_dict())
        return v
    gates.set(Gate.V4_CTC_WITNESS, GateResult.INCONCLUSIVE,
               "candidate failed validation")
    return Verdict(CTCStatus.UNRESOLVED, ClaimScope.SECTOR,
                    val.max_norm_normalised, "loop_search",
                    "candidate rejected by validation "
                    f"(timelike={val.timelike_everywhere}, closed={val.closed}, "
                    f"orientation={val.orientation_consistent}, "
                    f"stable={val.resolution_stable}, bounded={val.bounded}); "
                    "a failed search is never a no-CTC result",
                    model.id, dict(centre), gates, val.as_dict())


__all__ = ["LoopCandidate", "LoopValidation", "LoopSearch", "loop_search_verdict"]
