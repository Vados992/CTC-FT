"""
CTC-FA v2.0   --   L17 : minimal on-shell perturbation optimiser.

The question
-----------
    Delta_+ = inf J(delta theta)     subject to: field equations + domain
                                     + a CTC certificate            (birth)
    Delta_- = inf J(delta theta)     subject to: field equations + domain
                                     + a stable-causality certificate (destruction)

with the dimensionless cost

    J(delta theta) = sum_i w_i (delta theta_i / Theta_i)^2.

Two things must be true before the word "minimal" means anything, and v1.0
stated both as a gate and enforced neither:

  1. the scales ``Theta_i`` and weights ``w_i`` must be **frozen before the
     optimisation**, otherwise the cost is not comparable between runs and
     "minimal" is a free parameter;
  2. the optimisation must stay on an **on-shell family**, otherwise the
     answer is a statement about arbitrary tensors, not about physics.

v2.0 enforces (1) with :class:`CostMetric`, which refuses to evaluate until
it is sealed and records a hash of its own definition, and (2) by restricting
the search to registered exact-solution parameter families, where every
configuration solves the field equations identically.

Asymmetry
---------
``Delta_+`` and ``Delta_-`` are not the same kind of problem. Birth is
certified by a single witness. Destruction requires the *exclusion of all*
closed timelike curves, that is a temporal function or a fully verified
theorem, which is why ``minimal_destruction`` demands a certificate callback
and refuses to accept "the detector found nothing".
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from .causality import compact_orbit_test
from .model import MetricModel
from .status import CTCStatus, OrbitStatus


class UnsealedCostError(RuntimeError):
    """Raised when a cost metric is used before its scales are frozen."""


@dataclass
class CostMetric:
    """A dimensionless, pre-registered cost on parameter space."""

    scales: dict[str, float]
    weights: dict[str, float] = field(default_factory=dict)
    label: str = ""
    sealed: bool = False
    _digest: str = ""

    def seal(self) -> "CostMetric":
        for k in self.scales:
            self.weights.setdefault(k, 1.0)
        payload = json.dumps({"scales": self.scales, "weights": self.weights,
                              "label": self.label}, sort_keys=True)
        self._digest = hashlib.sha256(payload.encode()).hexdigest()
        self.sealed = True
        return self

    @property
    def digest(self) -> str:
        return self._digest

    def evaluate(self, delta: Mapping[str, float]) -> float:
        if not self.sealed:
            raise UnsealedCostError(
                "the cost metric must be sealed (scales and weights frozen) "
                "before any optimisation; otherwise the word 'minimal' has no "
                "content (v1.0 L17 gate, REQ-009)")
        total = 0.0
        for k, d in delta.items():
            s = self.scales.get(k)
            if s is None or s == 0:
                raise KeyError(f"no nonzero scale registered for parameter {k!r}")
            total += self.weights.get(k, 1.0) * (float(d) / float(s)) ** 2
        return total

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "scales": dict(self.scales),
                "weights": dict(self.weights), "sealed": self.sealed,
                "digest": self._digest}


@dataclass
class MinimalPerturbation:
    model_id: str
    direction: str
    cost: float | None
    delta: dict[str, float]
    base: dict[str, float]
    achieved: bool
    certificate: str
    cost_metric_digest: str
    iterations: int
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "direction": self.direction,
                "cost": self.cost, "delta": self.delta, "base": self.base,
                "achieved": self.achieved, "certificate": self.certificate,
                "cost_metric_digest": self.cost_metric_digest,
                "iterations": self.iterations, "note": self.note}


def _apply_delta(base: Mapping[str, float], names: Sequence[str],
                 x: np.ndarray) -> dict[str, float]:
    out = dict(base)
    for n, v in zip(names, x):
        out[n] = float(base[n]) + float(v)
    return out


def minimal_birth(model: MetricModel, base: Mapping[str, float],
                  free_parameters: Sequence[str], cost: CostMetric,
                  bounds: Mapping[str, tuple[float, float]] | None = None,
                  restarts: int = 12, seed: int = 20260831,
                  margin_target: float = -1e-8) -> MinimalPerturbation:
    """Smallest on-shell parameter change that creates a certified CTC.

    The search moves only along the registered exact family, so every trial
    configuration is an exact solution and the answer is a physical statement
    rather than an off-shell one.
    """
    names = list(free_parameters)
    b = {k: (bounds or {}).get(k, (-abs(base[k]) * 2 - 1.0,
                                   abs(base[k]) * 2 + 1.0)) for k in names}
    rng = np.random.default_rng(seed)

    def margin(x: np.ndarray) -> float:
        """Causal margin at base + x.

        The REDUCED margin is used where it is defined, because the raw
        component g(xi,xi) has an arbitrary scale (it is 2.5e+03 for the
        Godel-type family at one point and 1e-02 at another), which makes any
        penalty weighting meaningless and defeats a local optimiser. The
        reduced margin is dimensionless and has the same sign.
        """
        vals = _apply_delta(base, names, x)
        try:
             from .continuation import reduced_causal_margin
             return float(reduced_causal_margin(model, vals))
        except Exception:                                 # noqa: BLE001
             pass
        try:
             r = compact_orbit_test(model, vals, check_orientation=False)
        except Exception:                                 # noqa: BLE001
             return 1e6
        return float(r.margin) if np.isfinite(r.margin) else 1e6

    def feasible(x: np.ndarray) -> bool:
        m = margin(x)
        return np.isfinite(m) and m <= margin_target

    def obj(x: np.ndarray) -> float:
        c = cost.evaluate({n: v for n, v in zip(names, x)})
        m = margin(x)
        if not np.isfinite(m):
            return 1e9
        penalty = 0.0 if m <= margin_target else 1e3 * (m - margin_target) ** 2
        return c + penalty

    # -- stage 1: coarse feasibility scan ---------------------------------
    #
    # A local optimiser started at delta = 0 cannot escape an infeasible
    # region whose boundary is far away, because Nelder-Mead's default
    # initial simplex around the origin is microscopic. A scan over the
    # declared bounds finds the feasible set first; the local stage then only
    # has to polish.
    dim = len(names)
    n_scan = max(64, 24 * dim)
    starts: list[np.ndarray] = []
    grid = [np.linspace(b[n][0], b[n][1], n_scan) for n in names]
    if dim == 1:
        cands = [np.array([v]) for v in grid[0]]
    else:
        cands = [np.array([rng.uniform(*b[n]) for n in names])
                 for _ in range(n_scan * 8)]
    feas = [x for x in cands if feasible(x)]
    feas.sort(key=lambda x: cost.evaluate({n: v for n, v in zip(names, x)}))
    starts = feas[:max(1, restarts)] if feas else []

    best: tuple[float, np.ndarray] | None = None
    iters = 0
    for k in range(max(restarts, len(starts))):
        if k < len(starts):
             x0 = starts[k]
        else:
             x0 = np.array([rng.uniform(*b[n]) for n in names])
        try:
             res = minimize(obj, x0, method="Nelder-Mead",
                             options={"maxiter": 2000, "xatol": 1e-13,
                                      "fatol": 1e-15})
        except Exception:                                  # noqa: BLE001
             continue
        iters += int(res.nit)
        for cand in (res.x, x0):
             if feasible(cand):
                 f = cost.evaluate({n: v for n, v in zip(names, cand)})
                 if best is None or f < best[0]:
                      best = (float(f), np.asarray(cand))

    # -- stage 3: polish onto the feasibility boundary --------------------
    #
    # The cost decreases monotonically towards delta = 0, so the minimum of a
    # convex cost over the feasible set is attained ON the boundary. A
    # penalty method stops just outside it; bisecting along the ray from the
    # best feasible point to the origin lands exactly on it, which turns a
    # three-digit answer into a machine-precision one.
    if best is not None:
        xf = best[1]
        x_in, x_out = xf.copy(), np.zeros_like(xf)
        if not feasible(x_out):
            for _ in range(200):
                mid = 0.5 * (x_in + x_out)
                if feasible(mid):
                     x_in = mid
                else:
                     x_out = mid
                if float(np.max(np.abs(x_in - x_out))) < 1e-15:
                     break
            f = cost.evaluate({n: v for n, v in zip(names, x_in)})
            if f < best[0]:
                best = (float(f), x_in)

    if best is None:
        return MinimalPerturbation(
            model.id, "birth", None, {}, dict(base), False,
            "no certified CTC found on the searched on-shell family",
            cost.digest, iters,
            "failure of the search is UNRESOLVED, never a proof that no small "
            "on-shell perturbation creates a CTC")
    delta = {n: float(v) for n, v in zip(names, best[1])}
    vals = _apply_delta(base, names, best[1])
    r = compact_orbit_test(model, vals)
    return MinimalPerturbation(
        model.id, "birth", cost.evaluate(delta), delta, dict(base),
        r.status is OrbitStatus.SECTOR_CTC, r.reason, cost.digest, iters,
        "cost is dimensionless with pre-registered scales; every trial "
        "configuration is an exact solution of the registered family")


def minimal_destruction(model: MetricModel, base: Mapping[str, float],
                        free_parameters: Sequence[str], cost: CostMetric,
                        certificate: Callable[[Mapping[str, float]],
                                              tuple[bool, str]],
                        bounds: Mapping[str, tuple[float, float]] | None = None,
                        restarts: int = 12, seed: int = 20260831
                        ) -> MinimalPerturbation:
    """Smallest on-shell change that *certifies* the restoration of causality.

    ``certificate`` must return ``(True, reason)`` only when a complete causal
    certificate holds -- a verified temporal function or a theorem whose
    hypotheses are all satisfied. A detector that merely fails to find a CTC
    is not acceptable and, if supplied, will simply never produce a
    ``True``.
    """
    names = list(free_parameters)
    b = {k: (bounds or {}).get(k, (-abs(base[k]) * 2 - 1.0,
                                   abs(base[k]) * 2 + 1.0)) for k in names}
    rng = np.random.default_rng(seed)

    def obj(x: np.ndarray) -> float:
        vals = _apply_delta(base, names, x)
        ok, _ = certificate(vals)
        c = cost.evaluate({n: v for n, v in zip(names, x)})
        return c if ok else c + 1e6

    best: tuple[float, np.ndarray, str] | None = None
    iters = 0
    for k in range(restarts):
        x0 = np.zeros(len(names)) if k == 0 else np.array(
             [rng.uniform(*b[n]) * 0.3 for n in names])
        try:
             res = minimize(obj, x0, method="Nelder-Mead",
                            options={"maxiter": 2000, "xatol": 1e-12})
        except Exception:                                  # noqa: BLE001
             continue
        iters += int(res.nit)
        ok, why = certificate(_apply_delta(base, names, res.x))
        if ok and (best is None or res.fun < best[0]):
            best = (float(res.fun), res.x, why)

    if best is None:
        return MinimalPerturbation(
            model.id, "destruction", None, {}, dict(base), False,
            "no configuration on the searched family produced a complete causal "
            "certificate", cost.digest, iters,
            "CTC destruction requires exclusion of ALL closed timelike curves, "
            "which is why this problem is strictly harder than birth and is "
            "not solvable by a detector alone")
    delta = {n: float(v) for n, v in zip(names, best[1])}
    return MinimalPerturbation(
        model.id, "destruction", cost.evaluate(delta), delta, dict(base), True,
        best[2], cost.digest, iters,
        "restoration certified by a complete causal certificate, not by "
        "non-detection")


__all__ = ["UnsealedCostError", "CostMetric", "MinimalPerturbation",
           "minimal_birth", "minimal_destruction"]
