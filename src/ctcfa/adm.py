"""
CTC-FA v2.0   --   L12 : ADM / initial-data engine.

Constructibility from regular initial data is the strongest claim level the
architecture can reach short of quantum questions, because it is the only one
that distinguishes "this metric exists as a solution" from "this spacetime
can form". v1.0 shipped ``adm_hamiltonian_residual`` and
``adm_momentum_residual`` as symbolic expressions and nothing that used them.

v2.0 supplies:

  * a **3+1 decomposition** of a registered 4-metric on a declared slicing,
    producing lapse, shift, spatial metric and extrinsic curvature;
  * the **Hamiltonian and momentum constraints** evaluated on that data with
    a reported residual;
  * a **conformal York solver** for the Hamiltonian constraint on
    maximal, conformally flat data, so that "regular initial data" can be
    constructed rather than merely postulated;
  * an explicit statement of what is still missing (a full BSSN/Z4 evolution),
    with the gate held at ``NOT_ATTEMPTED`` rather than silently passed.

Conventions
-----------
    g = -(N^2 - N_i N^i) dt^2 + 2 N_i dt dx^i + h_ij dx^i dx^j
    K_ij = (1 / 2N) ( D_i N_j + D_j N_i - dh_ij/dt )
    Hamiltonian: R^(3) + K^2 - K_ij K^ij = 16 pi rho
    Momentum:     D_j ( K^ij - h^ij K ) = 8 pi j^i
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import sympy as sp

from .model import MetricModel
from .symbolic import christoffel, ricci_scalar, ricci_tensor


@dataclass
class ADMData:
    lapse: sp.Expr
    shift: sp.Matrix
    spatial_metric: sp.Matrix
    extrinsic_curvature: sp.Matrix
    spatial_coords: tuple[sp.Symbol, ...]
    time_coord: sp.Symbol
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"lapse": str(self.lapse),
                "shift": [str(s) for s in self.shift],
                "spatial_metric": [[str(self.spatial_metric[i, j])
                                    for j in range(self.spatial_metric.rows)]
                                   for i in range(self.spatial_metric.rows)],
                "extrinsic_curvature": [[str(self.extrinsic_curvature[i, j])
                                          for j in range(self.extrinsic_curvature.rows)]
                                         for i in range(self.extrinsic_curvature.rows)],
                "note": self.note}


def decompose(model: MetricModel, time_index: int = 0) -> ADMData:
    """3+1 decomposition of a registered metric on a constant-x^0 slicing."""
    g = model.require_metric()
    n = model.dim
    t = model.coords[time_index]
    sidx = [i for i in range(n) if i != time_index]
    scoords = tuple(model.coords[i] for i in sidx)

    h = g.extract(sidx, sidx)
    if sp.simplify(h.det()) == 0:
        raise ValueError(
            f"the constant-{model.coords[time_index]} slices of this chart are "
            f"degenerate (det h = 0), so no 3+1 decomposition exists on this "
            f"slicing. Null-adapted charts such as the Ori-2007 core "
            f"(g_vv, g_vr with g_rr = 0) require a different foliation before "
            f"the ADM constraints are even defined.")
    hinv = h.inv()
    Ni_down = sp.Matrix([g[time_index, i] for i in sidx])       # N_i
    Ni_up = sp.simplify(hinv * Ni_down)                          # N^i
    N2 = sp.simplify(-(g[time_index, time_index] - (Ni_down.T * Ni_up)[0]))
    N = sp.sqrt(sp.simplify(N2))

    m = len(sidx)
    Gam3 = christoffel(h, scoords, simplify=True)

    def D(i: int, j: int, vec_down: sp.Matrix) -> sp.Expr:
        expr = sp.diff(vec_down[j], scoords[i])
        for k in range(m):
            expr -= Gam3[k][i][j] * vec_down[k]
        return sp.simplify(expr)

    K = sp.zeros(m, m)
    for i in range(m):
        for j in range(i, m):
            val = sp.simplify((D(i, j, Ni_down) + D(j, i, Ni_down)
                               - sp.diff(h[i, j], t)) / (2 * N))
            K[i, j] = val
            K[j, i] = val
    return ADMData(sp.simplify(N), Ni_up, h, K, scoords, t,
                   "constant-x^0 slicing of the registered chart")


@dataclass
class ConstraintResidual:
    hamiltonian: sp.Expr
    momentum: list[sp.Expr]
    hamiltonian_value: float | None = None
    momentum_values: list[float] = field(default_factory=list)
    tolerance: float = 1e-9
    satisfied: bool | None = None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"hamiltonian": str(self.hamiltonian),
                "momentum": [str(m) for m in self.momentum],
                "hamiltonian_value": self.hamiltonian_value,
                "momentum_values": self.momentum_values,
                "tolerance": self.tolerance, "satisfied": self.satisfied,
                "note": self.note}


def constraints(data: ADMData, rho: sp.Expr = sp.S.Zero,
                j_up: Sequence[sp.Expr] | None = None,
                simplify: bool = True) -> ConstraintResidual:
    """Hamiltonian and momentum constraint residuals for the given data."""
    h, K = data.spatial_metric, data.extrinsic_curvature
    coords = data.spatial_coords
    m = h.rows
    hinv = h.inv()
    R3 = ricci_scalar(h, coords, simplify=simplify)
    Kmixed = hinv * K
    trK = sp.simplify(sum(Kmixed[i, i] for i in range(m)))
    KijKij = sp.simplify(sum(Kmixed[i, j] * Kmixed[j, i]
                             for i in range(m) for j in range(m)))
    ham = sp.simplify(R3 + trK ** 2 - KijKij - 16 * sp.pi * rho)

    Gam3 = christoffel(h, coords, simplify=simplify)
    Kup = sp.simplify(hinv * K * hinv)
    A = sp.Matrix(m, m, lambda i, j: sp.simplify(Kup[i, j] - hinv[i, j] * trK))
    j_up = list(j_up) if j_up is not None else [sp.S.Zero] * m
    mom = []
    for i in range(m):
        expr = sp.S.Zero
        for j in range(m):
             expr += sp.diff(A[i, j], coords[j])
             for k in range(m):
                 expr += Gam3[i][j][k] * A[k, j] + Gam3[j][j][k] * A[i, k]
        mom.append(sp.simplify(expr - 8 * sp.pi * j_up[i]))
    return ConstraintResidual(ham, mom,
                                note="ADM constraints in geometric units G=c=1")


def evaluate_constraints(model: MetricModel, values: Mapping[str, float],
                           time_index: int = 0, rho: sp.Expr = sp.S.Zero,
                           tolerance: float = 1e-9) -> ConstraintResidual:
    data = decompose(model, time_index)
    res = constraints(data, rho)
    sub = model.substitution(values)
    try:
         hv = float(sp.N(res.hamiltonian.subs(sub), 30))
    except Exception:                                      # noqa: BLE001
         hv = None
    mv = []
    for mexpr in res.momentum:
         try:
              mv.append(float(sp.N(mexpr.subs(sub), 30)))
         except Exception:                                 # noqa: BLE001
              mv.append(float("nan"))
    res.hamiltonian_value = hv
    res.momentum_values = mv
    res.tolerance = tolerance
    ok = (hv is not None and abs(hv) <= tolerance
          and all(np.isfinite(v) and abs(v) <= tolerance for v in mv))
    res.satisfied = ok
    return res


# --------------------------------------------------------------------------
# Conformal (York) construction of regular initial data
# --------------------------------------------------------------------------

@dataclass
class YorkSolution:
    psi: np.ndarray
    grid: np.ndarray
    residual: float
    iterations: int
    converged: bool
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"n_grid": int(self.grid.size), "residual": self.residual,
                "iterations": self.iterations, "converged": self.converged,
                "psi_min": float(np.min(self.psi)),
                "psi_max": float(np.max(self.psi)), "note": self.note}


def york_hamiltonian_1d(rho_hat: np.ndarray, r: np.ndarray,
                        psi0: np.ndarray | None = None,
                        tol: float = 1e-12, maxiter: int = 200) -> YorkSolution:
    """Solve the spherically symmetric conformal Hamiltonian constraint.

    On maximal (``K = 0``), conformally flat data the constraint reduces to

        Laplacian(psi) + 2 pi rho_hat psi^{-3} = 0,

    a semilinear elliptic equation for the conformal factor. Solved here by
    Newton iteration on a second-order radial finite-difference stencil with
    a regular centre and ``psi -> 1`` at the outer boundary, which is exactly
    the "construct regular initial data" step the L12 gate requires and which
    v1.0 left unimplemented.
    """
    r = np.asarray(r, dtype=float)
    n = r.size
    dr = float(r[1] - r[0])
    psi = np.ones(n) if psi0 is None else np.asarray(psi0, dtype=float).copy()

    def residual(p: np.ndarray) -> np.ndarray:
        F = np.zeros(n)
        F[0] = p[1] - p[0]                                  # regular centre
        F[-1] = p[-1] - 1.0                                 # asymptotic flatness
        for i in range(1, n - 1):
            lap = ((p[i + 1] - 2 * p[i] + p[i - 1]) / dr ** 2
                   + (2.0 / r[i]) * (p[i + 1] - p[i - 1]) / (2 * dr))
            F[i] = lap + 2 * np.pi * rho_hat[i] * p[i] ** (-3)
        return F

    it = 0
    for it in range(1, maxiter + 1):
        F = residual(psi)
        if float(np.max(np.abs(F))) < tol:
             break
        J = np.zeros((n, n))
        J[0, 0], J[0, 1] = -1.0, 1.0
        J[-1, -1] = 1.0
        for i in range(1, n - 1):
             J[i, i - 1] = 1.0 / dr ** 2 - (1.0 / r[i]) / dr
             J[i, i] = -2.0 / dr ** 2 - 6 * np.pi * rho_hat[i] * psi[i] ** (-4)
             J[i, i + 1] = 1.0 / dr ** 2 + (1.0 / r[i]) / dr
        try:
             dpsi = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
             return YorkSolution(psi, r, float(np.max(np.abs(F))), it, False,
                                 "Newton Jacobian is singular")
        psi = psi + dpsi
        if np.any(psi <= 0):
             return YorkSolution(psi, r, float("nan"), it, False,
                                 "conformal factor went non-positive")
    F = residual(psi)
    res = float(np.max(np.abs(F)))
    return YorkSolution(psi, r, res, it, res < 1e-8,
                         "maximal, conformally flat spherically symmetric data")


def constructibility_report(model: MetricModel,
                             values: Mapping[str, float] | None = None
                             ) -> dict[str, Any]:
    """L12 report with the evolution gate honestly left open."""
    values = dict(values or model.default_point)
    out: dict[str, Any] = {"model_id": model.id}
    if not model.has_metric:
         out["status"] = "NOT_APPLICABLE"
         out["note"] = "no local metric; constructibility is an external claim"
         return out
    try:
         res = evaluate_constraints(model, values)
         out["constraints"] = res.as_dict()
         out["status"] = "CONSTRAINTS_EVALUATED"
    except Exception as exc:                              # noqa: BLE001
         out["status"] = "CONSTRAINTS_FAILED"
         out["error"] = str(exc)
    out["evolution"] = {
         "status": "NOT_ATTEMPTED",
         "requirement": ("a convergent 3+1 evolution (BSSN or Z4-family) with "
                         "resolution, gauge and formulation convergence tests, "
                         "constraint-violation monitoring and an explicit CTC "
                         "witness reconstructed on the converged geometry"),
         "note": ("CTC-FA v2.0 does not include a production numerical-relativity "
                  "evolution. The correct path is integration with the Einstein "
                  "Toolkit; until that exists no model may be reported above "
                   "claim level L5. A visualisation of tilted light cones is "
                   "explicitly NOT sufficient evidence.")}
    return out


__all__ = ["ADMData", "decompose", "ConstraintResidual", "constraints",
           "evaluate_constraints", "YorkSolution", "york_hamiltonian_1d",
           "constructibility_report"]
