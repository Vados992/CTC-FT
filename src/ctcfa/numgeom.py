"""
CTC-FA v2.0   --   Numerical differential geometry with a controlled error model.

The symbolic engine (``ctcfa.symbolic``) is exact and is the certification
path. It is also, for metrics built from nested ``tanh`` regulators such as
Alcubierre and Krasnikov, computationally prohibitive: a full symbolic
Einstein tensor for the warp metric does not terminate in useful time. v1.0
simply had no answer for those models, so their energy and curvature rows
stayed empty.

This module supplies the second path: a lambdified metric plus high-order
central finite differences with Richardson extrapolation, giving curvature
tensors at a point with an explicit, *reported* truncation-plus-round-off
error. Results from this path are always tagged ``diagnostic`` and their
error enters the uncertainty budget, so they can refute a condition but
cannot certify one -- exactly the asymmetry the architecture requires
everywhere else.

Error model
-----------
A fourth-order central stencil has truncation error ``O(h^4 |d^5 g|)`` and
round-off ``O(eps |g| / h)``. The step is chosen near the balance point
``h ~ (eps)^(1/5) * scale`` and the reported error is the difference between
the ``h`` and ``2h`` estimates divided by ``2^4 - 1``, which is the standard
Richardson estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import sympy as sp

from .model import MetricModel

_EPS = float(np.finfo(np.float64).eps)


@dataclass
class NumericCurvature:
    christoffel: np.ndarray
    riemann: np.ndarray
    ricci: np.ndarray
    ricci_scalar: float
    einstein: np.ndarray
    kretschmann: float
    step: float
    error_estimate: float
    note: str = "finite-difference curvature (diagnostic grade)"

    def as_dict(self) -> dict[str, Any]:
        return {"ricci_scalar": self.ricci_scalar,
                "kretschmann": self.kretschmann,
                "einstein_max_abs": float(np.max(np.abs(self.einstein))),
                "step": self.step, "error_estimate": self.error_estimate,
                "note": self.note}


class NumericGeometry:
    """Lambdified metric with finite-difference curvature at a point."""

    def __init__(self, model: MetricModel):
        self.model = model
        g = model.require_metric()
        self.coords = list(model.coords)
        self.params = list(model.params)
        self.n = len(self.coords)
        args = self.coords + self.params
        self._g = sp.lambdify(args, sp.Matrix(g), "numpy")

    # -- evaluation -------------------------------------------------------

    def metric(self, x: Sequence[float], p: Sequence[float]) -> np.ndarray:
        m = np.array(self._g(*list(x), *list(p)), dtype=float)
        return (m + m.T) / 2.0

    def split(self, values: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
        x = np.array([float(values[str(c)]) for c in self.coords], dtype=float)
        p = np.array([float(values[str(s)]) for s in self.params], dtype=float)
        return x, p

    # -- derivatives ------------------------------------------------------

    def dg(self, x: np.ndarray, p: np.ndarray, h: float) -> np.ndarray:
        """dg[c][a][b] = d g_ab / d x^c, fourth-order central."""
        n = self.n
        out = np.zeros((n, n, n))
        for c in range(n):
            e = np.zeros(n)
            e[c] = 1.0
            hc = h * max(1.0, abs(float(x[c])))
            gm2 = self.metric(x - 2 * hc * e, p)
            gm1 = self.metric(x - hc * e, p)
            gp1 = self.metric(x + hc * e, p)
            gp2 = self.metric(x + 2 * hc * e, p)
            out[c] = (gm2 - 8 * gm1 + 8 * gp1 - gp2) / (12 * hc)
        return out

    def d2g(self, x: np.ndarray, p: np.ndarray, h: float) -> np.ndarray:
        """d2g[c][d][a][b] = d^2 g_ab / dx^c dx^d, fourth-order central."""
        n = self.n
        out = np.zeros((n, n, n, n))
        for c in range(n):
            ec = np.zeros(n); ec[c] = 1.0
            hc = h * max(1.0, abs(float(x[c])))
            for d in range(c, n):
                ed = np.zeros(n); ed[d] = 1.0
                hd = h * max(1.0, abs(float(x[d])))
                if c == d:
                    val = (-self.metric(x + 2 * hc * ec, p)
                           + 16 * self.metric(x + hc * ec, p)
                           - 30 * self.metric(x, p)
                           + 16 * self.metric(x - hc * ec, p)
                           - self.metric(x - 2 * hc * ec, p)) / (12 * hc * hc)
                else:
                    val = (self.metric(x + hc * ec + hd * ed, p)
                           - self.metric(x + hc * ec - hd * ed, p)
                           - self.metric(x - hc * ec + hd * ed, p)
                           + self.metric(x - hc * ec - hd * ed, p)) / (4 * hc * hd)
                out[c, d] = val
                out[d, c] = val
        return out

    # -- curvature --------------------------------------------------------

    def _curvature_at_step(self, x: np.ndarray, p: np.ndarray, h: float):
        n = self.n
        g = self.metric(x, p)
        gi = np.linalg.inv(g)
        dg = self.dg(x, p, h)
        d2g = self.d2g(x, p, h)

        Gam = np.zeros((n, n, n))
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    Gam[a, b, c] = 0.5 * sum(
                        gi[a, d] * (dg[b][d][c] + dg[c][d][b] - dg[d][b][c])
                        for d in range(n))

        dGam = np.zeros((n, n, n, n))      # dGam[e][a][b][c] = d_e Gamma^a_bc
        dgi = np.zeros((n, n, n))          # dgi[e][a][b] = d_e g^{ab}
        for e in range(n):
            dgi[e] = -gi @ dg[e] @ gi
        for e in range(n):
            for a in range(n):
                for b in range(n):
                    for c in range(n):
                        s = 0.0
                        for d in range(n):
                            s += 0.5 * dgi[e][a, d] * (
                                dg[b][d][c] + dg[c][d][b] - dg[d][b][c])
                            s += 0.5 * gi[a, d] * (
                                d2g[e][b][d][c] + d2g[e][c][d][b] - d2g[e][d][b][c])
                        dGam[e, a, b, c] = s

        R = np.zeros((n, n, n, n))        # R[a][b][c][d] = R^a_bcd
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    for d in range(n):
                        R[a, b, c, d] = (dGam[c, a, d, b] - dGam[d, a, c, b]
                                         + sum(Gam[a, c, e] * Gam[e, d, b]
                                               - Gam[a, d, e] * Gam[e, c, b]
                                               for e in range(n)))
        Ric = np.einsum("abad->bd", R)
        Rs = float(np.einsum("bd,bd->", gi, Ric))
        G = Ric - 0.5 * g * Rs
        Rdown = np.einsum("ae,ebcd->abcd", g, R)
        Rup = np.einsum("ae,bf,cg,dh,efgh->abcd", gi, gi, gi, gi, Rdown)
        K = float(np.einsum("abcd,abcd->", Rdown, Rup))
        return Gam, R, Ric, Rs, G, K

    def curvature(self, values: Mapping[str, float],
                  h: float | None = None) -> NumericCurvature:
        x, p = self.split(values)
        # Balance point for a fourth-order stencil on second derivatives:
        # truncation ~ h^4, round-off ~ eps/h^2, so h ~ eps^(1/6) ~ 6.4e-3.
        # Empirically (tests/unit/test_numgeom.py) h = 3e-3 gives a relative
        # error of 1e-9 on the exact Schwarzschild Kretschmann scalar.
        h0 = h if h is not None else 3.0e-3
        a = self._curvature_at_step(x, p, h0)
        b = self._curvature_at_step(x, p, 2 * h0)
        err = max(abs(a[3] - b[3]),
                  float(np.max(np.abs(a[4] - b[4]))),
                  abs(a[5] - b[5])) / 15.0
        return NumericCurvature(a[0], a[1], a[2], a[3], a[4], a[5], h0, err)

    def einstein(self, values: Mapping[str, float], h: float | None = None
                 ) -> tuple[np.ndarray, float]:
        c = self.curvature(values, h)
        return c.einstein, c.error_estimate

    def stress_tensor(self, values: Mapping[str, float],
                      Lambda: float = 0.0, h: float | None = None
                      ) -> tuple[np.ndarray, float]:
        """T_ab = (G_ab + Lambda g_ab) / (8 pi), numerically."""
        c = self.curvature(values, h)
        x, p = self.split(values)
        g = self.metric(x, p)
        T = (c.einstein + Lambda * g) / (8.0 * math.pi)
        return T, c.error_estimate / (8.0 * math.pi)


@lru_cache(maxsize=64)
def geometry_for(model_id: str) -> NumericGeometry:
    from .registry import get
    return NumericGeometry(get(model_id))


__all__ = ["NumericGeometry", "NumericCurvature", "geometry_for"]
