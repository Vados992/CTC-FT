"""
CTC-FA v2.0   --   L1 : Symbolic tensor engine.

Everything here is exact computer algebra in the (-,+,+,+) convention with
G = c = 1. The conventions are fixed once, here, and every downstream layer
is required to use them (v1.0 FM-019: "published formula copied with
incompatible convention").

Conventions
-----------
Christoffel        Gamma^a_bc = 1/2 g^{ad} ( d_b g_{dc} + d_c g_{db} - d_d g_{bc} )
Riemann            R^a_{bcd} = d_c Gamma^a_{db} - d_d Gamma^a_{cb}
                               + Gamma^a_{ce} Gamma^e_{db} - Gamma^a_{de} Gamma^e_{cb}
Ricci              R_{bd} = R^a_{bad}
Ricci scalar       R = g^{bd} R_{bd}
Einstein           G_{ab} = R_{ab} - 1/2 g_{ab} R
Field equation     G_{ab} + Lambda g_{ab} = 8 pi T_{ab}
Kretschmann        K = R_{abcd} R^{abcd}

With these conventions the Schwarzschild solution has R_{ab} = 0 and
K = 48 M^2 / r^6, and de Sitter with Lambda > 0 has R = 4 Lambda. Both are
asserted in ``tests/unit/test_symbolic.py``, so a convention drift is a test
failure rather than a silent sign error.

v1.0 defect closed here
-----------------------
D-09 ``kretschmann_placeholder`` raised ``NotImplementedError``. Since the
      Kretschmann scalar is the *only* tool in the architecture that can
      separate a coordinate singularity from a curvature singularity, its
      absence left failure mode FM-001 unmitigated in code. It is now
      implemented and benchmarked.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

import sympy as sp

_SIMPLIFY_BUDGET = 4000


def _s(expr: sp.Expr, simplify: bool = True) -> sp.Expr:
    """Simplify with a size guard so pathological metrics cannot hang a run."""
    if not simplify:
         return expr
    try:
         if sp.count_ops(expr) > _SIMPLIFY_BUDGET:
             return sp.cancel(sp.together(expr))
         return sp.simplify(expr)
    except Exception:                                    # noqa: BLE001
         return expr


# --------------------------------------------------------------------------
# Basic algebra
# --------------------------------------------------------------------------

def inverse_metric(g: sp.Matrix, simplify: bool = True) -> sp.Matrix:
    gi = g.inv()
    return gi.applyfunc(lambda e: _s(e, simplify)) if simplify else gi


def check_inverse(g: sp.Matrix, gi: sp.Matrix | None = None) -> sp.Matrix:
    """Return ``g * g^{-1} - I``; the zero matrix certifies gate V1."""
    gi = inverse_metric(g) if gi is None else gi
    return sp.simplify(g * gi - sp.eye(g.shape[0]))


def determinant(g: sp.Matrix, simplify: bool = True) -> sp.Expr:
    return _s(g.det(), simplify)


# --------------------------------------------------------------------------
# Connection and curvature
# --------------------------------------------------------------------------

def christoffel(g: sp.Matrix, coords: Sequence[sp.Symbol],
                simplify: bool = True, gi: sp.Matrix | None = None) -> list[list[list[sp.Expr]]]:
    """Gamma[a][b][c] = Gamma^a_{bc}."""
    n = len(coords)
    gi = inverse_metric(g, simplify=False) if gi is None else gi
    dg = [[[sp.diff(g[b, c], coords[d]) for c in range(n)] for b in range(n)]
          for d in range(n)]
    out = [[[sp.S.Zero] * n for _ in range(n)] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            for c in range(b, n):
                acc = sp.S.Zero
                for d in range(n):
                    if gi[a, d] == 0:
                        continue
                    acc += gi[a, d] * (dg[b][d][c] + dg[c][d][b] - dg[d][b][c])
                val = _s(acc / 2, simplify)
                out[a][b][c] = val
                out[a][c][b] = val
    return out


def riemann(g: sp.Matrix, coords: Sequence[sp.Symbol], simplify: bool = True,
             Gamma: list | None = None) -> list[list[list[list[sp.Expr]]]]:
    """R[a][b][c][d] = R^a_{bcd}."""
    n = len(coords)
    Gamma = christoffel(g, coords, simplify=False) if Gamma is None else Gamma
    R = [[[[sp.S.Zero] * n for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for a in range(n):
        for b in range(n):
             for c in range(n):
                 for d in range(c + 1, n):
                     expr = (sp.diff(Gamma[a][d][b], coords[c])
                             - sp.diff(Gamma[a][c][b], coords[d]))
                     for e in range(n):
                         expr += (Gamma[a][c][e] * Gamma[e][d][b]
                                  - Gamma[a][d][e] * Gamma[e][c][b])
                     val = _s(expr, simplify)
                     R[a][b][c][d] = val
                     R[a][b][d][c] = -val
    return R


def riemann_lower(g: sp.Matrix, coords: Sequence[sp.Symbol], simplify: bool = True,
                  R: list | None = None) -> list[list[list[list[sp.Expr]]]]:
    """R_{abcd} = g_{ae} R^e_{bcd}."""
    n = len(coords)
    R = riemann(g, coords, simplify=False) if R is None else R
    out = [[[[sp.S.Zero] * n for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(c + 1, n):
                    expr = sum(g[a, e] * R[e][b][c][d] for e in range(n))
                    val = _s(expr, simplify)
                    out[a][b][c][d] = val
                    out[a][b][d][c] = -val
    return out


def ricci_tensor(g: sp.Matrix, coords: Sequence[sp.Symbol], simplify: bool = True,
                 Gamma: list | None = None) -> sp.Matrix:
    """R_{bd} = R^a_{bad}, computed directly (cheaper than contracting Riemann)."""
    n = len(coords)
    Gamma = christoffel(g, coords, simplify=False) if Gamma is None else Gamma
    Ric = sp.zeros(n, n)
    for b in range(n):
        for d in range(b, n):
            expr = sp.S.Zero
            for a in range(n):
                expr += sp.diff(Gamma[a][d][b], coords[a]) - sp.diff(Gamma[a][a][b], coords[d])
                for e in range(n):
                    expr += Gamma[a][a][e] * Gamma[e][d][b] - Gamma[a][d][e] * Gamma[e][a][b]
            val = _s(expr, simplify)
            Ric[b, d] = val
            Ric[d, b] = val
    return Ric


def ricci_scalar(g: sp.Matrix, coords: Sequence[sp.Symbol], simplify: bool = True,
                 Ric: sp.Matrix | None = None, gi: sp.Matrix | None = None) -> sp.Expr:
    n = len(coords)
    gi = inverse_metric(g, simplify=False) if gi is None else gi
    Ric = ricci_tensor(g, coords, simplify=False) if Ric is None else Ric
    return _s(sum(gi[a, b] * Ric[a, b] for a in range(n) for b in range(n)), simplify)


def einstein_tensor(g: sp.Matrix, coords: Sequence[sp.Symbol],
                    simplify: bool = True) -> sp.Matrix:
    gi = inverse_metric(g, simplify=False)
    Gamma = christoffel(g, coords, simplify=False, gi=gi)
    Ric = ricci_tensor(g, coords, simplify=False, Gamma=Gamma)
    R = ricci_scalar(g, coords, simplify=False, Ric=Ric, gi=gi)
    G = Ric - sp.Rational(1, 2) * g * R
    return G.applyfunc(lambda e: _s(e, simplify))


def kretschmann(g: sp.Matrix, coords: Sequence[sp.Symbol],
                simplify: bool = True) -> sp.Expr:
    """K = R_{abcd} R^{abcd}.

    Uses the antisymmetry pairs to cut the work by 4 and the pair-exchange
    symmetry where cheap. Exact, not a placeholder.
    """
    n = len(coords)
    gi = inverse_metric(g, simplify=False)
    Gamma = christoffel(g, coords, simplify=False, gi=gi)
    Rud = riemann(g, coords, simplify=False, Gamma=Gamma)
    Rdd = riemann_lower(g, coords, simplify=False, R=Rud)

    # Raise all four indices once: R^{abcd} = g^{ae} g^{bf} g^{cg} g^{dh} R_{efgh}
    Ruu = [[[[sp.S.Zero] * n for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(c + 1, n):
                    expr = sp.S.Zero
                    for e in range(n):
                        if gi[a, e] == 0:
                            continue
                        for f in range(n):
                            if gi[b, f] == 0:
                                continue
                            for gg in range(n):
                                if gi[c, gg] == 0:
                                     continue
                                for h in range(n):
                                     if gi[d, h] == 0:
                                         continue
                                     term = Rdd[e][f][gg][h]
                                     if term == 0:
                                         continue
                                     expr += gi[a, e] * gi[b, f] * gi[c, gg] * gi[d, h] * term
                    Ruu[a][b][c][d] = expr
                    Ruu[a][b][d][c] = -expr

    total = sp.S.Zero
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(c + 1, n):
                    if Rdd[a][b][c][d] == 0 or Ruu[a][b][c][d] == 0:
                        continue
                    total += 2 * Rdd[a][b][c][d] * Ruu[a][b][c][d]
    return _s(total, simplify)


def field_equation_residual(g: sp.Matrix, coords: Sequence[sp.Symbol],
                            T: sp.Matrix | None = None,
                            Lambda: sp.Expr = sp.S.Zero,
                            simplify: bool = True) -> sp.Matrix:
    """E_{ab} = G_{ab} + Lambda g_{ab} - 8 pi T_{ab}."""
    n = len(coords)
    G = einstein_tensor(g, coords, simplify=simplify)
    rhs = sp.zeros(n, n) if T is None else 8 * sp.pi * T
    res = G + Lambda * g - rhs
    return res.applyfunc(lambda e: _s(e, simplify))


# --------------------------------------------------------------------------
# Symmetry (L4)
# --------------------------------------------------------------------------

def lie_derivative_metric(g: sp.Matrix, coords: Sequence[sp.Symbol],
                          xi: Sequence[sp.Expr], simplify: bool = True) -> sp.Matrix:
    """(L_xi g)_{ab} = xi^c d_c g_{ab} + g_{cb} d_a xi^c + g_{ac} d_b xi^c.

    A vanishing result certifies that ``xi`` is a Killing vector. v1.0
    declared compact generators but never verified this, so nothing stopped
    a registry entry from naming a non-Killing coordinate direction as a
    "symmetry generator" (v2.0 defect D-04).
    """
    n = len(coords)
    out = sp.zeros(n, n)
    for a in range(n):
        for b in range(a, n):
            expr = sum(xi[c] * sp.diff(g[a, b], coords[c]) for c in range(n))
            expr += sum(g[c, b] * sp.diff(xi[c], coords[a]) for c in range(n))
            expr += sum(g[a, c] * sp.diff(xi[c], coords[b]) for c in range(n))
            val = _s(expr, simplify)
            out[a, b] = val
            out[b, a] = val
    return out


def coordinate_killing_residual(g: sp.Matrix, coords: Sequence[sp.Symbol],
                                index: int, simplify: bool = True) -> sp.Matrix:
    """Residual of the Killing equation for xi = d/dx^index.

    For a coordinate vector this reduces to ``d g_{ab} / d x^index``.
    """
    xi = [sp.S.One if c == index else sp.S.Zero for c in range(len(coords))]
    return lie_derivative_metric(g, coords, xi, simplify=simplify)


def killing_equation_system(g: sp.Matrix, coords: Sequence[sp.Symbol],
                            ansatz: Sequence[sp.Expr]) -> list[sp.Expr]:
    """Return the independent components of ``L_xi g = 0`` for a given ansatz.

    The caller supplies ``xi^a`` containing free symbols; the returned list is
    fed to ``sympy.solve`` by :func:`ctcfa.symmetry.solve_killing`.
    """
    L = lie_derivative_metric(g, coords, ansatz, simplify=True)
    n = len(coords)
    return [L[a, b] for a in range(n) for b in range(a, n) if L[a, b] != 0]


# --------------------------------------------------------------------------
# Frames and invariants
# --------------------------------------------------------------------------

def signature_symbolic(g: sp.Matrix) -> str:
    """Best-effort symbolic signature string; numeric checks live in L2."""
    try:
         return str(sp.simplify(g.det()))
    except Exception:                                    # noqa: BLE001
         return "undetermined"


def norm2(g: sp.Matrix, v: Sequence[sp.Expr]) -> sp.Expr:
    n = g.shape[0]
    return sp.simplify(sum(g[a, b] * v[a] * v[b] for a in range(n) for b in range(n)))


__all__ = [
    "inverse_metric", "check_inverse", "determinant", "christoffel", "riemann",
    "riemann_lower", "ricci_tensor", "ricci_scalar", "einstein_tensor",
    "kretschmann", "field_equation_residual", "lie_derivative_metric",
    "coordinate_killing_residual", "killing_equation_system", "norm2",
]
