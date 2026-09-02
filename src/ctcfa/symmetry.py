"""
CTC-FA v2.0   --   L4 : Killing / symmetry engine.

Two jobs.

1. **Admissibility of declared compact generators.** v1.0 allowed a registry
   entry to declare any coordinate direction as a "compact generator" and the
   detector consumed it without question. A direction that is not a Killing
   vector does not generate a one-parameter isometry group, so the orbit it
   traces is not the symmetry orbit the compact-orbit argument assumes, and
   the resulting "witness" is meaningless. v2.0 refuses a generator whose
   Killing residual is non-zero unless the registry record explicitly waives
   the requirement with a documented reason.

2. **Discovery of Killing vectors.** ``solve_killing`` solves the Killing
   equation for a supplied ansatz (constant, linear, or user-provided). For
   the flat and maximally symmetric benchmarks it recovers the full isometry
   algebra, which is used as a correctness test of the tensor engine.

Compact-generator Gram matrix
-----------------------------
For generators ``xi_A`` the Gram matrix is ``K_AB(x) = g_{mu nu} xi_A^mu
xi_B^nu`` and the causal character of the closed orbit with integer winding
``n^A`` is the sign of ``q(x, n) = n^A K_AB(x) n^B``. q < 0 is an explicit
symmetry-generated CTC; q = 0 is a closed null orbit; q > 0 clears only the
scanned sector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import sympy as sp

from .model import CompactDirection, GeneratorCertificate, MetricModel
from .symbolic import coordinate_killing_residual, lie_derivative_metric


# --------------------------------------------------------------------------
# Verification of declared generators
# --------------------------------------------------------------------------

def verify_compact_generators(model: MetricModel) -> list[GeneratorCertificate]:
    """Certify each declared compact generator (Killing + closure)."""
    out: list[GeneratorCertificate] = []
    if model.metric is None:
        for cd in model.compact:
            out.append(GeneratorCertificate(
                name=cd.name, is_killing=False, killing_residual="no local metric",
                closes=True,
                note="global/composite model: symmetry claim is external"))
        return out
    for cd in model.compact:
        res = coordinate_killing_residual(model.metric, model.coords, cd.index)
        is_k = bool(res.is_zero_matrix)
        closes = True                      # coordinate periodicity closes by construction
        note = cd.note
        if not is_k:
            note = (note + " | " if note else "") + \
                   "declared generator is NOT Killing for this metric"
        out.append(GeneratorCertificate(
            name=cd.name, is_killing=is_k,
            killing_residual="0" if is_k else str(res), closes=closes, note=note))
    return out


def generators_admissible(model: MetricModel) -> tuple[bool, list[GeneratorCertificate]]:
    certs = verify_compact_generators(model)
    ok = True
    for cd, c in zip(model.compact, certs):
        if cd.requires_killing and not c.admissible:
            ok = False
    return ok, certs


# --------------------------------------------------------------------------
# Gram matrix of the compact generators
# --------------------------------------------------------------------------

def compact_gram_symbolic(model: MetricModel) -> sp.Matrix:
    """K_AB = g(xi_A, xi_B) for the declared coordinate generators."""
    g = model.require_metric()
    idx = [c.index for c in model.compact]
    if not idx:
        return sp.zeros(0, 0)
    return g.extract(idx, idx)


def compact_gram_numeric(model: MetricModel,
                         subs: Mapping[str | sp.Symbol, float]) -> np.ndarray:
    K = compact_gram_symbolic(model)
    s = model.substitution(subs)
    return np.array(sp.Matrix(K).subs(s).evalf(), dtype=float)


def winding_norm_symbolic(model: MetricModel, winding: Sequence[int]) -> sp.Expr:
    """q(n) = n^A K_AB n^B as an exact expression."""
    K = compact_gram_symbolic(model)
    n = sp.Matrix([sp.Integer(w) for w in winding])
    if n.rows != K.rows:
        raise ValueError(f"winding vector length {n.rows} != number of compact "
                         f"generators {K.rows} for model {model.id}")
    return sp.expand((n.T * K * n)[0])


def orbit_length_symbolic(model: MetricModel, winding: Sequence[int]) -> sp.Expr:
    """Proper length/time of one closed symmetry orbit with winding ``n``.

    For commuting Killing generators with periods ``P_A`` the orbit
    ``lambda -> exp(lambda n^A P_A xi_A) . x`` has tangent
    ``n^A P_A xi_A`` and constant norm, so the (squared) invariant interval
    of one full circuit is ``sum_{A,B} n^A P_A n^B P_B K_AB``.
    """
    K = compact_gram_symbolic(model)
    v = sp.Matrix([sp.Integer(w) * cd.period
                   for w, cd in zip(winding, model.compact)])
    return sp.expand((v.T * K * v)[0])


# --------------------------------------------------------------------------
# Killing equation solver
# --------------------------------------------------------------------------

@dataclass
class KillingSolution:
    dimension: int
    basis: list[list[sp.Expr]] = field(default_factory=list)
    ansatz: str = ""
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"dimension": self.dimension, "ansatz": self.ansatz,
                "basis": [[str(c) for c in v] for v in self.basis],
                "note": self.note}


def solve_killing(model: MetricModel, ansatz: str = "linear") -> KillingSolution:
    """Solve ``L_xi g = 0`` for a polynomial ansatz in the coordinates.

    ``ansatz`` is ``"constant"`` (translations only) or ``"linear"``
    (translations plus linear terms; recovers the full Poincare algebra on
    Minkowski, dimension 10).
    """
    g = model.require_metric()
    coords = list(model.coords)
    n = len(coords)
    a = sp.symbols(f"a0:{n}", real=True)
    unknowns = list(a)
    xi: list[sp.Expr] = list(a)
    if ansatz == "linear":
        b = sp.symbols(f"b0:{n * n}", real=True)
        unknowns += list(b)
        xi = [a[i] + sum(b[i * n + j] * coords[j] for j in range(n)) for i in range(n)]
    elif ansatz != "constant":
        raise ValueError("ansatz must be 'constant' or 'linear'")

    L = lie_derivative_metric(g, coords, xi, simplify=True)
    eqs: list[sp.Expr] = []
    for i in range(n):
        for j in range(i, n):
            e = sp.expand(L[i, j])
            if e == 0:
                continue
            poly_eqs = _split_into_coefficient_equations(e, coords)
            eqs.extend(poly_eqs)
    eqs = [e for e in {sp.simplify(e) for e in eqs} if e != 0]
    if not eqs:
        basis = _basis_from_free(xi, unknowns, unknowns)
        return KillingSolution(len(basis), basis, ansatz,
                                "every ansatz coefficient is unconstrained")
    sol = sp.solve(eqs, unknowns, dict=True)
    if not sol:
        return KillingSolution(0, [], ansatz, "no solution in this ansatz")
    s = sol[0]
    xi_sol = [sp.expand(c.subs(s)) for c in xi]
    free = sorted({u for u in unknowns if u not in s} |
                  {sym for v in s.values() for sym in v.free_symbols
                   if sym in set(unknowns)}, key=str)
    basis = _basis_from_free(xi_sol, free, unknowns)
    return KillingSolution(len(basis), basis, ansatz,
                            f"{len(eqs)} independent Killing equations solved")


def _split_into_coefficient_equations(expr: sp.Expr,
                                       coords: Sequence[sp.Symbol]) -> list[sp.Expr]:
    """Turn ``sum_k c_k(coeffs) f_k(x) = 0`` into ``{c_k = 0}`` where possible."""
    try:
         p = sp.Poly(sp.expand(expr), *coords)
    except sp.PolynomialError:
         return [expr]
    return [sp.simplify(c) for c in p.coeffs()]


def _basis_from_free(xi: Sequence[sp.Expr], free: Sequence[sp.Symbol],
                     unknowns: Sequence[sp.Symbol]) -> list[list[sp.Expr]]:
    basis: list[list[sp.Expr]] = []
    for f in free:
        sub = {u: (1 if u == f else 0) for u in unknowns}
        v = [sp.expand(sp.simplify(c.subs(sub))) for c in xi]
        if any(c != 0 for c in v):
            basis.append(v)
    return basis


__all__ = ["verify_compact_generators", "generators_admissible",
           "compact_gram_symbolic", "compact_gram_numeric",
           "winding_norm_symbolic", "orbit_length_symbolic",
           "KillingSolution", "solve_killing"]
