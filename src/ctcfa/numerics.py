"""
CTC-FA v2.0 -- Numerical precision, tolerance and uncertainty policy
(document section 29), implemented rather than described.

Rules enforced
--------------
N1 No unqualified sign decision near zero. Every causal sign test goes
    through :func:`classify_sign`, which takes an *uncertainty budget* and
    returns ``NEGATIVE``, ``MARGINAL`` or ``POSITIVE``. A raw value of
    ``-1e-15`` with a budget of ``1e-12`` is ``MARGINAL``, never a CTC.
N2 Automatic precision escalation. When ``|q| <= eps_total`` the quantity is
    recomputed symbolically at 50 significant digits with mpmath before the
    verdict is issued. v1.0 named this policy (FM-017) but the kernel had
    no escalation path.
N3 Uncertainty budgets are *composed*, not guessed: :class:`ErrorBudget`
    accumulates round-off, substitution, conditioning and (where used)
    integration and interpolation terms.
N4 A reported threshold always carries its final sign-changing bracket
    (section 29.3), never only the root-finder tolerance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import sympy as sp
from mpmath import mp

DEFAULT_TOL = 1e-10
HIGH_PRECISION_DIGITS = 50


class SignClass(str, Enum):
    NEGATIVE = "NEGATIVE"
    MARGINAL = "MARGINAL"
    POSITIVE = "POSITIVE"


@dataclass
class ErrorBudget:
    """Composable uncertainty budget for one scalar decision."""

    roundoff: float = 0.0
    substitution: float = 0.0
    conditioning: float = 0.0
    integration: float = 0.0
    interpolation: float = 0.0
    optimization: float = 0.0
    parameter: float = 0.0
    floor: float = 1e-14
    notes: list[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        return max(self.floor,
                   self.roundoff + self.substitution + self.conditioning
                   + self.integration + self.interpolation + self.optimization
                   + self.parameter)

    def add(self, **kw: float) -> "ErrorBudget":
        for k, v in kw.items():
            if not hasattr(self, k):
                raise KeyError(f"unknown budget term {k!r}")
            setattr(self, k, getattr(self, k) + float(v))
        return self

    def as_dict(self) -> dict[str, Any]:
        return {"roundoff": self.roundoff, "substitution": self.substitution,
                "conditioning": self.conditioning, "integration": self.integration,
                "interpolation": self.interpolation, "optimization": self.optimization,
                "parameter": self.parameter, "total": self.total,
                "notes": list(self.notes)}


def matrix_conditioning(A: np.ndarray) -> float:
    """A cheap conditioning proxy used to inflate the round-off term."""
    try:
         c = float(np.linalg.cond(A))
    except Exception:                                    # noqa: BLE001
        return float("inf")
    if not math.isfinite(c):
        return float("inf")
    return c


def budget_for_matrix(A: np.ndarray, scale: float = 1.0) -> ErrorBudget:
    """Round-off budget for a quantity built from the matrix ``A``."""
    eps = float(np.finfo(np.float64).eps)
    cond = matrix_conditioning(A)
    b = ErrorBudget()
    nrm = float(np.max(np.abs(A))) if A.size else 1.0
    b.roundoff = eps * max(1.0, nrm) * max(1.0, abs(scale)) * A.shape[0]
    if math.isfinite(cond):
        b.conditioning = eps * cond * max(1.0, nrm)
        b.notes.append(f"cond={cond:.3e}")
    else:
        b.conditioning = float("inf")
        b.notes.append("cond=inf (singular or near-singular metric)")
    return b


def classify_sign(value: float, budget: ErrorBudget | float) -> SignClass:
    """Sign decision with an explicit uncertainty band (rule N1)."""
    eps = budget.total if isinstance(budget, ErrorBudget) else float(budget)
    if not math.isfinite(value):
        return SignClass.MARGINAL
    if value < -eps:
        return SignClass.NEGATIVE
    if value > eps:
        return SignClass.POSITIVE
    return SignClass.MARGINAL


def high_precision_value(expr: sp.Expr, subs: Mapping[sp.Symbol, float],
                           digits: int = HIGH_PRECISION_DIGITS) -> float:
    """Re-evaluate a symbolic expression at high precision (rule N2)."""
    old = mp.dps
    try:
         mp.dps = digits
         rational = {k: sp.nsimplify(sp.Float(v, digits), rational=True)
                      for k, v in subs.items()}
         val = sp.N(expr.subs(rational), digits)
         return float(val)
    finally:
         mp.dps = old


def escalate_if_marginal(expr: sp.Expr, subs: Mapping[sp.Symbol, float],
                         value: float, budget: ErrorBudget,
                         digits: int = HIGH_PRECISION_DIGITS
                         ) -> tuple[float, ErrorBudget, bool]:
    """If ``value`` is inside the uncertainty band, recompute exactly.

    Returns ``(value, budget, escalated)``. On escalation the round-off and
    conditioning terms are replaced by the high-precision floor, because the
    quantity no longer passes through double-precision linear algebra.
    """
    if classify_sign(value, budget) is not SignClass.MARGINAL:
         return value, budget, False
    try:
         hp = high_precision_value(expr, subs, digits)
    except Exception as exc:                              # noqa: BLE001
         budget.notes.append(f"escalation failed: {exc}")
         return value, budget, False
    nb = ErrorBudget(floor=10.0 ** (-(digits - 6)))
    nb.notes = list(budget.notes) + [f"escalated to {digits} digits"]
    nb.parameter = budget.parameter
    return hp, nb, True


# --------------------------------------------------------------------------
# Signature
# --------------------------------------------------------------------------

def signature_counts(g: np.ndarray, tol: float = 1e-10) -> tuple[int, int, int]:
    """Return ``(n_negative, n_null, n_positive)`` by Sylvester's law.

    The *counts* of eigenvalue signs of a real symmetric matrix are invariant
    under congruence, so this is a coordinate-independent statement even
    though the eigenvalues themselves are not.
    """
    vals = np.linalg.eigvalsh((g + g.T) / 2.0)
    scale = max(1.0, float(np.max(np.abs(vals))))
    t = tol * scale
    return (int(np.sum(vals < -t)), int(np.sum(np.abs(vals) <= t)),
            int(np.sum(vals > t)))


def is_lorentzian(g: np.ndarray, tol: float = 1e-10) -> bool:
    n = g.shape[0]
    return signature_counts(g, tol) == (1, 0, n - 1)


# --------------------------------------------------------------------------
# Bracketed root finding with a reported bracket (rule N4)
# --------------------------------------------------------------------------

@dataclass
class ThresholdResult:
    """A causal threshold together with everything section 29.3 requires."""

    value: float
    bracket: tuple[float, float]
    left_margin: float
    right_margin: float
    residual: float
    iterations: int
    method: str
    analytic_value: float | None = None
    analytic_form: str = ""
    domain_ok: bool = True
    note: str = ""

    @property
    def bracket_width(self) -> float:
        return abs(self.bracket[1] - self.bracket[0])

    @property
    def analytic_error(self) -> float | None:
        if self.analytic_value is None:
            return None
        return abs(self.value - self.analytic_value)

    def as_dict(self) -> dict[str, Any]:
        return {"value": self.value, "bracket": list(self.bracket),
                "bracket_width": self.bracket_width,
                "left_margin": self.left_margin, "right_margin": self.right_margin,
                "residual": self.residual, "iterations": self.iterations,
                "method": self.method, "analytic_value": self.analytic_value,
                "analytic_form": self.analytic_form,
                "analytic_error": self.analytic_error,
                "domain_ok": self.domain_ok, "note": self.note}


def bracketed_bisection(f: Callable[[float], float], lo: float, hi: float,
                         xtol: float = 1e-13, maxiter: int = 400
                         ) -> tuple[float, tuple[float, float], int]:
    """Plain bisection that *returns the final bracket*, not only the root."""
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo, (lo, lo), 0
    if fhi == 0.0:
        return hi, (hi, hi), 0
    if flo * fhi > 0:
        raise ValueError(f"threshold is not bracketed on [{lo}, {hi}]: "
                          f"f(lo)={flo:.6e}, f(hi)={fhi:.6e}")
    it = 0
    while (hi - lo) > xtol and it < maxiter:
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm == 0.0:
            return mid, (mid, mid), it
        if flo * fm < 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
        it += 1
    root = 0.5 * (lo + hi)
    return root, (lo, hi), it


def coarse_sign_scan(f: Callable[[float], float], lo: float, hi: float,
                      n: int = 64) -> list[tuple[float, float]]:
    """Return every sub-interval on which ``f`` changes sign."""
    xs = np.linspace(lo, hi, n + 1)
    vals = []
    for x in xs:
        try:
             vals.append(float(f(float(x))))
        except Exception:                                 # noqa: BLE001
             vals.append(float("nan"))
    out = []
    for i in range(n):
        a, b = vals[i], vals[i + 1]
        if not (math.isfinite(a) and math.isfinite(b)):
             continue
        if a == 0.0:
             out.append((float(xs[i]), float(xs[i])))
        elif a * b < 0:
             out.append((float(xs[i]), float(xs[i + 1])))
    return out


__all__ = [
    "DEFAULT_TOL", "HIGH_PRECISION_DIGITS", "SignClass", "ErrorBudget",
    "budget_for_matrix", "matrix_conditioning", "classify_sign",
    "high_precision_value", "escalate_if_marginal", "signature_counts",
    "is_lorentzian", "ThresholdResult", "bracketed_bisection", "coarse_sign_scan",
]
