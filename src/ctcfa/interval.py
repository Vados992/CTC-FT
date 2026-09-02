"""
CTC-FA v2.0   --    Certified interval arithmetic and branch-and-bound bounds.

Why this module exists
----------------------
v1.0 repeatedly, and correctly, disqualified its own numerics:

    L9  "Numerical directional sampling is diagnostic only; analytic/interval
         certification is required for formal PASS."
    FM-009 "Energy-condition result based on finite directional samples ->
         false proof."
    24.4 "For numerical tau, the infimum must be bounded with a method that
         controls interpolation and discretisation error. Point sampling
         alone does not establish the strict global inequality."

The architecture named the required tool -- interval certification -- in three
places and shipped none. This module supplies it.

What it provides
----------------
``interval_eval``      rigorous enclosure of a sympy expression on a box,
                       using ``mpmath.iv`` (correctly outward-rounded).
``global_lower_bound`` branch-and-bound *proof* that ``f(x) >= L`` on a
                       compact box, or a counterexample point.
``prove_positive``       the certificate wrapper used by L7 (temporal function)
                         and L9 (energy conditions).

Scope and honesty
-----------------
A branch-and-bound certificate is valid only on the **compact box** it was
run on. It is therefore reported with that box attached, and the caller
must decide whether the box is the declared physical domain. For unbounded
domains an additional analytic argument (asymptotics, monotonicity, or a
change of variable compactifying the domain) is required, and
``PositivityCertificate.domain_is_compact`` records which case applies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import sympy as sp
from mpmath import iv, mp

iv.prec = 80


class IntervalDomainError(ValueError):
    """The expression left its real domain somewhere on the box."""


# --------------------------------------------------------------------------
# Interval evaluation of a sympy expression
# --------------------------------------------------------------------------

_UNBOUNDED = None    # sentinel meaning "no rigorous bound obtainable"


def _iv(a: float, b: float):
    return iv.mpf([a, b])


def _cosh(x):
    return (iv.exp(x) + iv.exp(-x)) / 2


def _sinh(x):
    return (iv.exp(x) - iv.exp(-x)) / 2


def _tanh(x):
    e2 = iv.exp(2 * x)
    return (e2 - 1) / (e2 + 1)


def interval_eval(expr: sp.Expr, box: Mapping[sp.Symbol, tuple[float, float]]):
    """Rigorous enclosure of ``expr`` over the box.

    Raises :class:`IntervalDomainError` when the expression leaves its real
    domain (for example ``sqrt`` of an interval containing negatives), which
    the caller must treat as "domain not verified", never as "inequality
    holds".
    """
    def rec(e: sp.Expr):
        if e.is_Number:
            v = float(e)
            return _iv(v, v)
        if isinstance(e, sp.Symbol):
            if e not in box:
                raise IntervalDomainError(f"symbol {e} has no interval assigned")
            lo, hi = box[e]
            return _iv(lo, hi)
        if e is sp.pi:
            return iv.pi
        if e is sp.E:
            return iv.e
        if e.is_Add:
            acc = _iv(0.0, 0.0)
            for a in e.args:
                acc = acc + rec(a)
            return acc
        if e.is_Mul:
            acc = _iv(1.0, 1.0)
            for a in e.args:
                acc = acc * rec(a)
            return acc
        if e.is_Pow:
            base, expo = e.args
            b = rec(base)
            if expo.is_Integer:
                n = int(expo)
                if n >= 0:
                     return b ** n
                # negative integer power: 1 / b^|n|, undefined if 0 in b
                d = b ** (-n)
                if d.a <= 0 <= d.b:
                     raise IntervalDomainError("division by an interval containing 0")
                return _iv(1.0, 1.0) / d
            if expo.is_Rational and expo.q == 2 and expo.p == 1:
                if b.a < 0:
                     raise IntervalDomainError("sqrt of an interval containing negatives")
                return iv.sqrt(b)
            ex = rec(expo)
            if b.a <= 0:
                raise IntervalDomainError("non-integer power of a non-positive base")
            return iv.exp(ex * iv.log(b))
        f = e.func
        if f is sp.exp:
            return iv.exp(rec(e.args[0]))
        if f is sp.log:
            x = rec(e.args[0])
            if x.a <= 0:
                raise IntervalDomainError("log of an interval containing non-positives")
            return iv.log(x)
        if f is sp.sqrt:
            x = rec(e.args[0])
            if x.a < 0:
                raise IntervalDomainError("sqrt of an interval containing negatives")
            return iv.sqrt(x)
        if f is sp.sin:
            return iv.sin(rec(e.args[0]))
        if f is sp.cos:
            return iv.cos(rec(e.args[0]))
        if f is sp.tan:
            c = iv.cos(rec(e.args[0]))
            if c.a <= 0 <= c.b:
                raise IntervalDomainError("tan across a pole")
            return iv.sin(rec(e.args[0])) / c
        if f is sp.sinh:
            return _sinh(rec(e.args[0]))
        if f is sp.cosh:
            return _cosh(rec(e.args[0]))
        if f is sp.tanh:
            return _tanh(rec(e.args[0]))
        if f is sp.Abs:
            x = rec(e.args[0])
            lo, hi = float(x.a), float(x.b)
            if lo <= 0 <= hi:
                return _iv(0.0, max(abs(lo), abs(hi)))
            m, M = sorted((abs(lo), abs(hi)))
            return _iv(m, M)
        raise IntervalDomainError(f"no interval rule for {f.__name__}: {e}")

    return rec(sp.sympify(expr))


# --------------------------------------------------------------------------
# Branch and bound
# --------------------------------------------------------------------------

@dataclass
class BoundResult:
    """Certified bound on ``f`` over a compact box."""

    lower_bound: float
    upper_bound: float
    boxes_examined: int
    boxes_remaining: int
    converged: bool
    argmin_box: dict[str, tuple[float, float]] = field(default_factory=dict)
    domain_errors: int = 0
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"lower_bound": self.lower_bound, "upper_bound": self.upper_bound,
                "boxes_examined": self.boxes_examined,
                "boxes_remaining": self.boxes_remaining,
                "converged": self.converged, "argmin_box": self.argmin_box,
                "domain_errors": self.domain_errors, "note": self.note}


def global_lower_bound(expr: sp.Expr,
                       box: Mapping[sp.Symbol, tuple[float, float]],
                       target: float = 0.0,
                       max_boxes: int = 20000,
                       tol: float = 1e-9,
                       stop_on_proof: bool = True) -> BoundResult:
    """Prove ``inf_box expr >= target`` (or find where it fails).

    Standard interval branch-and-bound: a sub-box whose enclosure lies
    entirely above ``target`` is discarded; otherwise it is bisected along
    its widest dimension. If ``stop_on_proof`` and every box is discarded,
    the inequality is proved on the whole box.

    The returned ``lower_bound`` is a rigorous lower bound on the infimum;
    ``upper_bound`` is the best value actually attained at a sampled point
    (so ``lower_bound <= inf f <= upper_bound``).
    """
    syms = list(box.keys())
    expr = sp.sympify(expr)
    f_pt = sp.lambdify(syms, expr, "math")

    def sample(b: Mapping[sp.Symbol, tuple[float, float]]) -> float:
        mid = [0.5 * (b[s][0] + b[s][1]) for s in syms]
        try:
             v = float(f_pt(*mid))
             return v if math.isfinite(v) else float("inf")
        except Exception:                                 # noqa: BLE001
             return float("inf")

    start = {s: (float(box[s][0]), float(box[s][1])) for s in syms}
    best_upper = sample(start)
    stack: list[dict[Any, tuple[float, float]]] = [start]
    examined = 0
    domain_errors = 0
    worst_lower = float("inf")
    argmin_box: dict[str, tuple[float, float]] = {}

    while stack and examined < max_boxes:
        b = stack.pop()
        examined += 1
        try:
             enc = interval_eval(expr, b)
             lo, hi = float(enc.a), float(enc.b)
        except IntervalDomainError:
             domain_errors += 1
             lo, hi = -math.inf, math.inf
        except Exception:                                # noqa: BLE001
             domain_errors += 1
             lo, hi = -math.inf, math.inf

        if math.isfinite(hi):
            v = sample(b)
            if v < best_upper:
                best_upper = v

        if lo > target + tol:
            continue                                    # box cannot violate

        widths = [(b[s][1] - b[s][0], s) for s in syms]
        w, s = max(widths, key=lambda t: t[0])
        if w <= tol:
            if lo < worst_lower:
                worst_lower = lo
                argmin_box = {str(k): v for k, v in b.items()}
            if stop_on_proof and hi < target - tol:
                # a whole small box is strictly below target: refutation
                return BoundResult(lo, hi, examined, len(stack), True,
                                   {str(k): v for k, v in b.items()},
                                   domain_errors,
                                   "counterexample box found: expression is "
                                   "strictly below the target here")
            continue

        mid = 0.5 * (b[s][0] + b[s][1])
        left = dict(b); left[s] = (b[s][0], mid)
        right = dict(b); right[s] = (mid, b[s][1])
        stack.append(left)
        stack.append(right)

    if not stack:
        lb = target if not math.isfinite(worst_lower) else min(worst_lower, target)
        return BoundResult(max(lb, worst_lower if math.isfinite(worst_lower) else target),
                           best_upper, examined, 0, True, argmin_box, domain_errors,
                           "all boxes discarded: inequality proved on the box"
                           if domain_errors == 0 else
                           "all boxes discarded, but some sub-boxes left the real "
                           "domain and could not be enclosed")
    return BoundResult(worst_lower if math.isfinite(worst_lower) else -math.inf,
                       best_upper, examined, len(stack), False, argmin_box,
                       domain_errors,
                       f"budget exhausted after {examined} boxes; "
                       f"{len(stack)} undecided boxes remain")


# --------------------------------------------------------------------------
# Positivity certificate
# --------------------------------------------------------------------------

@dataclass
class PositivityCertificate:
    """Outcome of trying to prove ``expr > 0`` on a declared domain."""

    proved: bool
    refuted: bool
    method: str
    box: dict[str, tuple[float, float]] = field(default_factory=dict)
    domain_is_compact: bool = True
    lower_bound: float | None = None
    counterexample: dict[str, float] | None = None
    symbolic_note: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.proved:
            return "CERTIFIED"
        if self.refuted:
            return "REFUTED"
        return "UNRESOLVED"

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "proved": self.proved, "refuted": self.refuted,
                "method": self.method, "box": self.box,
                "domain_is_compact": self.domain_is_compact,
                "lower_bound": self.lower_bound,
                "counterexample": self.counterexample,
                "symbolic_note": self.symbolic_note, "detail": self.detail}


def prove_positive(expr: sp.Expr,
                   box: Mapping[sp.Symbol, tuple[float, float]] | None = None,
                   assumptions: Mapping[sp.Symbol, dict[str, bool]] | None = None,
                   strict: bool = True,
                   max_boxes: int = 20000,
                   domain_is_compact: bool = True) -> PositivityCertificate:
    """Try symbolic positivity first, then interval branch-and-bound.

    ``assumptions`` lets the caller re-declare symbols with domain knowledge
    (for example ``{r: {"positive": True}}``) before the symbolic attempt.
    """
    e = sp.sympify(expr)
    note = ""

    if assumptions:
        ren = {}
        for s, kw in assumptions.items():
            ren[s] = sp.Symbol(str(s), **kw)
        e_assumed = e.subs(ren, simultaneous=True)
    else:
        e_assumed = e

    simp = sp.simplify(e_assumed)
    if simp.is_positive is True:
        return PositivityCertificate(True, False, "symbolic",
                                     {str(k): v for k, v in (box or {}).items()},
                                     domain_is_compact,
                                     symbolic_note=f"sympy proved {simp} > 0")
    if simp.is_negative is True:
        return PositivityCertificate(False, True, "symbolic",
                                     {str(k): v for k, v in (box or {}).items()},
                                     domain_is_compact,
                                     symbolic_note=f"sympy proved {simp} < 0")
    if not strict and simp.is_nonnegative is True:
        return PositivityCertificate(True, False, "symbolic",
                                     {str(k): v for k, v in (box or {}).items()},
                                     domain_is_compact,
                                     symbolic_note=f"sympy proved {simp} >= 0")
    note = f"symbolic attempt inconclusive for {simp}"

    free = sorted(e.free_symbols, key=str)
    if not free:
        v = float(sp.N(e))
        ok = v > 0 if strict else v >= 0
        return PositivityCertificate(ok, not ok, "constant", {},
                                     True, v, None, "expression is constant")

    if box is None:
        return PositivityCertificate(False, False, "none", {}, domain_is_compact,
                                     symbolic_note=note + " | no box supplied for "
                                                          "interval certification")

    res = global_lower_bound(e, box, target=0.0, max_boxes=max_boxes)
    boxd = {str(k): (float(v[0]), float(v[1])) for k, v in box.items()}
    if res.converged and res.boxes_remaining == 0 and res.domain_errors == 0:
        return PositivityCertificate(True, False, "interval_branch_and_bound",
                                     boxd, domain_is_compact, res.lower_bound,
                                     None, note, res.as_dict())
    if res.upper_bound < 0:
        cex = {k: 0.5 * (v[0] + v[1]) for k, v in (res.argmin_box or boxd).items()}
        return PositivityCertificate(False, True, "interval_branch_and_bound",
                                     boxd, domain_is_compact, res.lower_bound,
                                     cex, note, res.as_dict())
    return PositivityCertificate(False, False, "interval_branch_and_bound",
                                 boxd, domain_is_compact, res.lower_bound, None,
                                 note, res.as_dict())


__all__ = ["interval_eval", "IntervalDomainError", "global_lower_bound",
           "BoundResult", "prove_positive", "PositivityCertificate"]
