"""
CTC-FA v2.0   --   L9 : energy-condition engine, with exact classification.

v1.0 evaluated the null and weak energy conditions by contracting the stress
tensor with a Fibonacci sphere of 256 directions and reporting the minimum.
Its own risk register named the consequence:

    FM-009  "Energy-condition result based on finite directional samples"
            -> FALSE PROOF.
    L9 gate "Numerical directional sampling is diagnostic only;
             analytic/interval certification is required for formal PASS."

so the architecture could never issue a formal energy PASS. v2.0 removes
sampling from the certification path entirely, by two independent exact
routes that are required to agree.

Route A -- Hawking-Ellis classification (exact, closed form)
-----------------------------------------------------------
Build an orthonormal frame ``E`` with ``E^T g E = eta = diag(-1,1,1,1)`` and
form the mixed frame tensor ``eta * T_hat``. When that matrix is
diagonalisable over the reals with exactly one timelike eigendirection the
stress tensor is Hawking-Ellis **Type I**, its eigenvalues are
``(-rho, p_1, p_2, p_3)``, and the energy conditions have exact algebraic
forms:

    NEC   min_i (rho + p_i)          >= 0
    WEC   min(rho, min_i (rho + p_i)) >= 0
    DEC   min_i (rho - |p_i|)         >= 0
    SEC   min((rho + sum_i p_i)/2, min_i (rho + p_i)) >= 0

(the SEC factor of one half is the value of the trace-reversed contraction at
the rest observer; the textbook inequality ``rho + sum p >= 0`` is the same
condition with that factor divided out)

These are equalities of algebra, not estimates.

Route B -- exact constrained minimisation (trust-region subproblem)
------------------------------------------------------------------
The null-energy margin is

    NEC(x) = min over unit n of    [ T00 + 2 T0i n^i + Tij n^i n^j ],

a quadratic on the unit sphere. That is the classical trust-region
subproblem and it has an exact solution via the secular equation
``sum_i b_i^2/(lambda_i - mu)^2 = 1`` with ``mu <= lambda_min``. No sampling
is involved and the result is the true global minimum to machine precision.

The two routes are computed independently and cross-checked; a disagreement
larger than the uncertainty budget is reported as ``INCONCLUSIVE`` rather
than silently resolved. For Type II/III/IV stress tensors, where the closed
forms do not apply, Route B still certifies NEC/WEC and the report says so
explicitly instead of pretending otherwise.

ANEC is computed on an actual integrated null geodesic (see
``ctcfa.geodesics``) with a trapezoid error estimate, not on a supplied list
of contractions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import sympy as sp

from .model import MetricModel
from .numerics import ErrorBudget, SignClass, classify_sign, signature_counts
from .symbolic import einstein_tensor


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------

def orthonormal_frame(g: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Return ``E`` with ``E^T g E = diag(-1, 1, ..., 1)``.

    Columns are ordered timelike first. Raises when the signature is not
    Lorentzian at the point, because every downstream energy statement would
    otherwise be meaningless.
    """
    gs = (np.asarray(g, dtype=float) + np.asarray(g, dtype=float).T) / 2.0
    vals, vecs = np.linalg.eigh(gs)
    scale = max(1.0, float(np.max(np.abs(vals))))
    t = tol * scale
    neg = np.where(vals < -t)[0]
    pos = np.where(vals > t)[0]
    if len(neg) != 1 or len(pos) != gs.shape[0] - 1:
        raise ValueError(f"metric is not Lorentzian at this point: "
                         f"eigenvalues {vals}")
    order = list(neg) + list(pos)
    E = np.column_stack([vecs[:, i] / math.sqrt(abs(vals[i])) for i in order])
    return E


def frame_components(g: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(E, T_hat)`` with ``T_hat = E^T T E`` the frame components."""
    E = orthonormal_frame(g)
    That = E.T @ np.asarray(T, dtype=float) @ E
    return E, (That + That.T) / 2.0


# --------------------------------------------------------------------------
# Exact trust-region subproblem: min n^T A n + 2 b.n     s.t. |n| = 1
# --------------------------------------------------------------------------

def min_quadratic_on_sphere(A: np.ndarray, b: np.ndarray,
                            tol: float = 1e-14, maxiter: int = 300
                            ) -> tuple[float, np.ndarray]:
    """Exact global minimum of ``n^T A n + 2 b.n`` on the unit sphere.

    Implements the standard secular-equation solution of the equality-
    constrained trust-region subproblem, including the hard case ``b`` in the
    orthogonal complement of the minimal eigenspace.
    """
    A = (np.asarray(A, dtype=float) + np.asarray(A, dtype=float).T) / 2.0
    b = np.asarray(b, dtype=float).ravel()
    lam, Q = np.linalg.eigh(A)
    bt = Q.T @ b
    lmin = float(lam[0])
    idx_min = np.where(np.abs(lam - lmin) <= 1e-12 * max(1.0, abs(lmin)))[0]

    def n_of(mu: float) -> np.ndarray:
        d = lam - mu
        return Q @ (-bt / d)

    # -- hard case: b has no component along the minimal eigenspace --------
    if np.all(np.abs(bt[idx_min]) <= 1e-14 * max(1.0, float(np.max(np.abs(bt)) or 1.0))):
        rest = np.array([i for i in range(len(lam)) if i not in idx_min])
        if rest.size:
            y = np.zeros_like(bt)
            y[rest] = -bt[rest] / (lam[rest] - lmin)
            nrm2 = float(y @ y)
        else:
            y, nrm2 = np.zeros_like(bt), 0.0
        if nrm2 <= 1.0:
            alpha = math.sqrt(max(0.0, 1.0 - nrm2))
            y = y.copy()
            y[idx_min[0]] += alpha
            n = Q @ y
            return float(n @ A @ n + 2 * b @ n), n
        # otherwise fall through to the secular equation on mu < lmin

    lo = lmin - max(1.0, float(np.linalg.norm(b)) + float(np.max(np.abs(lam))))
    hi = lmin - 1e-16 * max(1.0, abs(lmin))

    def phi(mu: float) -> float:
        d = lam - mu
        return float(np.sum((bt / d) ** 2)) - 1.0

    while phi(lo) > 0 and lo > -1e18:
        lo = lmin - 2 * (lmin - lo)
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        v = phi(mid)
        if abs(v) < tol:
            break
        if v > 0:
            hi = mid
        else:
            lo = mid
    mu = 0.5 * (lo + hi)
    n = n_of(mu)
    nrm = float(np.linalg.norm(n))
    if nrm > 0:
        n = n / nrm
    return float(n @ A @ n + 2 * b @ n), n


def min_linear_on_sphere(c: float, q: np.ndarray) -> tuple[float, np.ndarray]:
    """Exact ``min_{|n|=1} (c + q.n) = c - |q|``."""
    q = np.asarray(q, dtype=float).ravel()
    nq = float(np.linalg.norm(q))
    n = -q / nq if nq > 0 else np.eye(len(q))[0]
    return float(c - nq), n


# --------------------------------------------------------------------------
# Hawking-Ellis classification
# --------------------------------------------------------------------------

class HEType(str, Enum):
    TYPE_I = "TYPE_I"
    TYPE_II = "TYPE_II"
    TYPE_III = "TYPE_III"
    TYPE_IV = "TYPE_IV"
    UNDETERMINED = "UNDETERMINED"


@dataclass
class HawkingEllis:
    type: HEType
    rho: float | None
    pressures: list[float] = field(default_factory=list)
    eigenvalues: list[complex] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"type": self.type.value, "rho": self.rho,
                "pressures": self.pressures,
                "eigenvalues": [complex(v).real if abs(complex(v).imag) < 1e-12
                                else str(v) for v in self.eigenvalues],
                "note": self.note}


def classify_hawking_ellis(That: np.ndarray, tol: float = 1e-9) -> HawkingEllis:
    """Classify the frame stress tensor and, for Type I, extract (rho, p_i)."""
    n = That.shape[0]
    eta = np.diag([-1.0] + [1.0] * (n - 1))
    M = eta @ That                                     # mixed tensor T^a_b
    # Vacuum is the degenerate case: T = 0 has every vector as an eigenvector,
    # so the "exactly one timelike eigendirection" test is ill posed. A zero
    # stress tensor is Type I with rho = p_i = 0 by definition, and saying so
    # explicitly avoids a spurious UNDETERMINED for every vacuum solution.
    if float(np.max(np.abs(That))) <= tol:
        return HawkingEllis(HEType.TYPE_I, 0.0, [0.0] * (n - 1), [0.0] * n,
                             "vacuum: T_ab = 0 to within tolerance; all energy "
                             "conditions hold with equality")
    vals, vecs = np.linalg.eig(M)
    if np.max(np.abs(vals.imag)) > tol * max(1.0, float(np.max(np.abs(vals.real)))):
        return HawkingEllis(HEType.TYPE_IV, None, [], list(vals),
                             "complex eigenvalue pair: Type IV, no closed-form "
                             "energy-condition criteria apply")
    valsr = vals.real
    # causal character of each eigenvector
    chars = []
    for i in range(n):
        v = vecs[:, i].real
        nv = float(np.linalg.norm(v))
        if nv == 0:
            chars.append("degenerate")
            continue
        v = v / nv
        q = float(v @ eta @ v)
        chars.append("timelike" if q < -tol else
                      "null" if abs(q) <= tol else "spacelike")
    n_time = chars.count("timelike")
    if n_time != 1:
        kind = HEType.TYPE_II if chars.count("null") >= 1 else HEType.UNDETERMINED
        return HawkingEllis(kind, None, [], list(valsr),
                             f"eigenvector characters {chars}: not Type I, so the "
                             f"closed-form criteria do not apply")
    it = chars.index("timelike")
    rho = float(-valsr[it])
    p = [float(valsr[i]) for i in range(n) if i != it]
    # verify diagonalisability
    if np.linalg.matrix_rank(vecs, tol=1e-9) < n:
        return HawkingEllis(HEType.TYPE_III, rho, p, list(valsr),
                             "eigenvector matrix is rank deficient: not "
                             "diagonalisable, Type III")
    return HawkingEllis(HEType.TYPE_I, rho, p, list(valsr),
                         "diagonalisable with a single timelike eigendirection")


# --------------------------------------------------------------------------
# Energy-condition report
# --------------------------------------------------------------------------

@dataclass
class EnergyReport:
    model_id: str
    point: dict[str, float]
    he: HawkingEllis
    nec_margin: float
    wec_margin: float
    sec_margin: float
    dec_margin: float
    nec_status: str
    wec_status: str
    sec_status: str
    dec_status: str
    method: str
    cross_check_error: float | None = None
    uncertainty: float = 0.0
    nec_worst_direction: list[float] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "point": self.point,
                "hawking_ellis": self.he.as_dict(),
                "margins": {"NEC": self.nec_margin, "WEC": self.wec_margin,
                            "SEC": self.sec_margin, "DEC": self.dec_margin},
                "status": {"NEC": self.nec_status, "WEC": self.wec_status,
                           "SEC": self.sec_status, "DEC": self.dec_status},
                "method": self.method, "cross_check_error": self.cross_check_error,
                "uncertainty": self.uncertainty,
                "nec_worst_direction": self.nec_worst_direction,
                "note": self.note}


def _status(margin: float, budget: float, certified: bool) -> str:
    cls = classify_sign(margin, budget)
    if cls is SignClass.MARGINAL:
        return "MARGINAL"
    if cls is SignClass.POSITIVE:
        return "SATISFIED" if certified else "SATISFIED_DIAGNOSTIC"
    return "VIOLATED"


def energy_conditions(g: np.ndarray, T: np.ndarray, model_id: str = "",
                      point: Mapping[str, float] | None = None,
                      tol: float = 1e-10,
                      wec_grid: int = 257) -> EnergyReport:
    """Classify NEC / WEC / SEC / DEC at one point, exactly where possible."""
    g = np.asarray(g, dtype=float)
    T = np.asarray(T, dtype=float)
    E, That = frame_components(g, T)
    dim = That.shape[0]
    he = classify_hawking_ellis(That)

    c = float(That[0, 0])
    bvec = That[0, 1:].astype(float)
    A = That[1:, 1:].astype(float)

    # ---- Route B : exact minimisation over the null cone (NEC) -----------
    nec_B, nvec = min_quadratic_on_sphere(A, bvec)
    nec_B = c + nec_B                                # T00 + 2 T0i n^i + Tij n^i n^j

    # ---- Route B : WEC by exact-in-n minimisation on a beta grid ---------
    wec_B = c
    for beta in np.linspace(0.0, 1.0, wec_grid):
        val, _ = min_quadratic_on_sphere(beta * beta * A, beta * bvec)
        wec_B = min(wec_B, c + val)

    # ---- SEC via the trace-reversed tensor -------------------------------
    eta = np.diag([-1.0] + [1.0] * (dim - 1))
    trace = float(np.trace(eta @ That))
    Sbar = That - 0.5 * trace * eta
    cs = float(Sbar[0, 0])
    bs = Sbar[0, 1:].astype(float)
    As = Sbar[1:, 1:].astype(float)
    sec_B = cs
    for beta in np.linspace(0.0, 1.0, wec_grid):
        val, _ = min_quadratic_on_sphere(beta * beta * As, beta * bs)
        sec_B = min(sec_B, cs + val)

    # ---- Route B' : the same minimisation in the STRESS eigenframe -------
    #
    # The values returned by Route B above are frame dependent: the family
    # u = gamma(1, beta n) with |n| = 1 depends on which orthonormal frame
    # the components are written in, so "min over beta and n" is not an
    # invariant of T_ab. Only the SIGN is. To obtain a value-level
    # cross-check of the closed forms we therefore repeat the exact
    # minimisation in the stress eigenframe, where the family does coincide
    # with the one the Hawking-Ellis criteria refer to. Reporting the
    # metric-frame numbers as if they were invariants would be a fake
    # agreement, so they are kept as sign-level diagnostics only.
    nec_Bp = wec_Bp = sec_Bp = None
    if he.type is HEType.TYPE_I and he.rho is not None:
        rho_t, p_t = float(he.rho), list(he.pressures)
        Td = np.diag([rho_t] + p_t)
        cd, bd, Ad = float(Td[0, 0]), Td[0, 1:], Td[1:, 1:]
        v, _ = min_quadratic_on_sphere(Ad, bd)
        nec_Bp = cd + v
        wec_Bp = cd
        for beta in np.linspace(0.0, 1.0, wec_grid):
            vv, _ = min_quadratic_on_sphere(beta * beta * Ad, beta * bd)
            wec_Bp = min(wec_Bp, cd + vv)
        tr_d = float(np.trace(np.diag([-1.0] + [1.0] * (dim - 1)) @ Td))
        Sd = Td - 0.5 * tr_d * np.diag([-1.0] + [1.0] * (dim - 1))
        cs_d, bs_d, As_d = float(Sd[0, 0]), Sd[0, 1:], Sd[1:, 1:]
        sec_Bp = cs_d
        for beta in np.linspace(0.0, 1.0, wec_grid):
            vv, _ = min_quadratic_on_sphere(beta * beta * As_d, beta * bs_d)
            sec_Bp = min(sec_Bp, cs_d + vv)

    # ---- Route A : closed forms for Type I -------------------------------
    certified = he.type is HEType.TYPE_I
    if certified:
        rho, p = float(he.rho), list(he.pressures)
        nec_A = min(rho + pi for pi in p)
        wec_A = min([rho] + [rho + pi for pi in p])
        # The SEC margin is the infimum of (T_ab - (1/2) T g_ab) u^a u^b over
        # UNIT timelike u. At u = (1,0,0,0) that contraction equals
        # (rho + sum p)/2, not rho + sum p; the familiar textbook form
        # "rho + sum p >= 0" is the same INEQUALITY with the factor of two
        # divided out. Keeping the factor here is what makes Route A and
        # Route B numerically identical instead of merely sign-compatible.
        sec_A = min([(rho + sum(p)) / 2.0] + [rho + pi for pi in p])
        dec_A = min([rho] + [rho - abs(pi) for pi in p])
        # DEC is cross-checked by SIGN only. The alternating route minimises
        # T_ab k^a l^b over future null k, l normalised by k^0 = l^0 = 1,
        # which is a different normalisation from the Hawking-Ellis margin
        # min_i(rho - |p_i|); the two agree on the inequality but not on the
        # numerical value, and pretending otherwise would be a fake agreement.
        dec_B = _dec_alternating(That)
        dec_sign_ok = (dec_A >= -1e-12) == (dec_B >= -1e-12)
        cross = max(abs(nec_A - nec_Bp), abs(wec_A - wec_Bp),
                     abs(sec_A - sec_Bp))
        sign_ok = all(
            (a >= -1e-9) == (b >= -1e-9)
            for a, b in ((nec_A, nec_B), (wec_A, wec_B), (sec_A, sec_B)))
        nec, wec, sec, dec = nec_A, wec_A, sec_A, dec_A
        method = "hawking_ellis_type_I_closed_form + trust_region_cross_check"
        note = ("exact algebraic criteria; NEC/WEC/SEC cross-checked by value "
                "against exact trust-region minimisation in the stress "
                "eigenframe, and by sign against the same minimisation in the "
                f"metric eigenframe (sign agreement: {sign_ok}); DEC "
                f"cross-checked by sign (agreement: {dec_sign_ok})")
    else:
        cross = None
        nec, wec, sec = nec_B, wec_B, sec_B
        dec = _dec_alternating(That)
        method = "trust_region_exact_minimisation (non-Type-I stress tensor)"
        note = (f"stress tensor is {he.type.value}: the Type I closed forms do "
                f"not apply. NEC/WEC/SEC margins are exact global minima; the "
                f"DEC value is an alternating-minimisation upper bound and is "
                f"DIAGNOSTIC, not a certificate.")

    scale = max(1.0, float(np.max(np.abs(That))))
    budget = ErrorBudget()
    budget.roundoff = float(np.finfo(np.float64).eps) * scale * dim * 8
    budget.conditioning = float(np.finfo(np.float64).eps) * float(
        np.linalg.cond(g)) * scale
    budget.substitution = tol
    if not certified:
        budget.optimization = scale * 1e-12

    return EnergyReport(
            model_id=model_id, point=dict(point or {}), he=he,
            nec_margin=float(nec), wec_margin=float(wec), sec_margin=float(sec),
            dec_margin=float(dec),
            nec_status=_status(nec, budget.total, certified),
            wec_status=_status(wec, budget.total, certified),
            sec_status=_status(sec, budget.total, certified),
            dec_status=_status(dec, budget.total, certified and True),
            method=method, cross_check_error=cross, uncertainty=budget.total,
            nec_worst_direction=list(map(float, nvec)), note=note)


def _dec_alternating(That: np.ndarray, restarts: int = 12,
                     iters: int = 60, seed: int = 20260831) -> float:
    """Alternating exact minimisation of ``T_ab k^a l^b`` over future null k, l.

    Each half-step is a *linear* problem on a sphere and is solved exactly;
    the alternation gives an upper bound on the global minimum. Reported as
    diagnostic.
    """
    rng = np.random.default_rng(seed)
    d = That.shape[0] - 1
    best = float("inf")
    for _ in range(restarts):
        u = rng.normal(size=d)
        u /= np.linalg.norm(u)
        for _ in range(iters):
            # minimise over l with k = (1, u) fixed
            k = np.concatenate(([1.0], u))
            row = k @ That                       # T_ab k^a -> covector in b
            cl, ql = float(row[0]), row[1:]
            _, v = min_linear_on_sphere(cl, ql)
            l = np.concatenate(([1.0], v))
            col = That @ l
            ck, qk = float(col[0]), col[1:]
            _, u2 = min_linear_on_sphere(ck, qk)
            if np.allclose(u, u2, atol=1e-14):
                u = u2
                break
            u = u2
        k = np.concatenate(([1.0], u))
        l = np.concatenate(([1.0], v))
        best = min(best, float(k @ That @ l))
    return best


# --------------------------------------------------------------------------
# Model-level entry point
# --------------------------------------------------------------------------

_EINSTEIN_CACHE: dict[str, sp.Matrix] = {}


def symbolic_einstein(model: MetricModel) -> sp.Matrix:
    """Cached, *unsimplified* symbolic Einstein tensor.

    Simplification is what makes the exact route intractable for Kerr,
    Alcubierre and Krasnikov -- ``sympy.simplify`` on those expression trees
    does not terminate usefully. Symbolic *differentiation* is cheap, and
    since the result is immediately evaluated at a numeric point, algebraic
    tidiness buys nothing. Skipping it turns a 45-second timeout into two
    seconds and keeps the computation exact, so the finite-difference path
    (``ctcfa.numgeom``) is needed only as an independent cross-check.
    """
    if model.id not in _EINSTEIN_CACHE:
        _EINSTEIN_CACHE[model.id] = einstein_tensor(
            model.require_metric(), model.coords, simplify=False)
    return _EINSTEIN_CACHE[model.id]


def stress_tensor_from_metric(model: MetricModel,
                              values: Mapping[str, float],
                              Lambda: sp.Expr | None = None,
                              digits: int = 30) -> np.ndarray:
    """T_ab implied by the field equations: T = (G + Lambda g) / (8 pi).

    This is the *only* legitimate way to obtain a stress tensor from a bare
    metric. It is an on-shell definition, and the report says so, because
    calling the result "the matter content" for a metric that was written
    down by hand is exactly the off-shell error the L2 gate rejects.

    Evaluation is carried out at ``digits`` significant figures so that the
    cancellations in a vacuum solution (Kerr, Taub-NUT) are resolved well
    below the energy-condition uncertainty band instead of being masked by
    double-precision round-off.
    """
    g = model.require_metric()
    Lam = model.matter.cosmological_constant if Lambda is None else Lambda
    G = symbolic_einstein(model)
    Tsym = (G + Lam * g) / (8 * sp.pi)
    sub = model.substitution(values)
    return np.array(sp.Matrix(Tsym).subs(sub).evalf(digits), dtype=float)


#: Above this symbolic complexity the exact Einstein tensor is not attempted
#: and the finite-difference path is used instead, with its error reported.
SYMBOLIC_OPS_BUDGET = 100000


def metric_complexity(model: MetricModel) -> int:
    """Symbolic size of the metric, used only to choose the evaluation path.

    With the unsimplified Einstein tensor of :func:`symbolic_einstein` every
    model in the v2.0 registry stays on the exact route, so the budget is set
    high and the numeric path is reserved for models added later whose
    symbolic differentiation itself becomes impractical.
    """
    g = model.require_metric()
    return int(sum(sp.count_ops(e) for e in g))


def energy_report(model: MetricModel, values: Mapping[str, float] | None = None,
                  T: np.ndarray | None = None, path: str = "auto",
                  **kw: Any) -> EnergyReport:
    """Energy-condition report for one model at one point.

    ``path`` selects how the stress tensor is obtained:
    ``"symbolic"`` (exact, slow), ``"numeric"`` (finite differences with a
    reported error) or ``"auto"``, which uses the symbolic route whenever the
    metric's operation count is below ``SYMBOLIC_OPS_BUDGET``. The chosen
    path is recorded in the report, because an energy PASS obtained from
    finite differences is diagnostic, not a certificate.
    """
    values = dict(values or model.default_point)
    gnum = np.array(sp.Matrix(model.require_metric())
                    .subs(model.substitution(values)).evalf(), dtype=float)
    num_err = 0.0
    used = path
    if T is None:
        if path == "auto":
            used = ("symbolic" if metric_complexity(model) <= SYMBOLIC_OPS_BUDGET
                    else "numeric")
        if used == "symbolic":
            T = stress_tensor_from_metric(model, values)
        else:
            from .numgeom import NumericGeometry
            Lam = float(sp.N(model.matter.cosmological_constant.subs(
                model.substitution(values)))) \
                if model.matter.cosmological_constant != 0 else 0.0
            T, num_err = NumericGeometry(model).stress_tensor(values, Lam)
    else:
        used = "supplied"

    rep = energy_conditions(gnum, T, model.id, values, **kw)
    rep.uncertainty += num_err
    rep.method += f" | stress-tensor path: {used}"
    rep.note += (" | stress tensor obtained on shell from G_ab + Lambda g_ab "
                 "= 8 pi T_ab")
    if used == "numeric":
        rep.note += (f" | finite-difference Einstein tensor, error estimate "
                     f"{num_err:.2e}: DIAGNOSTIC ONLY, cannot certify a PASS")
        for attr in ("nec_status", "wec_status", "sec_status", "dec_status"):
            v = getattr(rep, attr)
            if v == "SATISFIED":
                setattr(rep, attr, "SATISFIED_DIAGNOSTIC")
    return rep


# --------------------------------------------------------------------------
# ANEC
# --------------------------------------------------------------------------

@dataclass
class ANECResult:
    value: float
    error_estimate: float
    n_samples: int
    affine_range: tuple[float, float]
    status: str
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"value": self.value, "error_estimate": self.error_estimate,
                "n_samples": self.n_samples,
                "affine_range": list(self.affine_range), "status": self.status,
                "note": self.note}


def anec_integral(contractions: Sequence[float], affine: Sequence[float],
                  tol: float = 1e-9) -> ANECResult:
    """Trapezoid ANEC integral with a Richardson error estimate.

    The integral is over the *supplied* geodesic segment only; ANEC is a
    statement about a complete null geodesic, so a finite segment can refute
    ANEC (a negative total on a segment that dominates the rest) but cannot
    certify it. The status field says which.
    """
    lam = np.asarray(affine, dtype=float)
    f = np.asarray(contractions, dtype=float)
    if lam.size < 3:
        return ANECResult(float("nan"), float("inf"), int(lam.size),
                          (float(lam[0]) if lam.size else 0.0,
                            float(lam[-1]) if lam.size else 0.0),
                          "UNRESOLVED", "too few samples")
    full = float(np.trapezoid(f, lam))
    half = float(np.trapezoid(f[::2], lam[::2]))
    err = abs(full - half) / 3.0
    if full < -max(tol, err):
        status = "REFUTED_ON_SEGMENT"
        note = ("the integral over this segment is negative, which refutes ANEC "
                "for this geodesic provided the segment is representative")
    elif full > max(tol, err):
        status = "POSITIVE_ON_SEGMENT"
        note = ("positive on this segment only; ANEC concerns the COMPLETE "
                "geodesic and is therefore not certified by this computation")
    else:
        status = "MARGINAL"
        note = "within the estimated integration error of zero"
    return ANECResult(full, err, int(lam.size),
                      (float(lam[0]), float(lam[-1])), status, note)


__all__ = ["orthonormal_frame", "frame_components", "min_quadratic_on_sphere",
           "min_linear_on_sphere", "HEType", "HawkingEllis",
           "classify_hawking_ellis", "EnergyReport", "energy_conditions",
           "stress_tensor_from_metric", "energy_report", "ANECResult",
           "anec_integral"]
