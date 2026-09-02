"""Unit tests for the individual engines."""

import math

import numpy as np
import pytest
import sympy as sp

from ctcfa import registry as R
from ctcfa.energy import (classify_hawking_ellis, min_quadratic_on_sphere,
                           orthonormal_frame)
from ctcfa.interval import global_lower_bound, interval_eval, prove_positive
from ctcfa.numerics import (ErrorBudget, SignClass, classify_sign,
                             signature_counts)
from ctcfa.numgeom import NumericGeometry
from ctcfa.orientation import check_curve_orientation, timelike_eigenvector
from ctcfa.symmetry import solve_killing, verify_compact_generators


# -- numerics --------------------------------------------------------------

def test_sign_classification_uses_the_uncertainty_band():
    b = ErrorBudget(floor=1e-12)
    assert classify_sign(-1e-15, b) is SignClass.MARGINAL
    assert classify_sign(-1e-6, b) is SignClass.NEGATIVE
    assert classify_sign(+1e-6, b) is SignClass.POSITIVE


def test_signature_counts_are_congruence_invariant():
    g = np.diag([-1.0, 1.0, 1.0, 1.0])
    rng = np.random.default_rng(3)
    for _ in range(20):
        A = rng.normal(size=(4, 4))
        if abs(np.linalg.det(A)) < 1e-6:
            continue
        assert signature_counts(A.T @ g @ A) == (1, 0, 3)


# -- interval arithmetic ---------------------------------------------------

def test_interval_enclosure_contains_the_true_range():
    x = sp.Symbol("x")
    e = sp.sin(x) * x ** 2
    enc = interval_eval(e, {x: (0.0, 2.0)})
    xs = np.linspace(0, 2, 5000)
    vals = np.sin(xs) * xs ** 2
    assert float(enc.a) <= vals.min() + 1e-12
    assert float(enc.b) >= vals.max() - 1e-12


def test_branch_and_bound_proves_and_refutes():
    x = sp.Symbol("x")
    assert prove_positive(1 + x ** 2, {x: (-3.0, 3.0)}).proved
    c = prove_positive(1 - x ** 2, {x: (-3.0, 3.0)})
    assert c.refuted and not c.proved


# -- exact trust-region solver ---------------------------------------------

def test_trust_region_beats_dense_random_search():
    rng = np.random.default_rng(11)
    for _ in range(40):
        d = int(rng.integers(2, 5))
        A = rng.normal(size=(d, d))
        A = (A + A.T) / 2
        b = rng.normal(size=d)
        val, n = min_quadratic_on_sphere(A, b)
        assert abs(np.linalg.norm(n) - 1) < 1e-8
        V = rng.normal(size=(4000, d))
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        best = float(np.min(np.einsum("ij,jk,ik->i", V, A, V) + 2 * V @ b))
        assert val <= best + 1e-9


def test_hawking_ellis_type_I_for_a_perfect_fluid():
    That = np.diag([2.0, 0.5, 0.5, 0.5])
    he = classify_hawking_ellis(That)
    assert he.type.value == "TYPE_I"
    assert abs(he.rho - 2.0) < 1e-12
    assert all(abs(p - 0.5) < 1e-12 for p in he.pressures)


def test_orthonormal_frame_diagonalises_the_metric():
    for mid in ("kerr", "godel", "misner", "ori_2005"):
        m = R.get(mid)
        g = np.array(sp.Matrix(m.metric)
                      .subs(m.substitution(m.default_point)).evalf(), dtype=float)
        E = orthonormal_frame(g)
        eta = E.T @ g @ E
        assert np.allclose(eta, np.diag([-1, 1, 1, 1]), atol=1e-9), mid


# -- symmetry --------------------------------------------------------------

def test_minkowski_has_the_full_poincare_algebra():
    sol = solve_killing(R.get("minkowski"), "linear")
    assert sol.dimension == 10


def test_every_declared_compact_generator_is_killing():
    for m in R.models():
        if not m.has_metric:
            continue
        for cd, c in zip(m.compact, verify_compact_generators(m)):
            if cd.requires_killing:
                assert c.is_killing, f"{m.id}: {c.name} -> {c.killing_residual[:120]}"


# -- orientation -----------------------------------------------------------

def test_orientation_detects_a_reversing_curve():
    g = np.diag([-1.0, 1.0, 1.0, 1.0])
    metrics = [g] * 8
    tangents = [np.array([1.0, 0, 0, 0])] * 4 + [np.array([-1.0, 0, 0, 0])] * 4
    res = check_curve_orientation(metrics, tangents, closed=False)
    assert not res.consistent
    assert res.reversals >= 1


def test_orientation_accepts_a_consistent_curve():
    g = np.diag([-1.0, 1.0, 1.0, 1.0])
    metrics = [g] * 8
    tangents = [np.array([1.0, 0.1, 0, 0])] * 8
    assert check_curve_orientation(metrics, tangents, closed=False).consistent


# -- numeric geometry ------------------------------------------------------

def test_numeric_curvature_matches_the_exact_kretschmann():
    m = R.get("schwarzschild")
    c = NumericGeometry(m).curvature(m.default_point)
    exact = 48 * 1.0 ** 2 / 5.0 ** 6
    assert abs(c.kretschmann - exact) / exact < 1e-7


def test_numeric_and_symbolic_einstein_agree_on_a_hard_metric():
    """Alcubierre is the model the exact route could not simplify in v1.0."""
    from ctcfa.energy import symbolic_einstein
    m = R.get("alcubierre")
    Ga = np.array(sp.Matrix(symbolic_einstein(m))
                  .subs(m.substitution(m.default_point)).evalf(30), dtype=float)
    Gb, err = NumericGeometry(m).einstein(m.default_point)
    assert np.max(np.abs(Ga - Gb)) < max(1e-3, 200 * err)
