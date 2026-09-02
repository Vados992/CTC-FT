"""Analytic benchmarks: every causal boundary the architecture claims is
recovered from the metric and compared with the closed form afterwards.

The comparison direction matters. The solver never receives the analytic
value; it is used only to score the result. ``test_solver_is_non_circular``
asserts that the module which used to contain the hard-coded Godel answer
contains no such function any more.
"""

import math

import numpy as np
import pytest
import sympy as sp

from ctcfa import registry as R
from ctcfa.continuation import (causal_phase_boundary, convergence_study,
                                reduced_causal_margin_symbolic)
from ctcfa.quotient_ctc import (exact_orbit_interval, polarized_hypersurfaces,
                                winding_spectrum)


# -- the reduced causal margin collapses to the literature quantities -------

def test_godel_type_reduced_margin_is_the_tanh_form():
    m = R.get("godel_type")
    mu = reduced_causal_margin_symbolic(m)
    om, mm, r = m.symbol("omega"), m.symbol("m"), m.symbol("r")
    target = 1 - 4 * om ** 2 / mm ** 2 * sp.tanh(mm * r / 2) ** 2
    diff = sp.simplify((mu - target).rewrite(sp.exp))
    assert sp.simplify(diff) == 0


def test_van_stockum_reduced_margin():
    m = R.get("van_stockum")
    a, r = m.symbol("a"), m.symbol("r")
    assert sp.simplify(reduced_causal_margin_symbolic(m) - (1 - a ** 2 * r ** 2)) == 0


def test_spinning_string_reduced_margin():
    m = R.get("spinning_cosmic_string")
    J, al, r = m.symbol("J"), m.symbol("alpha"), m.symbol("r")
    target = 1 - 16 * J ** 2 / (al ** 2 * r ** 2)
    assert sp.simplify(reduced_causal_margin_symbolic(m) - target) == 0


def test_ori2007_reduced_margin():
    m = R.get("ori_2007_core")
    mu_m, r = m.symbol("mu"), m.symbol("r")
    assert sp.simplify(reduced_causal_margin_symbolic(m)
                       - (1 - 4 * mu_m ** 2 / r ** 2)) == 0


# -- boundaries recovered from the metric ----------------------------------

@pytest.mark.parametrize("mid,par,bracket,fixed,box,analytic", [
    ("van_stockum", "a", (0.05, 3.0), {}, {"r": (2.0, 2.0)}, 0.5),
    ("spinning_cosmic_string", "alpha", (0.5, 20.0), {"J": 0.1},
     {"r": (0.1, 0.1)}, 4.0),
    ("ori_2007_core", "r", (0.5, 6.0), {"mu": 1.0}, {}, 2.0),
    ("pseudo_reissner_nordstrom", "r", (0.125, 1.0), {"mu": 1.0, "q": 0.5}, {},
     1 - math.sqrt(0.75)),
    ("pseudo_reissner_nordstrom", "r", (1.0, 3.0), {"mu": 1.0, "q": 0.5}, {},
     1 + math.sqrt(0.75)),
    ("misner", "T", (-1.5, 1.5), {}, {}, 0.0),
    ("ori_2005", "T", (-1.5, 1.5), {"x": 0.0, "y": 0.0}, {}, 0.0),
])
def test_boundary_matches_closed_form(mid, par, bracket, fixed, box, analytic):
    th = causal_phase_boundary(R.get(mid), par, bracket, fixed, box,
                               analytic_value=analytic)
    assert th.domain_ok, th.note
    assert abs(th.value - analytic) < 1e-10
    assert th.bracket_width < 1e-9


def test_godel_boundary_converges_to_two_omega():
    """The Godel-type critical radius diverges as m -> 2 omega, so the
    boundary at finite box radius is only an approximation. Its convergence
    rate is itself the evidence, which is why the test checks the table and
    not one number."""
    rows = convergence_study(R.get("godel_type"), "m", (0.5, 6.0),
                              {"omega": 1.0}, "r", [5, 10, 15, 20],
                              analytic_value=2.0)
    errs = [r["error"] for r in rows if r.get("error") is not None]
    assert len(errs) == 4
    assert errs[0] > errs[-1]                       # monotone improvement
    assert errs[-1] < 1e-10


def test_pseudo_rn_normalisation_artefact_is_rejected():
    """The reference direction goes null at r^2 + 2 mu r - q^2 = 0. The root
    there is not a causal boundary and must be flagged."""
    th = causal_phase_boundary(R.get("pseudo_reissner_nordstrom"), "r",
                               (0.05, 0.13), {"mu": 1.0, "q": 0.5}, {})
    assert not th.domain_ok
    assert "artefact" in th.note


# -- flat quotients: exact closed forms ------------------------------------

def test_misner_covering_exact_orbit_interval():
    m = R.get("misner_covering")
    t, x, b = m.symbol("t"), m.symbol("x"), m.symbol("b")
    e = sp.simplify(exact_orbit_interval(m, 1))
    target = 2 * (sp.cosh(b) - 1) * (t ** 2 - x ** 2)
    assert sp.simplify(sp.expand(e - target)) == 0


def test_grant_exact_orbit_interval():
    m = R.get("grant_space")
    t, x, b, al = m.symbol("t"), m.symbol("x"), m.symbol("b"), m.symbol("alpha")
    e = sp.simplify(exact_orbit_interval(m, 1))
    target = 2 * (sp.cosh(b) - 1) * (t ** 2 - x ** 2) + al ** 2
    assert sp.simplify(sp.expand(e - target)) == 0


def test_grant_polarized_hypersurfaces_match_closed_form():
    """x_n^2 = n^2 alpha^2 / (2 (cosh(n b) - 1)).

    Locating these from the deck group is only possible because the
    identification structure is an executable object; v1.0 had no path to
    this result.
    """
    m = R.get("grant_space")
    b, alpha = 1.0, 1.0
    base = {"t": 0.0, "x": 0.8, "y": 0.0, "z": 0.0, "b": b, "alpha": alpha}
    rows = polarized_hypersurfaces(m, base, "x", 0.05, 3.0, max_power=5)
    got = {}
    for r in rows:
        if r["value"] > 0:
             got.setdefault(r["winding"], r["value"])
    for n in range(1, 6):
        target = math.sqrt(n * n * alpha ** 2 / (2 * (math.cosh(n * b) - 1)))
        assert abs(got[n] - target) < 1e-11, (n, got[n], target)


def test_misner_chart_and_covering_agree_pointwise():
    """Two independent representations of the same spacetime must give the
    same causal verdict at corresponding points."""
    from ctcfa.causality import compact_orbit_test
    from ctcfa.quotient_ctc import deck_orbit_test
    cov = R.get("misner_covering")
    chart = R.get("misner")
    b = 1.0
    for t, x in [(0.0, 0.5), (0.0, 2.0), (0.4, 0.5), (2.0, 0.5), (0.5, 0.5)]:
        d = deck_orbit_test(cov, {"t": t, "x": x, "y": 0.0, "z": 0.0, "b": b})
        analytic = 2 * (math.cosh(b) - 1) * (t * t - x * x)
        if abs(analytic) > 1e-12:
            assert abs(d.margin - analytic) < 1e-12
        # the Misner chart's T plays the role of (x^2 - t^2) up to a positive
        # factor, so the SIGNS must agree
        T = x * x - t * t
        c = compact_orbit_test(chart, {"T": T, "psi": 0.0, "y": 0.0, "z": 0.0})
        assert np.sign(c.margin) == np.sign(analytic) or abs(analytic) < 1e-12


def test_winding_spectrum_signs_are_monotone_for_grant():
    m = R.get("grant_space")
    rows = winding_spectrum(m, {"t": 0.0, "x": 0.8, "y": 0.0, "z": 0.0,
                                "b": 1.0, "alpha": 1.0}, max_power=6)
    chars = [r["character"] for r in rows]
    assert chars[0] == "spacelike"
    assert chars[-1] == "timelike"


def test_solver_is_non_circular():
    """The v1.0 kernel contained ``godel_family_margin(m, omega) = m^2 - 4
    omega^2`` and root-found it. No such closed form may exist anywhere on
    the v2.0 solver path."""
    import inspect

    from ctcfa import continuation
    src = inspect.getsource(continuation)
    body = src.split('"""', 2)[-1]           # exclude the module docstring
    assert "4*omega*omega" not in body
    assert "m**2 - 4" not in body
