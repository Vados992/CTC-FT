"""Convention pinning: the sign and normalisation choices that a drift would
silently corrupt. Each test here corresponds to a v1.0 defect or failure
mode that could otherwise recur unnoticed."""

import numpy as np
import pytest
import sympy as sp

from ctcfa import registry as R
from ctcfa.symbolic import (christoffel, einstein_tensor, kretschmann,
                            ricci_scalar, ricci_tensor)


def test_schwarzschild_is_ricci_flat():
    m = R.get("schwarzschild")
    assert ricci_tensor(m.metric, m.coords).is_zero_matrix


def test_schwarzschild_kretschmann_exact():
    """K = 48 M^2 / r^6 pins the Riemann normalisation and every index order."""
    m = R.get("schwarzschild")
    M, r = m.symbol("M"), m.symbol("r")
    K = kretschmann(m.metric, m.coords)
    assert sp.simplify(K - 48 * M ** 2 / r ** 6) == 0


def test_de_sitter_ricci_scalar_is_four_lambda():
    """R = 4 Lambda pins the Einstein-tensor sign convention in 4D."""
    m = R.get("de_sitter_static")
    L = m.symbol("L")
    R4 = ricci_scalar(m.metric, m.coords)
    assert sp.simplify(R4 - 4 * (3 / L ** 2)) == 0


def test_ads_ricci_scalar_is_negative():
    m = R.get("ads4_universal_cover")
    L = m.symbol("L")
    R4 = ricci_scalar(m.metric, m.coords)
    assert sp.simplify(R4 - 4 * (-3 / L ** 2)) == 0


def test_kerr_g_t_phi_sign_is_negative():
    """v1.0 defect D-01.

    In Boyer-Lindquist with the (-,+,+,+) signature the frame-dragging term is
    ``-(4 M a r sin^2 theta / Sigma) dt dphi``, so ``g_t_phi < 0`` for
    ``M, a, r > 0``. The v1.0 registry table printed the opposite sign.
    """
    m = R.get("kerr")
    sub = {m.symbol("M"): 1.0, m.symbol("a"): 0.9, m.symbol("r"): 4.0,
           m.symbol("theta"): np.pi / 2}
    gtph = float(sp.N(m.metric[0, 3].subs(sub)))
    assert gtph < 0
    assert abs(gtph - (-2 * 1.0 * 0.9 * 4.0 * 1.0 / 16.0)) < 1e-12


def test_kerr_reduces_to_schwarzschild_when_a_zero():
    m = R.get("kerr")
    a = m.symbol("a")
    g = sp.simplify(m.metric.subs(a, 0))
    s = R.get("schwarzschild")
    assert sp.simplify(g - s.metric).is_zero_matrix


def test_kerr_newman_reduces_to_kerr_when_q_zero():
    kn = R.get("kerr_newman")
    g = sp.simplify(kn.metric.subs(kn.symbol("Q"), 0))
    k = R.get("kerr")
    assert sp.simplify(g - k.metric).is_zero_matrix


def test_reissner_nordstrom_reduces_to_schwarzschild():
    rn = R.get("reissner_nordstrom")
    g = sp.simplify(rn.metric.subs(rn.symbol("Q"), 0))
    assert sp.simplify(g - R.get("schwarzschild").metric).is_zero_matrix


def test_godel_is_the_m2_equals_2omega2_member():
    base = R.get("godel_type")
    om = base.symbol("omega")
    target = sp.simplify(base.metric.subs(base.symbol("m"), sp.sqrt(2) * om))
    assert sp.simplify(R.get("godel").metric - target).is_zero_matrix


def test_krasnikov_tx_block_determinant_is_minus_one():
    """An exact algebraic identity of the Everett-Roman form: det of the (t,x)
    block equals -1 for every value of the regulator, so the chart is
    Lorentzian everywhere and the tube is not a chart pathology."""
    m = R.get("krasnikov")
    g = m.metric
    det2 = sp.simplify(g[0, 0] * g[1, 1] - g[0, 1] * g[1, 0])
    assert sp.simplify(det2 + 1) == 0


def test_misner_determinant_is_minus_one_everywhere():
    m = R.get("misner")
    assert sp.simplify(m.metric.det() + 1) == 0


def test_van_stockum_metric_is_symmetric():
    """v1.0 defect D-02: the kernel stored an asymmetric van Stockum matrix."""
    m = R.get("van_stockum")
    assert sp.simplify(m.metric - m.metric.T).is_zero_matrix


def test_every_registered_metric_is_symmetric():
    for m in R.models():
        if m.has_metric:
            assert sp.simplify(m.metric - m.metric.T).is_zero_matrix, m.id


def test_van_stockum_two_block_determinant():
    """det of the (t,phi) block is -r^2, so the chart stays Lorentzian even
    where g_phi_phi < 0 -- the CTC region is not a signature failure."""
    m = R.get("van_stockum")
    r = m.symbol("r")
    g = m.metric
    det2 = sp.simplify(g[0, 0] * g[2, 2] - g[0, 2] * g[2, 0])
    assert sp.simplify(det2 + r ** 2) == 0
