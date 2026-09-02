"""Mandatory negative controls (v1.0 section 15.3, extended).

Each control targets one specific way the architecture could produce a false
positive or a false negative. A control that stops failing when the
architecture is broken is worthless, so several of these deliberately assert
what the system must *refuse* to say.
"""

import numpy as np
import pytest
import sympy as sp

from ctcfa import registry as R
from ctcfa.causality import (certify_temporal_function, compact_orbit_test,
                             compact_orbit_verdict)
from ctcfa.status import ClaimScope, CTCStatus, OrbitStatus
from ctcfa.topology import compare_global_structure


# -- 1. flat space must not produce a CTC ---------------------------------

def test_minkowski_is_certified_chronal_globally():
    c = certify_temporal_function(R.get("minkowski"))
    assert c.proved and c.status is CTCStatus.NO_CTC_CERTIFIED
    assert c.scope is ClaimScope.GLOBAL


def test_flrw_temporal_certificate_is_scale_factor_independent():
    """g^tt = -1 for ANY a(t) > 0, so cosmic time works without knowing a."""
    c = certify_temporal_function(R.get("flrw_flat"))
    assert c.proved
    assert c.expression == "1"


# -- 2. strong curvature and horizons must not imply a CTC ----------------

@pytest.mark.parametrize("mid", ["schwarzschild", "reissner_nordstrom",
                                 "de_sitter_static"])
def test_horizons_do_not_produce_ctc(mid):
    m = R.get(mid)
    r = compact_orbit_test(m, m.default_point)
    assert r.status is OrbitStatus.SECTOR_SPACELIKE


def test_reissner_cauchy_horizon_is_not_a_chronology_horizon():
    from ctcfa.horizons import horizon_report
    rep = horizon_report(R.get("reissner_nordstrom"), "r")
    kinds = {c["kind"] for c in rep["candidates"]}
    assert "KILLING_HORIZON" in kinds
    assert "CHRONOLOGY_HORIZON" not in kinds
    for c in rep["candidates"]:
        assert c["forbidden_inference"]


# -- 3. a sector clear is NOT a global no-CTC result ----------------------

def test_clear_sector_is_reported_as_unresolved_not_certified():
    m = R.get("schwarzschild")
    v = compact_orbit_verdict(m, m.default_point)
    assert v.status is CTCStatus.UNRESOLVED
    assert "NOT a global no-CTC result" in v.reason


# -- 4. topology, not geometry, decides global causality ------------------

def test_ads_pair_shares_one_metric_and_splits_on_the_deck_group():
    a, b = R.get("ads4_universal_cover"), R.get("ads4_periodic")
    cmp = compare_global_structure(a, b)
    assert cmp["identical_local_metric"]
    assert cmp["demonstrates_L3_gate"]
    ra = compact_orbit_test(a, a.default_point)
    rb = compact_orbit_test(b, b.default_point)
    assert ra.status is OrbitStatus.SECTOR_SPACELIKE
    assert rb.status is OrbitStatus.SECTOR_CTC


def test_minkowski_pair_shares_one_metric_and_splits_on_the_deck_group():
    cmp = compare_global_structure(R.get("minkowski"),
                                   R.get("minkowski_time_periodic"))
    assert cmp["identical_local_metric"]
    assert cmp["demonstrates_L3_gate"]


def test_multivalued_time_is_refused_as_a_temporal_function():
    for mid in ("ads4_periodic", "minkowski_time_periodic"):
        m = R.get(mid)
        c = certify_temporal_function(m, tau=m.symbol("t"))
        assert not c.proved
        assert c.method == "quotient_invariance"
        assert "single valued" in c.note


# -- 5. traversable wormhole is not a time machine ------------------------

def test_static_wormhole_is_chronal():
    c = certify_temporal_function(R.get("morris_thorne"))
    assert c.proved
    r = compact_orbit_test(R.get("morris_thorne"),
                           R.get("morris_thorne").default_point)
    assert r.status is OrbitStatus.SECTOR_SPACELIKE


# -- 6. effective superluminality is not a time machine -------------------

def test_single_alcubierre_bubble_is_chronal_on_its_chart():
    c = certify_temporal_function(R.get("alcubierre"))
    assert c.proved and c.expression == "1"


def test_single_krasnikov_tube_is_not_a_ctc():
    m = R.get("krasnikov")
    r = compact_orbit_test(m, m.default_point)
    assert r.status is OrbitStatus.SECTOR_SPACELIKE


# -- 7. a quotient of AdS is not automatically causality violating --------

def test_btz_exterior_chart_is_spacelike_in_the_azimuthal_sector():
    m = R.get("btz_rotating")
    r = compact_orbit_test(m, m.default_point)
    assert r.status is OrbitStatus.SECTOR_SPACELIKE


# -- 8. the CTC region is not created by expansion or curvature -----------

def test_expanding_cosmology_is_chronal():
    m = R.get("flrw_exponential")
    c = certify_temporal_function(m)
    assert c.proved


# -- 9. out-of-domain points are rejected, not answered -------------------

def test_point_outside_the_declared_domain_is_refused():
    m = R.get("schwarzschild")
    bad = dict(m.default_point)
    bad["r"] = 1.0                                # inside the horizon
    r = compact_orbit_test(m, bad)
    assert r.status is OrbitStatus.SECTOR_UNRESOLVED
    assert "domain" in r.reason


# -- 10. a non-Killing "generator" is refused -----------------------------

def test_non_killing_generator_is_rejected():
    from ctcfa.model import CompactDirection, MetricModel, Provenance
    t, r, ph, z = sp.symbols("t r phi z", real=True)
    # g_phiphi depends on phi, so d/dphi is NOT a Killing vector
    g = sp.diag(-1, 1, r ** 2 * (2 + sp.sin(ph)), 1)
    m = MetricModel(id="fake", title="non-Killing generator control",
                    coords=(t, r, ph, z), metric=g,
                    compact=(CompactDirection(2, "phi", 2 * sp.pi),),
                    default_point={"t": 0, "r": 1, "phi": 0, "z": 0},
                    provenance=Provenance(("control",)))
    res = compact_orbit_test(m, m.default_point)
    assert res.status is OrbitStatus.SECTOR_UNRESOLVED
    assert "Killing" in res.reason


# -- 11. asymmetric metrics cannot enter the registry ---------------------

def test_asymmetric_metric_is_refused_at_load():
    from ctcfa.model import MetricModel, Provenance
    t, x, y, z = sp.symbols("t x y z", real=True)
    g = sp.Matrix([[-1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    with pytest.raises(ValueError, match="not symmetric"):
        MetricModel(id="bad", title="asymmetric", coords=(t, x, y, z), metric=g,
                    provenance=Provenance(("control",)))


# -- 12. reduced predicates cannot be printed as global theorems ----------

def test_reduced_predicate_stays_at_reduced_scope():
    from ctcfa.causality import global_predicate_verdict
    m = R.get("time_shifted_wormhole")
    v = global_predicate_verdict(m, m.default_point)
    assert v.status is CTCStatus.CTC_PROVED
    assert v.scope is ClaimScope.REDUCED_MODEL
    assert "not" in v.reason.lower() or "NOT" in v.reason


def test_external_spec_refuses_to_produce_a_metric():
    for mid in ("tipler_cylinder", "ori_2007_full", "paired_warp",
                "gott_two_strings"):
        with pytest.raises(ValueError, match="refuses to fabricate"):
            R.get(mid).require_metric()
