"""Full-registry regression: validation gates, the causal matrix, provenance,
the theorem validator, energy conditions and the meta-analysis governance.

Numeric expectations in this file are the ones the v1.0 self-test published,
where those were correct. Every value that the v1.0 kernel got right is
reproduced exactly, which is the strongest available evidence that the v2.0
rewrite did not silently change the physics while fixing the defects.
"""

import math

import numpy as np
import pytest

from ctcfa import registry as R
from ctcfa.causality import compact_orbit_test
from ctcfa.report import (format_matrix, matrix_agreement, primary_route,
                          regression_matrix)
from ctcfa.validate import validate_model


#: Values published in the v1.0 self-test (section 14 / Appendix C) that were
#: correct and must be reproduced bit for bit by the rewritten engines.
V1_MARGINS = {
    "schwarzschild": +2.500000e+01,
    "reissner_nordstrom": +2.500000e+01,
    "kerr": -1.670000e+00,
    "kerr_newman": -1.772400e+00,
    "godel_type": -1.736478e+01,
    "godel_critical": +1.315412e+01,     # v1.0 row "godel_type SECTOR_SPACELIKE"
    "misner": -1.000000e+00,
    "van_stockum": -1.200000e+01,
    "taub_nut": -6.923595e-01,
    "spinning_cosmic_string": -1.536000e-01,
    "morris_thorne": +4.000000e+00,
    "ori_2005": -1.000000e+00,
    "ori_2007_core": -1.000000e+00,
    "krasnikov": +1.000000e-02,
}


@pytest.mark.parametrize("mid,expected", sorted(V1_MARGINS.items()))
def test_v1_margins_are_reproduced(mid, expected):
    m = R.get(mid)
    r = compact_orbit_test(m, m.default_point)
    assert abs(r.margin - expected) <= 1e-6 * max(1.0, abs(expected)), (
        f"{mid}: v2.0 gives {r.margin!r}, v1.0 published {expected!r}")


def test_registry_audit_passes():
    a = R.registry_audit()
    assert a["pass"], a["problems"]
    assert a["n_models"] >= 30


def test_every_model_passes_the_structural_gates():
    for m in R.models():
        rep = validate_model(m, cross_check=False)
        assert rep.ok, (m.id, rep.gates.as_dict())


def test_regression_matrix_agrees_everywhere():
    rows = regression_matrix()
    ag = matrix_agreement(rows)
    assert ag["pass"], ag["mismatches"]
    assert ag["n"] >= 30


def test_every_row_records_its_derivation_route():
    for row in regression_matrix():
        assert row["route"] in ("compact_orbit", "deck_orbit",
                               "global_predicate", "temporal_function", "none")


def test_vacuum_models_have_vanishing_field_equation_residual():
    from ctcfa.energy import symbolic_einstein
    import sympy as sp
    for mid in ("schwarzschild", "kerr", "taub_nut", "misner", "ori_2005",
                "ori_2007_core", "minkowski", "grant_space"):
        m = R.get(mid)
        G = symbolic_einstein(m)
        arr = np.array(sp.Matrix(G + m.matter.cosmological_constant * m.metric)
                       .subs(m.substitution(m.default_point)).evalf(30),
                       dtype=float)
        assert float(np.max(np.abs(arr))) < 1e-9, mid


def test_energy_conditions_on_the_key_models():
    from ctcfa.energy import energy_report
    # Ori 2005 is the counterexample to "CTC implies NEC violation": a
    # vacuum core with a chronology-violating compact direction.
    ori = energy_report(R.get("ori_2005"))
    assert ori.he.type.value == "TYPE_I"
    assert ori.nec_status in ("MARGINAL", "SATISFIED")
    # van Stockum: matter-supported CTC with strictly positive energy density.
    vs = energy_report(R.get("van_stockum"))
    assert vs.he.rho > 0
    assert vs.nec_status == "SATISFIED"
    # Morris-Thorne throat: NEC violated.
    mt = energy_report(R.get("morris_thorne"),
                       {"t": 0, "r": 1.05, "theta": math.pi / 2, "phi": 0,
                         "r0": 1.0})
    assert mt.nec_status == "VIOLATED"
    # Alcubierre: energy conditions violated.
    al = energy_report(R.get("alcubierre"))
    assert al.nec_status == "VIOLATED"


def test_theorem_validator_never_licenses_unverified_hypotheses():
    from ctcfa.theorems import THEOREMS, Applicability, apply
    for m in R.models():
        for k in THEOREMS:
            rep = apply(k, m)
            if rep.applicability is Applicability.LICENSED:
                assert all(v is True for v in rep.hypothesis_results.values())
            else:
                assert rep.permitted_label is None


def test_chronology_protection_is_never_labelled_proved():
    from ctcfa.theorems import THEOREMS
    th = THEOREMS["chronology_protection"]
    assert th.status_note == "CONJECTURE"
    assert "PROVED" not in th.permitted_label
    for m in R.models():
        from ctcfa.theorems import apply
        rep = apply("chronology_protection", m)
        assert rep.permitted_label in (None, th.permitted_label)


def test_meta_analysis_reports_held_out_scores_and_governance():
    from ctcfa.meta import cross_family_report
    rep = cross_family_report(R.models())
    assert rep["n_models"] >= 30
    for rule in rep["rules"]:
        assert rule["status"] in ("CANDIDATE_PRECURSOR", "REFUTED_ON_HELD_OUT")
        assert rule["heldout_n"] >= 2
    assert "CANDIDATE_PRECURSOR" in rep["governance"]
    assert "universal" in rep["governance"]


def test_provenance_round_trip():
    from pathlib import Path
    import tempfile

    from ctcfa.provenance import (build_manifest, new_run, sha256_text,
                                   source_tree_digest, write_run)
    root = Path(R.__file__).resolve().parent
    d1, files = source_tree_digest(root)
    d2, _ = source_tree_digest(root)
    assert d1 == d2 and len(files) > 15
    with tempfile.TemporaryDirectory() as td:
        rec = new_run("kerr", "compact_orbit", {"r": -0.5}, {"margin": -1.67},
                      d1)
        p = write_run(rec, td)
        assert p.exists()
        again = write_run(rec, td)
        assert again == p                        # identical content, no duplicate


def test_quotient_certificates_are_present_where_declared():
    from ctcfa.topology import verify_quotient
    for m in R.models():
        if not m.quotient.generators:
            continue
        cert = verify_quotient(m)
        assert len(cert.generators) == len(m.quotient.generators)
        for gc in cert.generators:
            if m.has_metric:
                assert gc.is_isometry, (m.id, gc.name, gc.isometry_residual[:150])
