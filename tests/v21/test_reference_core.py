from __future__ import annotations

import math
from contextlib import contextmanager

import numpy as np

from ctcfa_v21_ref.adapters import (
    EFTValidityReport,
    MINKOWSKI_INVERSE,
    kessence_characteristic,
    kessence_constant_bundle,
    metric_mode,
)
from ctcfa_v21_ref.cones import common_temporal_certificate, temporal_single_valued
from ctcfa_v21_ref.model import (
    CTCWitness,
    CharacteristicMode,
    ClaimScope,
    ConeBundle,
    FieldEquationReport,
    ImplementationState,
    ModeKind,
    SolutionSpec,
    TheorySpec,
    TopologySpec,
    VerdictStatus,
)
from ctcfa_v21_ref.pipeline import classify
from ctcfa_v21_ref.topology import (
    axial_witness,
    killing_block,
    killing_orbit_closure,
    periodic_time_witness,
)


class _Approx:
    def __init__(self, expected: float, abs_tol: float = 1e-12, rel_tol: float = 1e-9):
        self.expected = float(expected)
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol

    def __eq__(self, actual: object) -> bool:
        try:
            return math.isclose(float(actual), self.expected, abs_tol=self.abs_tol, rel_tol=self.rel_tol)
        except (TypeError, ValueError):
            return False


def approx(expected: float) -> _Approx:
    return _Approx(expected)


@contextmanager
def raises(exc_type):
    try:
        yield
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def theory() -> TheorySpec:
    return TheorySpec(
        theory_id="gr.v1",
        fields=("metric:g",),
        coupling_dimensions={},
        field_equation_adapter="vacuum_gr",
        principal_symbol_adapter="metric_cone",
        implementation=ImplementationState.VERIFIED,
        provenance=("Einstein 1915",),
    )


def solution(topology: TopologySpec | None = None) -> SolutionSpec:
    return SolutionSpec(
        solution_id="minkowski.control",
        theory_id="gr.v1",
        domain="all R4",
        topology=topology or TopologySpec(),
        implementation=ImplementationState.VERIFIED,
        provenance=("analytic control",),
    )


def field_report(residual: float = 0.0, validity: bool = True) -> FieldEquationReport:
    return FieldEquationReport(
        checked_equations=("G_mu_nu=0",),
        max_abs_residual=residual,
        tolerance=1e-12,
        domain_covered=True,
        validity_passed=validity,
    )


def metric_bundle(solution_id: str = "minkowski.control") -> ConeBundle:
    return ConeBundle(solution_id, (metric_mode(),), ("metric",), True)


def test_valid_theory_and_solution():
    assert theory().validate()[0]
    assert solution().validate_for(theory())[0]


def test_theory_requires_metric_field():
    bad = TheorySpec("bad", ("scalar:phi",), {}, field_equation_adapter="x", principal_symbol_adapter="y", provenance=("p",))
    ok, errors = bad.validate()
    assert not ok and "metric:g field is required" in errors


def test_solution_theory_mismatch_is_invalid():
    bad = SolutionSpec("s", "other", "D", provenance=("p",))
    verdict = classify(theory(), bad, field_report(), metric_bundle())
    assert verdict.status is VerdictStatus.INVALID_SPEC


def test_off_shell_precedes_causality():
    verdict = classify(theory(), solution(), field_report(1e-4), metric_bundle(), temporal_gradient=(1, 0, 0, 0))
    assert verdict.status is VerdictStatus.OFF_SHELL


def test_outside_eft_precedes_causality():
    verdict = classify(theory(), solution(), field_report(validity=False), metric_bundle(), temporal_gradient=(1, 0, 0, 0))
    assert verdict.status is VerdictStatus.OUTSIDE_EFT


def test_incomplete_bundle_is_not_implemented():
    incomplete = ConeBundle("minkowski.control", (metric_mode(),), ("metric", "scalar"), True)
    verdict = classify(theory(), solution(), field_report(), incomplete)
    assert verdict.status is VerdictStatus.NOT_IMPLEMENTED


def test_minkowski_dt_is_temporal():
    cert = common_temporal_certificate(metric_bundle(), (1, 0, 0, 0), TopologySpec())
    assert cert.proved and cert.mode_margins["metric"] == approx(-1.0)


def test_periodic_time_makes_t_non_single_valued():
    topology = TopologySpec(time_period=2 * math.pi)
    assert not temporal_single_valued(np.array([1.0, 0.0, 0.0, 0.0]), topology)


def test_periodic_angle_makes_phi_non_single_valued():
    assert not temporal_single_valued(np.array([0.0, 0.0, 0.0, 1.0]), TopologySpec())


def test_periodic_time_has_exact_ctc_witness():
    w = periodic_time_witness(-1.0, TopologySpec(time_period=2 * math.pi))
    assert w.proved and w.scope is ClaimScope.GLOBAL


def test_unwrapped_time_has_no_periodic_time_witness():
    assert not periodic_time_witness(-1.0, TopologySpec()).proved


def test_axial_negative_norm_is_ctc_witness():
    assert axial_witness(-0.25, TopologySpec()).proved


def test_axial_positive_norm_is_not_ctc_witness():
    assert not axial_witness(+0.25, TopologySpec()).proved


def test_killing_block_determinant_is_sector_diagnostic():
    row = killing_block(-1.0, 0.2, 4.0)
    assert row["lorentzian_killing_plane"] and not row["axial_ctc_candidate"]


def test_helical_orbit_is_open_with_unwrapped_time():
    c = killing_orbit_closure(1.0, 2.0, TopologySpec())
    assert not c.closed and c.delta_t == approx(math.pi)


def test_helical_orbit_closes_with_commensurate_time_quotient():
    c = killing_orbit_closure(1.0, 2.0, TopologySpec(time_period=math.pi))
    assert c.closed


def test_pure_axial_orbit_closes():
    assert killing_orbit_closure(0.0, 1.0, TopologySpec()).closed


def test_kessence_beta_zero_is_conformal_metric():
    z, x, px, cs2 = kessence_characteristic(2.0, 0.0, np.array([1.0, 0.0, 0.0, 0.0]))
    assert x == approx(0.5)
    assert px == approx(2.0)
    assert cs2 == approx(1.0)
    assert np.allclose(z, 2.0 * MINKOWSKI_INVERSE)


def test_kessence_sound_speed_matches_formula():
    _, x, px, cs2 = kessence_characteristic(1.0, 0.2, np.array([1.0, 0.0, 0.0, 0.0]))
    expected = px / (px + 4.0 * 0.2 * x)
    assert cs2 == approx(expected)


def test_kessence_regular_bundle_is_hyperbolic():
    eft = EFTValidityReport(100.0, 0.01)
    bundle = kessence_constant_bundle("k.control", 1.0, 0.1, np.array([0.5, 0.0, 0.0, 0.0]), eft)
    assert bundle.complete and bundle.hyperbolic


def test_eft_gate_rejects_low_scale_separation():
    assert not EFTValidityReport(2.0, 0.01).passed


def test_eft_gate_rejects_large_derivative_ratio():
    assert not EFTValidityReport(100.0, 0.5).passed


def test_degenerate_scalar_mode_is_ill_posed():
    zero = CharacteristicMode("scalar", ModeKind.SCALAR, np.zeros((4, 4)), hyperbolic=False)
    bundle = ConeBundle("minkowski.control", (metric_mode(), zero), ("metric", "scalar"), True)
    verdict = classify(theory(), solution(), field_report(), bundle)
    assert verdict.status is VerdictStatus.ILL_POSED


def test_any_mode_witness_proves_solution_ctc():
    w = CTCWitness("metric", True, True, -0.5, ClaimScope.SECTOR, ("analytic",))
    verdict = classify(theory(), solution(), field_report(), metric_bundle(), witnesses=(w,))
    assert verdict.status is VerdictStatus.CTC_PROVED


def test_positive_solution_verdict_can_widen_only_existentially():
    w = CTCWitness("metric", True, True, -0.5, ClaimScope.SECTOR)
    verdict = classify(theory(), solution(), field_report(), metric_bundle(), witnesses=(w,))
    wide = verdict.theory_existence_claim()
    assert wide.scope is ClaimScope.THEORY_EXISTENCE
    assert "existential" in wide.notes[-1]


def test_negative_solution_verdict_cannot_become_theory_claim():
    verdict = classify(theory(), solution(), field_report(), metric_bundle(), temporal_gradient=(1, 0, 0, 0))
    assert verdict.status is VerdictStatus.ALL_CONES_CAUSAL_CERTIFIED
    with raises(ValueError):
        verdict.theory_existence_claim()


def test_common_temporal_certificate_requires_all_modes():
    outward = CharacteristicMode("scalar", ModeKind.SCALAR, np.diag([1.0, -1.0, 1.0, 1.0]))
    bundle = ConeBundle("minkowski.control", (metric_mode(), outward), ("metric", "scalar"), True)
    verdict = classify(theory(), solution(), field_report(), bundle, temporal_gradient=(1, 0, 0, 0))
    # Both modes are individually Lorentzian, but dt is not temporal for the
    # second mode.  One failed candidate gradient cannot prove that the dual
    # cone intersection is empty, so the refusal state is UNRESOLVED.
    assert verdict.status is VerdictStatus.UNRESOLVED


def test_no_witness_no_certificate_is_unresolved():
    verdict = classify(theory(), solution(), field_report(), metric_bundle())
    assert verdict.status is VerdictStatus.UNRESOLVED


def test_temporal_certificate_reports_every_mode():
    eft = EFTValidityReport(100.0, 0.01)
    bundle = kessence_constant_bundle("k.control", 1.0, 0.1, np.array([0.5, 0.0, 0.0, 0.0]), eft)
    cert = common_temporal_certificate(bundle, (1, 0, 0, 0), TopologySpec())
    assert cert.proved and set(cert.mode_margins) == {"metric", "scalar"}


def test_mode_validation_rejects_asymmetry():
    a = np.eye(4)
    a[0, 1] = 2.0
    mode = CharacteristicMode("bad", ModeKind.MIXED, a)
    assert not mode.validate()[0]


def test_field_report_requires_domain_coverage():
    report = FieldEquationReport(("E=0",), 0.0, 1e-12, False)
    assert not report.on_shell
