"""The evidence logic itself: scope algebra, claim ladder, overclaim refusal."""

import pytest

from ctcfa.status import (ClaimLevel, ClaimScope, CTCStatus, Gate, GateRecord,
                          GateResult, OrbitResult, OrbitStatus, OverclaimError,
                          Verdict)


def _gates(*passed: Gate) -> GateRecord:
    g = GateRecord()
    for p in passed:
        g.set(p, GateResult.PASS)
    return g


def test_sector_spacelike_never_becomes_no_ctc_certified():
    """v1.0 REQ-005 / FM-002, now a type-level guarantee."""
    r = OrbitResult(OrbitStatus.SECTOR_SPACELIKE, 25.0, (1,), {}, "clear sector")
    v = r.to_verdict("schwarzschild")
    assert v.status is CTCStatus.UNRESOLVED
    assert v.status is not CTCStatus.NO_CTC_CERTIFIED


def test_sector_null_maps_to_critical():
    r = OrbitResult(OrbitStatus.SECTOR_NULL, 0.0, (1,), {}, "marginal")
    assert r.to_verdict("misner").status is CTCStatus.CRITICAL


def test_unresolved_cannot_be_promoted():
    v = Verdict(CTCStatus.UNRESOLVED, ClaimScope.SECTOR, float("nan"),
                "search", "nothing found")
    with pytest.raises(OverclaimError):
        v.promote(ClaimScope.GLOBAL, "certificate")


def test_negative_result_needs_a_certificate_to_go_global():
    v = Verdict(CTCStatus.NO_CTC_CERTIFIED, ClaimScope.DECLARED_DOMAIN, 1.0,
                "temporal_function", "ok")
    with pytest.raises(OverclaimError):
        v.promote(ClaimScope.GLOBAL, "looks fine to me")
    ok = v.promote(ClaimScope.GLOBAL, "verified temporal-function certificate")
    assert ok.scope is ClaimScope.GLOBAL


def test_claim_ladder_blocks_unsupported_levels():
    v = Verdict(CTCStatus.CTC_PROVED, ClaimScope.SECTOR, -1.0, "compact_orbit",
                "witness", "godel",
                gates=_gates(Gate.V0_PARSE, Gate.V1_ALGEBRA, Gate.V4_CTC_WITNESS))
    assert v.permitted_level() is ClaimLevel.L2_CTC_VERIFIED
    v.render()                                       # allowed at its own level
    with pytest.raises(OverclaimError):
        v.render(ClaimLevel.L6_CONSTRUCTIBILITY)
    with pytest.raises(OverclaimError):
        v.render(ClaimLevel.L7_STABILITY)


def test_l9_is_unreachable_by_construction():
    g = GateRecord()
    for gate in Gate:
        g.set(gate, GateResult.PASS)
    assert g.highest_permitted_level() is not ClaimLevel.L9_QUANTUM_GRAVITY


def test_scope_composition_never_strengthens():
    v = Verdict(CTCStatus.CTC_PROVED, ClaimScope.GLOBAL, -1.0, "x", "y")
    assert v.restrict(ClaimScope.SECTOR).scope is ClaimScope.SECTOR
    assert v.restrict(ClaimScope.SECTOR).restrict(
        ClaimScope.GLOBAL).scope is ClaimScope.SECTOR


def test_verdict_render_includes_scope_and_level():
    v = Verdict(CTCStatus.CTC_PROVED, ClaimScope.SECTOR, -1.0, "compact_orbit",
                "witness", "kerr",
                gates=_gates(Gate.V0_PARSE, Gate.V1_ALGEBRA, Gate.V4_CTC_WITNESS))
    line = v.render()
    assert "scope=SECTOR" in line
    assert "level=L2_CTC_VERIFIED" in line
    assert "CTC_PROVED" in line
