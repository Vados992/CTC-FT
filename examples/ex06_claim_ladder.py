"""
EXAMPLE 6 -- Scope and the claim ladder: what the system refuses to say.

WHAT THIS UNIFIES
    v1.0 stated the claim ladder in Appendix D and the scope rules in
    sections 2, 9.2 and 14, and enforced neither in code. Nothing stopped
    ``global_model_test`` from returning a bare ``CTC_PROVED`` for a REDUCED
    predicate, and nothing stopped a sector-clear result from being read as a
    no-CTC proof.

    In v2.0 both are types. This example is mostly a list of things the
    architecture REFUSES to do, because for a falsification architecture the
    refusals are the product.

DATA REQUIRED
    nothing beyond the registry; this is the evidence logic, not physics.
"""

from _common import banner, section

from ctcfa import registry as R
from ctcfa.causality import (compact_orbit_test, compact_orbit_verdict,
                             global_predicate_verdict)
from ctcfa.status import (ClaimLevel, ClaimScope, CTCStatus, Gate, GateRecord,
                          GateResult, OrbitResult, OrbitStatus, OverclaimError,
                          Verdict)


def main() -> None:
    banner(6, "Scope, the claim ladder, and the refusals")

    section("1. A clear sector is UNRESOLVED, never NO_CTC_CERTIFIED")
    v = compact_orbit_verdict(R.get("schwarzschild"),
                              R.get("schwarzschild").default_point)
    print(f" Schwarzschild azimuthal sector margin = {v.margin:+.6e}")
    print(f" status = {v.status.value}    (NOT NO_CTC_CERTIFIED)")
    print(f" reason = {v.reason[:150]}")

    section("2. A reduced predicate cannot be printed at global scope")
    m = R.get("time_shifted_wormhole")
    v = global_predicate_verdict(m, m.default_point)
    print(f" {v.render()}")
    print(f" scope is {v.scope.value}: a route-closure model under declared")
    print(f" synchronisation assumptions, not a theorem about wormholes.")

    section("3. UNRESOLVED is absorbing")
    u = Verdict(CTCStatus.UNRESOLVED, ClaimScope.SECTOR, float('nan'),
                 "loop_search", "optimiser found nothing")
    try:
         u.promote(ClaimScope.GLOBAL, "verified certificate")
         print(" ERROR: promotion succeeded (this would be a bug)")
    except OverclaimError as exc:
         print(f" refused: {exc}")

    section("4. A negative result needs a named certificate to go global")
    n = Verdict(CTCStatus.NO_CTC_CERTIFIED, ClaimScope.DECLARED_DOMAIN, 1.0,
                 "temporal_function", "verified on a box")
    try:
         n.promote(ClaimScope.GLOBAL, "it looks fine")
         print(" ERROR: promotion succeeded")
    except OverclaimError as exc:
         print(f" refused: {str(exc)[:130]}")
    ok = n.promote(ClaimScope.GLOBAL, "verified temporal-function certificate")
    print(f" accepted with a certificate: scope = {ok.scope.value}")

    section("5. The claim ladder blocks statements above the evidence")
    g = GateRecord()
    g.set(Gate.V0_PARSE, GateResult.PASS)
    g.set(Gate.V1_ALGEBRA, GateResult.PASS)
    g.set(Gate.V4_CTC_WITNESS, GateResult.PASS)
    v = Verdict(CTCStatus.CTC_PROVED, ClaimScope.DECLARED_DOMAIN, -20.55,
                 "compact_orbit", "explicit closed timelike orbit", "godel",
                 gates=g)
    print(f" highest permitted level: {v.permitted_level().name}")
    print(f" {v.render()}")
    for want in (ClaimLevel.L5_ENERGY, ClaimLevel.L6_CONSTRUCTIBILITY,
                  ClaimLevel.L7_STABILITY, ClaimLevel.L8_SEMICLASSICAL):
        try:
             v.render(want)
             print(f" ERROR: {want.name} was allowed")
        except OverclaimError as exc:
             print(f" refused {want.name:24s}: {str(exc)[-70:]}")

    section("6. L9 is unreachable by construction")
    g2 = GateRecord()
    for gate in Gate:
        g2.set(gate, GateResult.PASS)
    print(f" every gate PASS -> highest permitted level = "
          f"{g2.highest_permitted_level().name}")
    print(" L9 (quantum-gravity consistency) is not emittable because no")
    print(" accepted complete theory exists to certify it against.")

    section("7. External specifications refuse to produce a metric")
    for mid in ("tipler_cylinder", "ori_2007_full", "paired_warp"):
        try:
             R.get(mid).require_metric()
             print(f" ERROR: {mid} returned a metric")
        except ValueError as exc:
             print(f" {mid:20s} -> {str(exc)[:110]}")

    section("8. The claim-level vocabulary actually in use")
    for lvl in ClaimLevel:
        from ctcfa.status import CLAIM_TEXT
        mark = " (unreachable)" if lvl is ClaimLevel.L9_QUANTUM_GRAVITY else ""
        print(f" {lvl.name:<26} {CLAIM_TEXT[lvl]}{mark}")


if __name__ == "__main__":
    main()
