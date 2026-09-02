"""
EXAMPLE 4 -- Certified energy conditions, and the counterexample that matters.

WHAT THIS UNIFIES
    v1.0 sampled 256 directions on a Fibonacci sphere and called the result
    diagnostic, which meant no energy statement could ever be certified.
    v2.0 computes the same quantities two ways, exactly:

        Route A   Hawking-Ellis classification -> closed-form criteria
        Route B   exact trust-region minimisation on the null/timelike cone

    and cross-checks them. It then integrates ANEC along a real null
    geodesic instead of a supplied list of numbers.

    The scientific payload is the pair of results at the bottom: the Ori 2005
    core and van Stockum are chronology-violating spacetimes whose matter
    does NOT violate the null energy condition. Any architecture that treats
    "CTC" and "exotic matter" as synonyms is wrong, and this is the
    computation that shows it.

DATA REQUIRED
    the metric; the cosmological constant; the evaluation point. The stress
    tensor is obtained ON SHELL from G_ab + Lambda g_ab = 8 pi T_ab -- it is
    never assumed.
"""

from _common import banner, section

import math
import numpy as np

from ctcfa import registry as R
from ctcfa.energy import energy_report
from ctcfa.geodesics import anec_on_geodesic


def main() -> None:
    banner(4, "Energy conditions: exact classification and a real ANEC integral")

    section("Hawking-Ellis type and the four conditions, on shell")
    print(f" {'model':<26}{'type':<10}{'rho':>10}"
          f"{'NEC':>12}{'WEC':>12}{'SEC':>12}{'DEC':>12}")
    order = ["minkowski", "schwarzschild", "kerr", "reissner_nordstrom",
             "kerr_newman", "de_sitter_static", "ads4_universal_cover",
             "misner", "ori_2005", "ori_2007_core", "godel", "van_stockum",
             "spinning_cosmic_string", "morris_thorne", "alcubierre",
             "krasnikov"]
    reports = {}
    for mid in order:
        rep = energy_report(R.get(mid))
        reports[mid] = rep
        rho = " -" if rep.he.rho is None else f"{rep.he.rho:+.4f}"
        short = {"SATISFIED": "OK", "SATISFIED_DIAGNOSTIC": "OK*",
                 "MARGINAL": "marg", "VIOLATED": "VIOL"}
        print(f" {mid:<26}{rep.he.type.value.replace('TYPE_',''):<10}{rho:>10}"
              f"{short.get(rep.nec_status, rep.nec_status):>12}"
              f"{short.get(rep.wec_status, rep.wec_status):>12}"
              f"{short.get(rep.sec_status, rep.sec_status):>12}"
              f"{short.get(rep.dec_status, rep.dec_status):>12}")

    section("Cross-check between the two exact routes")
    for mid in ("schwarzschild", "kerr", "godel", "van_stockum", "morris_thorne"):
        rep = reports[mid]
        x = rep.cross_check_error
        print(f" {mid:<26} |Route A - Route B| = "
              f"{'n/a (non Type I)' if x is None else f'{x:.3e}'}"
              f"   uncertainty {rep.uncertainty:.2e}")

    section("The counterexample: chronology violation WITHOUT exotic matter")
    for mid in ("ori_2005", "ori_2007_core", "van_stockum", "godel"):
        rep = reports[mid]
        m = R.get(mid)
        print(f" {mid}")
        print(f"       causal status : {(m.expected_at_default or m.expected).value}")
        print(f"       matter        : {m.matter.label}")
        print(f"       rho           : {rep.he.rho:+.6f}")
        print(f"       NEC margin    : {rep.nec_margin:+.6e} [{rep.nec_status}]")
        print(f"       SEC margin    : {rep.sec_margin:+.6e} [{rep.sec_status}]")
    print()
    print(" Ori 2005 has an exactly vacuum core (T_ab = 0 verified, not")
    print(" assumed) and a chronology-violating compact direction.")
    print(" van Stockum has strictly positive energy density and satisfies")
    print(" NEC, WEC, SEC and DEC while containing closed timelike curves.")
    print(" The inference 'CTC implies energy-condition violation' is false.")

    section("Morris-Thorne: the converse control")
    mt = energy_report(R.get("morris_thorne"),
                       {"t": 0, "r": 1.05, "theta": math.pi / 2, "phi": 0,
                        "r0": 1.0})
    print(f" a static traversable wormhole near the throat:")
    print(f"      rho = {mt.he.rho:+.6f}   NEC = {mt.nec_margin:+.6e} "
          f"[{mt.nec_status}]")
    print(" It violates the null energy condition and contains NO closed")
    print(" timelike curve (its temporal-function certificate is exact).")
    print(" So the two properties are logically independent in both")
    print(" directions.")

    section("ANEC along an integrated null geodesic")
    for mid, pt, lam in (
            ("morris_thorne", {"t": 0, "r": 1.2, "theta": math.pi / 2,
                                "phi": 0, "r0": 1.0}, 3.0),
            ("schwarzschild", None, 10.0)):
        m = R.get(mid)
        res, geo = anec_on_geodesic(m, pt or m.default_point, lam_max=lam,
                                     n_out=201)
        print(f" {mid}")
        print(f"       geodesic kind        : {geo.kind}")
        print(f"       norm drift g(k,k)    : {geo.norm_drift:.3e} "
              f"(independent accuracy check)")
        print(f"       affine range         : [{geo.affine[0]:.1f}, "
              f"{geo.affine[-1]:.1f}], {res.n_samples} samples")
        print(f"       int T_ab k^a k^b dl : {res.value:+.6e} "
              f"+- {res.error_estimate:.1e}")
        print(f"       status               : {res.status}")


if __name__ == "__main__":
    main()
