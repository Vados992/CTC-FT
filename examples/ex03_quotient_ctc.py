"""
EXAMPLE 3 -- Flat quotients: Misner and Grant space from the deck group.

WHAT THIS UNIFIES
    Misner space, Grant space and the polarised Misner-type spacetimes that
    dominate the modern causal-classification literature are quotients of
    FLAT Minkowski space by an isometry that is not a coordinate translation.
    v1.0 had no engine for them: its Misner entry used an ad-hoc chart and
    nothing could express a boost identification at all.

    Here the same spacetime is computed twice, from two completely different
    representations, and the answers are checked against each other and
    against a closed form. Then the engine locates the polarised
    hypersurfaces -- the loci where a quantum field's two-point function
    diverges -- automatically, which is a result v1.0 had no path to.

DATA REQUIRED
    the covering metric (flat, verified); the deck generators with their
    inverses; the parameter values (rapidity b, translation alpha); a test
    point. Nothing about the quotient chart is needed.
"""

from _common import banner, section

import math
import sympy as sp

from ctcfa import registry as R
from ctcfa.causality import compact_orbit_test
from ctcfa.quotient_ctc import (certify_flat, deck_orbit_test,
                                exact_orbit_interval, polarized_hypersurfaces,
                                winding_spectrum)


def main() -> None:
    banner(3, "Flat quotients: Misner, Grant, and the polarised hypersurfaces")

    section("Step 1 -- flatness of the covering is VERIFIED, not assumed")
    for mid in ("misner_covering", "grant_space", "minkowski_time_periodic"):
        c = certify_flat(R.get(mid))
        print(f" {mid:26s} Riemann = {c.residual}    flat = {c.flat}")

    section("Step 2 -- exact closed form for the winding-n closed geodesic")
    for mid, n in (("misner_covering", 1), ("misner_covering", 2),
                   ("grant_space", 1), ("grant_space", 2)):
        e = sp.simplify(exact_orbit_interval(R.get(mid), n))
        print(f" {mid:20s} n={n}: interval^2 = {sp.factor(sp.expand(e))}")

    section("Step 3 -- engine vs closed form, point by point (Misner covering)")
    b = 1.0
    print(f" {'t':>6}{'x':>8}{'engine q':>22}{'2(cosh b -1)(t^2-x^2)':>26}{'status':>16}")
    for t, x in [(0.0, 0.5), (0.0, 1.0), (0.0, 2.0), (0.4, 0.5), (0.5, 0.5),
                 (2.0, 0.5)]:
        d = deck_orbit_test(R.get("misner_covering"),
                            {"t": t, "x": x, "y": 0.0, "z": 0.0, "b": b})
        an = 2 * (math.cosh(b) - 1) * (t * t - x * x)
        print(f" {t:>6.2f}{x:>8.2f}{d.margin:>22.15f}{an:>26.15f}"
              f"{d.status.value.replace('SECTOR_',''):>16}")

    section("Step 4 -- the same spacetime through the Misner chart")
    print(" In the covering the sign of the interval is that of (t^2 - x^2);")
    print(" in the quotient chart it is the sign of -T. The two must agree")
    print(" under T = x^2 - t^2.")
    print(f" {'t':>6}{'x':>8}{'T = x^2-t^2':>14}{'chart margin':>18}"
          f"{'covering margin':>20}{'signs agree':>14}")
    for t, x in [(0.0, 0.5), (0.0, 2.0), (0.4, 0.5), (2.0, 0.5)]:
        T = x * x - t * t
        c = compact_orbit_test(R.get("misner"),
                               {"T": T, "psi": 0.0, "y": 0.0, "z": 0.0})
        d = deck_orbit_test(R.get("misner_covering"),
                            {"t": t, "x": x, "y": 0.0, "z": 0.0, "b": b})
        agree = (c.margin < 0) == (d.margin < 0)
        print(f" {t:>6.2f}{x:>8.2f}{T:>14.4f}{c.margin:>18.6e}"
              f"{d.margin:>20.6e}{str(agree):>14}")

    section("Step 5 -- Grant space: winding spectrum at one point")
    m = R.get("grant_space")
    base = {"t": 0.0, "x": 0.8, "y": 0.0, "z": 0.0, "b": 1.0, "alpha": 1.0}
    print(" at x = 0.8 (t = 0, b = 1, alpha = 1):")
    for row in winding_spectrum(m, base, 6):
        print(f"      n = {row['n']}    interval^2 = {row['interval_squared']:+.6e}"
              f"   {row['character']}")
    print()
    print(" The winding-1 and winding-2 loops are spacelike here; the")
    print(" winding-3 loop is timelike. The shortest chronology-violating")
    print(" loop through a point is therefore a computed quantity, not an")
    print(" assumption.")

    section("Step 6 -- polarised hypersurfaces located from the deck group")
    rows = polarized_hypersurfaces(m, base, "x", 0.05, 3.0, max_power=6)
    seen = {}
    for r in rows:
        if r["value"] > 0:
            seen.setdefault(r["winding"], r["value"])
    print(f" {'winding n':>10}{'x_n (engine)':>22}"
          f"{'n alpha / sqrt(2(cosh nb -1))':>32}{'error':>12}")
    for n in sorted(seen):
        an = math.sqrt(n * n * 1.0 / (2 * (math.cosh(n * 1.0) - 1)))
        print(f" {n:>10}{seen[n]:>22.15f}{an:>32.15f}{abs(seen[n]-an):>12.2e}")
    print()
    print(" x_n decreases monotonically to zero, so the union over all n is")
    print(" the whole region x^2 > t^2: the translation alpha does NOT move")
    print(" the chronology horizon, which stays at x^2 = t^2 exactly as in")
    print(" Misner space. What alpha creates is this discrete family of")
    print(" polarised hypersurfaces accumulating on the horizon -- precisely")
    print(" where the renormalised stress tensor of a quantum field diverges.")


if __name__ == "__main__":
    main()
