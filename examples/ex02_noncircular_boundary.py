"""
EXAMPLE 2 -- Deriving a causal phase boundary WITHOUT the answer.

WHAT THIS UNIFIES
    v1.0 "recovered" the Godel-type boundary m = 2 omega by root-finding the
    hard-coded expression m^2 - 4 omega^2. That is failure mode FM-016,
    circular validation, committed by the architecture that named it.

    v2.0 derives the boundary from the metric alone.     The chain is:

        g_ab    ->   reduced causal margin mu = g(xi,xi)/h(xi)
                ->   infimum over the declared coordinate box
                ->   sign change in the parameter
                ->   bracketed bisection
                ->   comparison with the closed form AFTERWARDS

DATA REQUIRED
    the metric; which coordinate is the compact generator; a timelike
    reference direction (declared only for charts with no timelike coordinate
    direction); a coordinate box; a parameter bracket. The literature answer
    is NOT an input.
"""

from _common import banner, section

import math
import inspect
import sympy as sp

from ctcfa import registry as R
from ctcfa import continuation as C


def main() -> None:
    banner(2, "Non-circular derivation of causal phase boundaries")

    section("Step 1 -- the reduced causal margin, derived from g_ab")
    for mid in ("godel_type", "van_stockum", "spinning_cosmic_string",
                "ori_2007_core", "misner", "ori_2005"):
        m = R.get(mid)
        mu = C.reduced_causal_margin_symbolic(m)
        print(f" {mid:26s} mu = {sp.simplify(mu)}")

    section("Step 2 -- Godel-type: boundary vs coordinate-box radius")
    rows = C.convergence_study(R.get("godel_type"), "m", (0.5, 6.0),
                               {"omega": 1.0}, "r", [5, 10, 15, 20, 25, 30],
                               analytic_value=2.0)
    print(f" {'box radius R':>14} {'m*(R)':>20} {'|m* - 2|':>12}")
    for r in rows:
        print(f" {r['box_radius']:>14.1f} {r['boundary']:>20.15f} {r['error']:>12.3e}")
    print()
    print(" The analytic value m = 2 omega is an ASYMPTOTIC statement: the")
    print(" Godel-type critical radius diverges as m -> 2 omega, so a finite")
    print(" box can only approach it. The convergence table is the evidence;")
    print(" a single number would not be.")

    section("Step 3 -- boundaries with exact closed forms")
    cases = [
        ("van_stockum", "a", (0.05, 3.0), {}, {"r": (2.0, 2.0)}, 0.5, "a r = 1"),
        ("spinning_cosmic_string", "alpha", (0.5, 20.0), {"J": 0.1},
         {"r": (0.1, 0.1)}, 4.0, "alpha r = 4 J"),
        ("ori_2007_core", "r", (0.5, 6.0), {"mu": 1.0}, {}, 2.0, "r = 2 mu"),
        ("pseudo_reissner_nordstrom", "r", (1.0, 3.0), {"mu": 1.0, "q": 0.5},
         {}, 1 + math.sqrt(0.75), "r_+ = mu + sqrt(mu^2 - q^2)"),
        ("pseudo_reissner_nordstrom", "r", (0.125, 1.0), {"mu": 1.0, "q": 0.5},
         {}, 1 - math.sqrt(0.75), "r_- = mu - sqrt(mu^2 - q^2)"),
        ("misner", "T", (-1.5, 1.5), {}, {}, 0.0, "T = 0"),
        ("ori_2005", "T", (-1.5, 1.5), {"x": 0.0, "y": 0.0}, {}, 0.0, "T = F(x,y)"),
    ]
    print(f" {'model':<26}{'par':<7}{'derived':>21}{'closed form':>21}{'error':>11}")
    for mid, par, brk, fx, box, an, form in cases:
        th = C.causal_phase_boundary(R.get(mid), par, brk, fx, box,
                                     analytic_value=an, analytic_form=form)
        print(f" {mid:<26}{par:<7}{th.value:>21.15f}{an:>21.15f}"
              f"{th.analytic_error:>11.2e}")
        print(f"      bracket = [{th.bracket[0]:.15f}, {th.bracket[1]:.15f}] "
              f"width {th.bracket_width:.2e} domain_ok={th.domain_ok}")
        print(f"      closed form: {form}")

    section("Step 4 -- rejecting a normalisation artefact")
    th = C.causal_phase_boundary(R.get("pseudo_reissner_nordstrom"), "r",
                                 (0.05, 0.13), {"mu": 1.0, "q": 0.5}, {})
    print(f" a sign change exists at r = {th.value:.15f}")
    print(f" domain_ok = {th.domain_ok}")
    print(f" {th.note.split('|')[-1].strip()}")
    print()
    print(" The zero of -g(u,u) is where the timelike REFERENCE direction")
    print(" becomes null, not where the orbit changes causal character.")
    print(" Reporting it as a causal boundary would be failure mode FM-001.")

    section("Step 5 -- the non-circularity check that runs in CI")
    src = inspect.getsource(C).split('"""', 2)[-1]
    print(f" '4*omega*omega' present in the solver body : {'4*omega*omega' in src}")
    print(f" 'm**2 - 4' present in the solver body       : {'m**2 - 4' in src}")
    print(" (tests/analytic/test_analytic_benchmarks.py::test_solver_is_non_circular)")


if __name__ == "__main__":
    main()
