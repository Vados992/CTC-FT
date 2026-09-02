"""
EXAMPLE 5 -- Horizon taxonomy: four different objects that are not each other.

WHAT THIS UNIFIES
    "There is a horizon, therefore there may be a time machine" is the single
    most common category error in this subject. v1.0 stated the taxonomy in
    a table and implemented one solver (g^rr = 0) that cannot tell the four
    objects apart. v2.0 computes each separately and attaches to each the
    inference it must never license.

    Reissner-Nordstrom is the decisive control: it HAS an inner Cauchy
    horizon and has NO chronology violation. Misner is the converse: a
    chronology horizon with no curvature singularity anywhere.

DATA REQUIRED
    the metric; which coordinate to solve in; which coordinate direction is
    the candidate Killing generator; for the chronology horizon, a parameter
    slice and a search range.
"""

from _common import banner, section

from ctcfa import registry as R
from ctcfa.horizons import horizon_report


def show(mid, variable="r", killing=0, chron=None, fixed=None, curvature=True):
    m = R.get(mid)
    rep = horizon_report(m, variable, killing, chron, fixed, curvature)
    print(f" {mid} (solving in {variable})")
    for c in rep["candidates"]:
        sols = c["solutions"] or [f"{v:.12g}" for v in c["numeric_roots"]]
        print(f"      {c['kind']:<26} {sols if sols else 'none'}")
        print(f"          forbidden: {c['forbidden_inference']}")
    print()


def main() -> None:
    banner(5, "Horizon taxonomy and the inferences each object forbids")

    section("Reissner-Nordstrom: a Cauchy horizon and NO chronology violation")
    show("reissner_nordstrom")

    section("Schwarzschild: an event horizon, no chronology horizon")
    show("schwarzschild")

    section("de Sitter: a cosmological horizon, no chronology horizon")
    show("de_sitter_static")

    section("Misner: a chronology horizon and NO curvature singularity")
    show("misner", variable="T", killing=1, chron=(-2.0, 2.0), fixed={},
         curvature=True)

    section("Ori 2007 core: chronology horizon at r = 2 mu")
    show("ori_2007_core", variable="r", killing=0, chron=(0.2, 6.0),
         fixed={"mu": 1.0}, curvature=True)

    section("Reading")
    print(" Reissner-Nordstrom produces two Killing horizons and exactly one")
    print(" curvature singularity (r = 0) and NO chronology horizon.")
    print(" Misner produces a chronology horizon at T = 0 with an identically")
    print(" constant determinant and no curvature singularity at all.")
    print(" The two objects are therefore logically independent, and the")
    print(" architecture refuses to convert one into the other.")


if __name__ == "__main__":
    main()
