"""
EXAMPLE 1 -- The L3 gate: one metric, two spacetimes, opposite verdicts.

WHAT THIS UNIFIES
    The single most important structural claim of the architecture is that
    global causal structure is not a function of the local metric. This
    example proves it computationally: ``ads4_universal_cover`` and
    ``ads4_periodic`` carry the *identical* metric tensor and receive
    opposite global verdicts, purely because of the deck group.

    The same demonstration is repeated with the simplest possible pair,
    Minkowski and Minkowski with periodic time, so that no one can attribute
    the effect to curvature.

DATA REQUIRED
    coordinate ordering; the metric in the (-,+,+,+) convention; the deck
    group generators with their inverses; a candidate temporal function.
    Nothing else. In particular NO numerical parameters are needed for the
    structural part of the result.
"""

from _common import banner, section

import sympy as sp

from ctcfa import registry as R
from ctcfa.causality import certify_temporal_function, compact_orbit_test
from ctcfa.topology import compare_global_structure, verify_quotient


def main() -> None:
    banner(1, "The L3 gate: identical local metric, opposite causal structure")

    a = R.get("ads4_universal_cover")
    b = R.get("ads4_periodic")

    section("The two registry entries")
    print(f" A : {a.id:24s} {a.title}")
    print(f"      topology : {a.topology}")
    print(f"      deck group: {a.quotient.label}")
    print(f" B : {b.id:24s} {b.title}")
    print(f"      topology : {b.topology}")
    print(f"      deck group: {b.quotient.label}")

    section("The local metric tensors")
    for m in (a, b):
        print(f" {m.id}:")
        for i in range(m.dim):
            print(f"      g_{i}{i} = {sp.simplify(m.metric[i, i])}")

    cmp = compare_global_structure(a, b)
    print()
    print(f" identical_local_metric : {cmp['identical_local_metric']}")
    print(f" demonstrates_L3_gate    : {cmp['demonstrates_L3_gate']}")

    section("Deck-group certificate for the periodic member")
    cert = verify_quotient(b)
    for g in cert.generators:
        print(f" generator {g.name}")
        print(f"      isometry residual        : {g.isometry_residual}")
        print(f"      is an isometry           : {g.is_isometry}")
        print(f"      time-orientation preserving: {g.time_orientation_preserving}")
        print(f"      has a fixed point        : {g.has_fixed_point}")

    section("Causal verdicts at the same coordinate point")
    for m in (a, b):
        r = compact_orbit_test(m, m.default_point)
        print(f" {m.id:24s} {r.status.value:18s} margin = {r.margin:+.6e}")
        print(f"      {r.reason[:120]}")

    section("Why the temporal-function route splits the pair")
    for m in (a, b):
        c = certify_temporal_function(m, tau=m.symbol("t"))
        print(f" {m.id:24s} tau = t -> {c.status.value} [{c.method}]")
        print(f"      {c.note[:150]}")

    section("The same effect with no curvature at all")
    cmp2 = compare_global_structure(R.get("minkowski"),
                                    R.get("minkowski_time_periodic"))
    print(f" minkowski vs minkowski_time_periodic")
    print(f"       identical_local_metric : {cmp2['identical_local_metric']}")
    print(f"       demonstrates_L3_gate   : {cmp2['demonstrates_L3_gate']}")
    for mid in ("minkowski", "minkowski_time_periodic"):
        m = R.get(mid)
        if m.compact:
             r = compact_orbit_test(m, m.default_point)
             print(f"      {mid:26s} {r.status.value} margin {r.margin:+.3e}")
        else:
             c = certify_temporal_function(m)
             print(f"      {mid:26s} {c.status.value} scope {c.scope.value}")

    section("Reading")
    print(" A v1.0-style architecture that stores topology as prose cannot")
    print(" distinguish these pairs at all: the tensors are equal, so every")
    print(" local computation returns the same answer. The distinction is")
    print(" carried entirely by the deck group, which is why v2.0 makes it")
    print(" an executable object rather than a comment.")


if __name__ == "__main__":
    main()
