"""
EXAMPLE 7 -- Theorem hypotheses and the quantum gate.

WHAT THIS UNIFIES
    A theorem is a conditional. v1.0's risk register lists three separate
    ways of quoting a conclusion without its hypotheses (FM-010, FM-011,
    FM-012) and describes the mitigation as a "hypothesis checklist" that was
    never shipped. Here every theorem is an object whose hypotheses are
    CHECKED before any conclusion is licensed, and every one carries the
    inference it must never be used for.

    The quantum layer is the same discipline applied to <T_mu_nu>_ren: a
    metric does not determine it, so the engine refuses to produce a number
    without a full field/state/renormalisation specification.

DATA REQUIRED
    for theorems: the registry record (matter, topology, horizon metadata).
    for the quantum layer: field content, spin, mass, couplings, state,
    state class, boundary conditions, renormalisation prescription, sampling
    function, trajectory, spacetime dimension. All of them.
"""

from _common import banner, section

from ctcfa import registry as R
from ctcfa.qft import (QEI_REGISTRY, QuantumSpecification, assess, evaluate_qei,
                       krw_hypotheses)
from ctcfa.theorems import THEOREMS, apply


def main() -> None:
    banner(7, "Theorem applicability and the quantum admissibility gate")

    section("The registered theorems and their status")
    for k in sorted(THEOREMS):
        th = THEOREMS[k]
        print(f" {k} [{th.status_note}]")
        print(f"      {th.title} -- {th.reference}")
        print(f"      permitted label : {th.permitted_label}")
        print(f"      forbidden use   : {th.forbidden_use}")

    section("Applicability across representative models")
    models = ["minkowski", "schwarzschild", "morris_thorne", "misner",
              "godel", "ori_2005", "alcubierre"]
    keys = ["stable_causality", "topological_censorship",
            "chronology_protection", "kay_radzikowski_wald",
            "morris_thorne_energy", "ori_counterexample"]
    print(f" {'model':<20}" + "".join(f"{k[:14]:>17}" for k in keys))
    short = {"LICENSED": "LICENSED", "HYPOTHESES_UNVERIFIED": "unverified",
             "NOT_APPLICABLE": "n/a"}
    for mid in models:
        m = R.get(mid)
        row = "".join(f"{short[apply(k, m).applicability.value]:>17}" for k in keys)
        print(f" {mid:<20}{row}")

    section("Why 'chronology protection' is never LICENSED as a theorem")
    m = R.get("misner")
    rep = apply("chronology_protection", m)
    print(f" misner -> {rep.applicability.value}")
    for h, v in rep.hypothesis_results.items():
        print(f"      {h:<32} {v}")
    print(f" {rep.note[:200]}")

    section("Kay-Radzikowski-Wald hypotheses, model by model")
    for mid in ("misner", "grant_space", "ads4_periodic", "godel", "schwarzschild"):
        h = krw_hypotheses(R.get(mid))
        print(f" {mid:<20} horizon={h['chronology_horizon_present']!s:<6} "
              f"compactly_generated={h['compactly_generated']!s:<6} "
              f"applicable={h['applicable']}")

    section("The quantum layer refuses to guess")
    a = assess(R.get("misner"), None)
    print(f" no specification supplied -> {a.status.value}")
    print(f"      {a.note[:200]}")

    section("With a complete specification")
    spec = QuantumSpecification(
        field_content="massless minimally coupled scalar", spin="0", mass=0.0,
        couplings={"xi": 0.0}, state="Minkowski-like Hadamard state",
        state_class="Hadamard", boundary_conditions="periodic on the quotient",
        renormalisation="Hadamard point-splitting", sampling_function="Lorentzian",
        trajectory="static inertial", spacetime_dimension=4)
    print(f" specification complete: {spec.complete}")
    a = assess(R.get("misner"), spec)
    print(f" misner -> {a.status.value}")
    print(f"      Hadamard : {a.hadamard}")
    print(f"      <T>_ren : {a.renormalised_stress}")
    print(f"      backreact: {a.backreaction}")

    section("Polarised hypersurfaces feed the quantum layer")
    from ctcfa.quotient_ctc import polarized_hypersurfaces
    g = R.get("grant_space")
    base = {"t": 0.0, "x": 0.8, "y": 0.0, "z": 0.0, "b": 1.0, "alpha": 1.0}
    pol = [r for r in polarized_hypersurfaces(g, base, "x", 0.05, 3.0, 4)
           if r["value"] > 0]
    a = assess(g, spec, pol)
    print(f" grant_space -> {a.status.value}")
    for loc in a.divergence_loci[:4]:
        print(f"      winding {loc['winding']}: {loc['variable']} = "
              f"{loc['value']:.12f}")

    section("Quantum energy inequalities are keyed by their assumptions")
    for k, rule in QEI_REGISTRY.items():
        print(f" {k}")
        print(f"      field={rule.field_content}, dim={rule.dimension}, "
              f"state={rule.state_class}")
        print(f"      sampling={rule.sampling}, trajectory={rule.trajectory}")
        print(f"      {rule.reference}")
    res = evaluate_qei(spec, {"tau": 1.0})
    print(f" matched rule : {res.rule}")
    print(f" bound         : {res.bound:+.9e} (= -3/(32 pi^2 tau^4))")
    bad = QuantumSpecification(field_content="Dirac spinor", state="vacuum",
                               renormalisation="Hadamard", state_class="Hadamard",
                               sampling_function="Gaussian",
                               trajectory="uniformly accelerated")
    res = evaluate_qei(bad, {"tau": 1.0})
    print(f" mismatched specification -> applicable={res.applicable}")
    print(f"      {res.reason[:190]}")


if __name__ == "__main__":
    main()
