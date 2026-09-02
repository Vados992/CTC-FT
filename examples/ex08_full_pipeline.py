"""
EXAMPLE 8 -- End-to-end pipeline on one model, and the data it needs.

WHAT THIS UNIFIES
    Everything. This is the single-model dossier the architecture produces
    when the whole layer stack is run: registry record, structural gates,
    curvature, the causal detector, the causal boundary, the horizon
    taxonomy, energy conditions, theorem applicability, the quantum gate,
    constructibility, the claim ladder and an immutable run record.

      Read together with the DATA REQUIREMENTS block at the end, this is the
      answer to "what do I actually have to supply for the system to work".
"""

from _common import banner, section

import json
import numpy as np
import sympy as sp

from ctcfa import registry as R
from ctcfa.adm import constructibility_report, evaluate_constraints
from ctcfa.causality import compact_orbit_verdict
from ctcfa.continuation import causal_phase_boundary, reduced_causal_margin_symbolic
from ctcfa.energy import energy_report
from ctcfa.horizons import horizon_report
from ctcfa.provenance import new_run, reproduction_script, source_tree_digest
from ctcfa.qft import QuantumSpecification, assess
from ctcfa.report import analyse
from ctcfa.theorems import applicability_matrix
from ctcfa.validate import validate_model

MODEL = "ori_2007_core"
POINT = {"v": 0.0, "r": 1.0, "theta": 1.0, "phi": 0.0, "mu": 1.0, "P": 1.0}


def main() -> None:
    banner(8, f"Full pipeline on {MODEL}")
    m = R.get(MODEL)

    section("L0 -- registry record")
    rec = m.as_record()
    for k in ("model_id", "title", "kind", "theory", "coordinates",
              "parameters", "signature_convention", "topology"):
        print(f" {k:<24}: {rec[k]}")
    print(f" {'metric components':<24}:")
    for k, v in sorted(rec["metric_components"].items()):
        print(f"      g_{k} = {v}")
    print(f" {'identifications':<24}: {rec['identifications']}")
    print(f" {'matter':<24}: {rec['matter']}")
    print(f" {'domain':<24}: {[c['name'] + ': ' + c['expression'] for c in rec['domain']]}")
    print(f" {'source refs':<24}: {rec['provenance']['source_refs']}")

    section("L0/L1/L2/L3 -- structural and algebraic gates")
    rep = validate_model(m, POINT, cross_check=True)
    for g, r in rep.gates.as_dict()["results"].items():
        note = rep.gates.as_dict()["notes"].get(g, "")
        print(f" {g:<22} {r:<16} {note[:80]}")
    cc = rep.details.get("einstein_cross_check", {})
    if cc:
        print(f" independent curvature routes differ by "
              f"{cc['max_difference']:.3e} (numeric error estimate "
              f"{cc['numeric_error_estimate']:.3e})")

    section("L4 -- symmetry: is the declared generator really Killing?")
    for c in rep.details["compact_generators"]:
        print(f" generator {c['name']}: Killing = {c['is_killing']}, "
              f"residual = {c['residual']}")

    section("L5 -- causal detector")
    v = compact_orbit_verdict(m, POINT)
    print(f" {v.render()}")
    print(f" evidence: {json.dumps(v.evidence, default=str)[:220]}")

    section("L16 -- causal phase boundary, derived from the metric")
    mu_expr = reduced_causal_margin_symbolic(m)
    print(f" reduced causal margin: mu(r) = {sp.simplify(mu_expr)}")
    th = causal_phase_boundary(m, "r", (0.5, 6.0), {"mu": 1.0}, {},
                               analytic_value=2.0, analytic_form="r = 2 mu")
    print(f" boundary r* = {th.value:.15f}    closed form r = 2 mu -> 2.0")
    print(f" bracket [{th.bracket[0]:.15f}, {th.bracket[1]:.15f}], "
          f"width {th.bracket_width:.2e}, error {th.analytic_error:.2e}")

    section("L8 -- horizon taxonomy")
    h = horizon_report(m, "r", 0, (0.2, 6.0), {"mu": 1.0})
    for c in h["candidates"]:
        sols = c["solutions"] or [f"{x:.12g}" for x in c["numeric_roots"]]
        print(f" {c['kind']:<26} {sols}")

    section("L9 -- energy conditions, on shell")
    e = energy_report(m, POINT)
    print(f" Hawking-Ellis type : {e.he.type.value}")
    print(f" rho                 : {e.he.rho:+.9f}")
    print(f" pressures           : {[round(p, 9) for p in e.he.pressures]}")
    print(f" NEC {e.nec_margin:+.6e} [{e.nec_status}]    "
          f"WEC {e.wec_margin:+.6e} [{e.wec_status}]")
    print(f" SEC {e.sec_margin:+.6e} [{e.sec_status}]    "
          f"DEC {e.dec_margin:+.6e} [{e.dec_status}]")
    print(f" cross-check |A-B| : {e.cross_check_error:.3e}")

    section("L12 -- constructibility (constraints evaluated, evolution NOT)")
    cr = constructibility_report(m, POINT)
    print(f" status : {cr['status']}")
    if "error" in cr:
        print(f" reason : {cr['error'][:240]}")
        print(" This is an honest NOT_APPLICABLE, not a silent pass: the")
        print(" constant-v slices of a null-adapted chart are degenerate, so")
        print(" the constraints are undefined on this foliation. The L12")
        print(" demonstration on a spacelike slicing is shown below for")
        print(" Schwarzschild.")
        from ctcfa.adm import decompose, evaluate_constraints as ec
        sm = R.get("schwarzschild")
        d = decompose(sm)
        c2 = ec(sm, sm.default_point)
        print(f" Schwarzschild lapse N = {sp.simplify(d.lapse)}, "
              f"shift = {list(d.shift)}, K = 0 : "
              f"{sp.simplify(d.extrinsic_curvature).is_zero_matrix}")
        print(f" Hamiltonian residual = {c2.hamiltonian_value}, "
              f"momentum = {c2.momentum_values}, satisfied = {c2.satisfied}")
    if "constraints" in cr:
        c = cr["constraints"]
        print(f" Hamiltonian residual : {c['hamiltonian_value']}")
        print(f" momentum residuals    : {c['momentum_values']}")
        print(f" satisfied             : {c['satisfied']}")
    print(f" evolution: {cr['evolution']['status']}")
    print(f"      {cr['evolution']['note'][:190]}")

    section("L15 -- theorem applicability")
    for row in applicability_matrix(m):
        print(f" {row['theorem']:<26} {row['applicability']:<24} "
              f"{row['permitted_label'] or '-'}")

    section("L13/L14 -- quantum gate")
    a = assess(m, None)
    print(f" without a specification -> {a.status.value}")
    print(f" KRW hypotheses            -> applicable = "
          f"{a.theorem_hypotheses['applicable']}")

    section("Headline verdict and permitted claim level")
    dossier = analyse(m, POINT, include_energy=False, include_theorems=False)
    hd = dossier["headline"]
    print(f" {hd['line']}")
    print(f" permitted claim level : {hd['claim_level']} -- {hd['claim_text']}")

    section("Immutable run record and reproduction script")
    digest, files = source_tree_digest(
        __import__("pathlib").Path(R.__file__).resolve().parent)
    rr = new_run(MODEL, "compact_orbit", POINT, v.as_dict(), digest,
                 {"tol": 1e-10}, m.provenance.source_refs)
    print(f" code digest    : {digest}")
    print(f" files hashed : {len(files)}")
    print(f" run content    : {rr.content_hash()}")
    print(f" environment    : python {rr.environment['python']}, "
          f"sympy {rr.environment['modules']['sympy']}, "
          f"numpy {rr.environment['modules']['numpy']}")
    print()
    print(" Minimal independent reproduction script:")
    for line in reproduction_script(MODEL, "compact_orbit", POINT).splitlines():
        print(f"      {line}")

    section("DATA REQUIREMENTS -- what a new model must supply")
    reqs = [
        ("L0 registry", "coordinate ordering; signature convention; the metric "
                        "components; parameter list and domain; topology and "
                        "every identification; matter content; source refs"),
        ("L1 algebra", "nothing further: the inverse, connection and curvature "
                       "are computed"),
        ("L2 field equations", "a closed-form T_ab if the model claims to solve "
                               "them; otherwise the residual only DEFINES the "
                               "required source"),
        ("L3 topology", "deck generators AND their inverses, as explicit "
                        "coordinate maps; a probe point for the "
                        "time-orientation test"),
        ("L4 symmetry", "which coordinate directions are claimed compact, and "
                        "their periods"),
        ("L5 detector", "a fully specified numeric point: every coordinate and "
                        "every parameter"),
        ("L6 loop search", "a centre point, a winding number, and a mode count"),
        ("L7 certificate", "a candidate temporal function; a compact box if the "
                           "proof is to be certified by interval arithmetic"),
        ("L8 horizons", "which coordinate to solve in; which direction is the "
                        "candidate Killing generator"),
        ("L9 energy", "the cosmological constant; the evaluation point"),
        ("L12 initial data", "a slicing; matter density and current if not "
                             "vacuum"),
        ("L13/L14 quantum", "field content, spin, mass, couplings, state, state "
                            "class, boundary conditions, renormalisation "
                            "prescription, sampling function, trajectory, "
                            "dimension"),
        ("L16 continuation", "a parameter, a bracket, values for every other "
                             "symbol, and a coordinate box"),
        ("L17 optimiser", "a SEALED cost metric: scales and weights frozen "
                          "before the run"),
        ("L18 accessibility", "an observer point and a predicate for the "
                              "chronology-violating set"),
        ("L19 meta-analysis", "nothing: features are computed from the other "
                                   "layers, never from the model's name or its "
                                   "expected status"),
    ]
    for k, v_ in reqs:
        print(f" {k:<20} {v_}")


if __name__ == "__main__":
    main()
