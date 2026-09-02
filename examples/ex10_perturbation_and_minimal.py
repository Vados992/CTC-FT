"""
EXAMPLE 10 -- On-shell perturbation, minimal CTC birth, and initial data.

WHAT THIS UNIFIES
    Three gates that v1.0 stated and did not enforce:

      L11   "Off-shell perturbations cannot establish physical robustness"
            -- v1.0's perturb_metric added an arbitrary tensor and nothing
            stopped the result being quoted as stability evidence.
      L17   "The cost metric and normalization scales are preregistered;
            otherwise the word minimal is meaningless"
            -- v1.0 had no cost object at all.
      L12   "Constructibility from regular initial data is stronger than
            existence of an eternal metric"
            -- v1.0 had constraint expressions and no solver.

DATA REQUIRED
    for L11: a base point, a parameter to move, and a step; or an explicit
    symmetric tensor whose on-shell status will be MEASURED.
    for L17: a SEALED cost metric -- scales and weights frozen before the run.
    for L12: a slicing, and matter data if not vacuum.
"""

from _common import banner, section

import numpy as np
import sympy as sp

import scipy.optimize
from ctcfa import registry as R
from ctcfa.adm import decompose, evaluate_constraints, york_hamiltonian_1d
from ctcfa.optimize import CostMetric, UnsealedCostError, minimal_birth
from ctcfa.perturbation import (OnShellStatus, parameter_family_perturbation,
                                robustness_scan, tensor_perturbation)


def main() -> None:
    banner(10, "On-shell perturbation, minimal birth, and regular initial data")

    section("L11a -- ON-SHELL: move along the exact family")
    m = R.get("godel_type")
    base = dict(m.default_point)
    rows = robustness_scan(m, "m", base, [-0.4, -0.2, -0.05, 0.05, 0.2, 0.4,
                                          0.8, 1.0])
    print(f" base: omega = {base['omega']}, m = {base['m']}, r = {base['r']}")
    print(f" {'delta m':>10}{'margin':>18}{'status':>20}{'changed':>10}")
    for r in rows:
        print(f" {r['delta']:>10.2f}{r['margin']:>18.6e}"
              f"{r['status'].replace('SECTOR_',''):>20}{str(r['changed']):>10}")
    print(" Every row here is an exact solution of the same family, so the")
    print(" sign changes are physical, not artefacts of an arbitrary tensor.")

    section("L11b -- OFF-SHELL: the same perturbation, measured and rejected")
    vs = R.get("van_stockum")
    n = vs.dim
    h = sp.zeros(n, n)
    h[2, 2] = sp.Symbol("r", real=True) ** 2           # bump g_phi_phi
    res = tensor_perturbation(vs, h, 0.3, vs.default_point)
    print(f" perturbation kind : {res.kind.value}")
    print(f" exploratory        : {res.exploratory}")
    print(f" margin {res.base_margin:+.4e} -> {res.perturbed_margin:+.4e} "
          f"(status {res.status_before} -> {res.status_after})")
    print(f" {res.note[:230]}")

    section("L17 -- a cost metric must be SEALED before it can be used")
    cost = CostMetric(scales={"m": 1.0, "omega": 1.0},
                       weights={"m": 1.0, "omega": 1.0},
                       label="Godel-type family, unit scales")
    try:
         cost.evaluate({"m": 0.1})
         print(" ERROR: unsealed cost evaluated")
    except UnsealedCostError as exc:
         print(f" refused: {str(exc)[:150]}")
    cost.seal()
    print(f" sealed. digest = {cost.digest}")
    print(f" J(delta m = 0.1) = {cost.evaluate({'m': 0.1}):.6f}")

    section("L17 -- minimal on-shell perturbation that CREATES a CTC")
    chronal = dict(m.default_point)
    chronal["m"] = 3.0                                  # m > 2 omega: chronal
    from ctcfa.causality import compact_orbit_test
    r0 = compact_orbit_test(m, chronal)
    print(f" starting point m = 3.0, omega = 1.0 -> {r0.status.value} "
          f"(margin {r0.margin:+.4e})")
    mb = minimal_birth(m, chronal, ["m"], cost, {"m": (-2.5, 2.5)},
                       restarts=6)
    print(f" achieved    : {mb.achieved}")
    print(f" delta       : { {k: round(v, 9) for k, v in mb.delta.items()} }")
    print(f" cost J      : {mb.cost:.9f}")
    m_final = chronal["m"] + mb.delta["m"]
    print(f" m_final     : {m_final:.9f}")
    print(f" cost digest: {mb.cost_metric_digest[:32]}...")
    print(f" {mb.note[:180]}")
    print()
    print(" This is NOT 2.0, and that is the correct answer. The FAMILY")
    print(" boundary m = 2 omega is the r -> infinity limit; at the fixed")
    print(" radius r = 2 the reduced margin")
    print("      mu = 1 - (4 omega^2/m^2) tanh^2(m r / 2)")
    print(" vanishes where tanh(m) = m/2, whose root is:")
    import scipy.optimize as _so
    root = _so.brentq(lambda mm: np.tanh(mm) - mm / 2, 1.0, 1.99, xtol=1e-14)
    print(f"      m* (r = 2) = {root:.12f}")
    print(f" engine        = {m_final:.12f}   |difference| = "
          f"{abs(m_final - root):.2e}")
    print(" Asking for the cheapest change that creates a CTC AT A POINT is a")
    print(" different question from asking where the family stops being")
    print(" causal, and the architecture keeps them apart.")

    section("L12 -- ADM decomposition and the constraints")
    for mid in ("schwarzschild", "flrw_exponential", "minkowski"):
        sm = R.get(mid)
        d = decompose(sm)
        c = evaluate_constraints(sm, sm.default_point)
        print(f" {mid}")
        print(f"      lapse N           = {sp.simplify(d.lapse)}")
        print(f"      shift N^i         = {list(d.shift)}")
        print(f"      K_ij zero         = "
              f"{sp.simplify(d.extrinsic_curvature).is_zero_matrix}")
        print(f"      Hamiltonian res. = {c.hamiltonian_value}")
        print(f"      momentum res.     = {c.momentum_values}")
        print(f"      satisfied         = {c.satisfied} (with rho = 0)")
        if c.hamiltonian_value and abs(c.hamiltonian_value) > 1e-9:
            rho_req = c.hamiltonian_value / (16 * np.pi)
            print(f"      => required rho = {rho_req:.12f}; for H = 1 the")
            print(f"          Friedmann value 3H^2/(8 pi) = "
                  f"{3.0/(8*np.pi):.12f}. The residual is not an error: with")
            print(f"          no matter declared it DEFINES the source, which is")
            print(f"          exactly the Friedmann equation.")

    section("L12 -- CONSTRUCTING regular initial data (conformal York solve)")
    r = np.linspace(1e-3, 20.0, 400)
    for amp in (0.01, 0.05, 0.2):
        rho = amp * np.exp(-r ** 2 / 2)
        y = york_hamiltonian_1d(rho, r)
        adm_mass = 2.0 * (float(y.psi[-1]) - 1.0) * float(r[-1])
        print(f" rho_hat amplitude {amp:<6} -> converged={y.converged} "
              f"residual={y.residual:.2e} iters={y.iterations} "
              f"psi(0)={float(y.psi[0]):.9f}")
    print()
    print(" The conformal factor solves")
    print("      Laplacian(psi) + 2 pi rho_hat psi^(-3) = 0")
    print(" with a regular centre and psi -> 1 at infinity. This is the step")
    print(" that turns 'assume regular initial data' into 'here is some'.")

    section("What is still missing, stated plainly")
    print(" A convergent 3+1 EVOLUTION of that data is not implemented, so no")
    print(" model in this registry may be reported above claim level L5.")
    print(" The correct path is integration with the Einstein Toolkit, and")
    print(" until it exists the L12 evolution gate stays NOT_ATTEMPTED.")


if __name__ == "__main__":
    main()
