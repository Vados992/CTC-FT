"""Blind benchmark protocol (v1.0 section 15.2), executed.

The point of a blind benchmark is that the solver must not be able to read
the answer. v1.0 described the protocol in six numbered steps and shipped a
regression table whose expected values were literals in the source, which is
the opposite arrangement.

Here the answers are HMAC-sealed with a key the *evaluator* holds. The
solver receives only the sealed digests, computes a boundary from the metric,
and submits it; the evaluator verifies. A solver that has not been given
the key cannot recover the target, and a solver that hard-codes a literal
will fail on the perturbed parameter sets, which are drawn at run time.
"""

import math
import os

import numpy as np
import pytest

from ctcfa import registry as R
from ctcfa.continuation import causal_phase_boundary
from ctcfa.provenance import check_answer, seal_answers

EVALUATOR_KEY = b"CTC-FA-v2.0-blind-benchmark-evaluator-key"
QUANT = 9                      # digits retained when sealing / submitting


def _q(x: float) -> float:
    return round(float(x), QUANT)


def _make_cases(seed: int = 20260831):
    """Parameter sets drawn at run time, so a literal cannot pass."""
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(4):
        r0 = float(rng.uniform(0.8, 3.0))
        cases.append({
            "name": f"van_stockum_r{r0:.6f}",
            "model": "van_stockum", "parameter": "a",
            "bracket": (0.05, 5.0), "fixed": {}, "box": {"r": (r0, r0)},
            "truth": 1.0 / r0,
        })
    for _ in range(4):
        J = float(rng.uniform(0.02, 0.3))
        r0 = float(rng.uniform(0.05, 0.5))
        cases.append({
            "name": f"string_J{J:.6f}_r{r0:.6f}",
            "model": "spinning_cosmic_string", "parameter": "alpha",
            "bracket": (0.1, 60.0), "fixed": {"J": J}, "box": {"r": (r0, r0)},
            "truth": 4.0 * J / r0,
        })
    for _ in range(4):
        mu = float(rng.uniform(0.3, 3.0))
        cases.append({
            "name": f"ori2007_mu{mu:.6f}",
            "model": "ori_2007_core", "parameter": "r",
            "bracket": (0.05, 20.0), "fixed": {"mu": mu}, "box": {},
            "truth": 2.0 * mu,
        })
    for _ in range(4):
        mu = float(rng.uniform(1.0, 3.0))
        q = float(rng.uniform(0.1, 0.8)) * mu
        rminus = mu - math.sqrt(mu * mu - q * q)
        rplus = mu + math.sqrt(mu * mu - q * q)
        cases.append({
            "name": f"pseudoRN_outer_mu{mu:.6f}_q{q:.6f}",
            "model": "pseudo_reissner_nordstrom", "parameter": "r",
            "bracket": (rplus * 0.6, rplus * 2.0),
            "fixed": {"mu": mu, "q": q}, "box": {},
            "truth": rplus,
        })
    return cases


CASES = _make_cases()
SEALED = seal_answers({c["name"]: _q(c["truth"]) for c in CASES}, EVALUATOR_KEY)


def test_sealed_answers_are_not_readable():
    """The sealed file must not leak the value."""
    for c in CASES:
        blob = SEALED[c["name"]]
        assert len(blob) == 64 and all(ch in "0123456789abcdef" for ch in blob)
        assert str(round(c["truth"], 6)) not in blob


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_solver_recovers_sealed_boundary(case):
    """The solver sees only the model, the parameter and the bracket."""
    th = causal_phase_boundary(R.get(case["model"]), case["parameter"],
                               case["bracket"], case["fixed"], case["box"])
    submitted = _q(th.value)
    assert check_answer(case["name"], submitted, SEALED, EVALUATOR_KEY), (
        f"{case['name']}: submitted {submitted}")


def test_a_wrong_submission_is_rejected():
    c = CASES[0]
    assert not check_answer(c["name"], _q(c["truth"] * 1.01), SEALED,
                            EVALUATOR_KEY)


def test_score_report():
    """Aggregate score, in the form the protocol asks for."""
    errs, false_pos, false_neg = [], 0, 0
    for c in CASES:
        th = causal_phase_boundary(R.get(c["model"]), c["parameter"],
                                   c["bracket"], c["fixed"], c["box"])
        errs.append(abs(th.value - c["truth"]))
        if th.left_margin * th.right_margin >= 0:
            false_pos += 1
    assert max(errs) < 1e-9
    assert false_pos == 0
    assert false_neg == 0
