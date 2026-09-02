"""
CTC-FA v2.0   --   L15 : theorem and no-go applicability validator.

A theorem is a conditional. Quoting its conclusion without checking its
hypotheses is the most common and most damaging failure in this literature,
and v1.0's own register lists three instances of it (FM-010 chronology
protection cited as a universal theorem, FM-011 KRW generalised outside
compactly generated horizons, FM-012 QEIs applied without their assumptions).
v1.0 described the mitigation as a "hypothesis checklist" but shipped no
checklist.

This module is that checklist, as executable objects.    Each theorem carries

  * its hypotheses, each with a *checker* that returns
    ``True`` / ``False`` / ``None`` (undetermined) for a given model,
  * the conclusion it licenses **and its exact label**,
  * the specific overreach it must never be used for.

``apply(theorem, model)`` returns ``LICENSED`` only when every hypothesis is
verified. A single ``None`` yields ``HYPOTHESES_UNVERIFIED``; a single
``False`` yields ``NOT_APPLICABLE``. There is no path that emits a
conclusion from unverified hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

import sympy as sp

from .model import MetricModel
from .status import ModelKind


class Applicability(str, Enum):
    LICENSED = "LICENSED"
    HYPOTHESES_UNVERIFIED = "HYPOTHESES_UNVERIFIED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass
class Hypothesis:
    name: str
    statement: str
    check: Callable[[MetricModel], bool | None]

    def evaluate(self, model: MetricModel) -> bool | None:
        try:
             return self.check(model)
        except Exception:                                # noqa: BLE001
             return None


@dataclass
class Theorem:
    key: str
    title: str
    reference: str
    hypotheses: list[Hypothesis]
    conclusion: str
    permitted_label: str
    forbidden_use: str
    status_note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "title": self.title, "reference": self.reference,
                "hypotheses": [{"name": h.name, "statement": h.statement}
                               for h in self.hypotheses],
                "conclusion": self.conclusion,
                "permitted_label": self.permitted_label,
                "forbidden_use": self.forbidden_use,
                "status_note": self.status_note}


@dataclass
class TheoremReport:
    theorem: str
    applicability: Applicability
    hypothesis_results: dict[str, bool | None]
    permitted_label: str | None
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {"theorem": self.theorem,
                "applicability": self.applicability.value,
                "hypothesis_results": self.hypothesis_results,
                "permitted_label": self.permitted_label, "note": self.note}


# --------------------------------------------------------------------------
# Hypothesis checkers
# --------------------------------------------------------------------------

def _is_vacuum(m: MetricModel) -> bool | None:
    if m.matter.vacuum:
        return True
    if m.matter.label in ("unspecified", ""):
        return None
    return False


def _asymptotically_flat(m: MetricModel) -> bool | None:
    lam = m.matter.cosmological_constant
    if lam != 0:
        return False
    hint = (m.topology + " " + m.domain_notes).lower()
    if "asymptotically flat" in hint:
        return True
    if m.id in ("minkowski", "schwarzschild", "reissner_nordstrom", "kerr",
                 "kerr_newman", "morris_thorne"):
        return True
    if m.id in ("godel", "godel_type", "godel_critical", "van_stockum",
                 "misner", "misner_covering", "grant_space",
                 "spinning_cosmic_string", "ori_2005", "ori_2007_core",
                 "minkowski_time_periodic", "btz_rotating"):
        return False
    return None


def _globally_hyperbolic(m: MetricModel) -> bool | None:
    if m.temporal_function is not None and not m.quotient.generators \
            and not any(c.index == 0 for c in m.compact):
        return None      # stably causal is necessary but not sufficient
    if m.expected.value == "CTC_PROVED":
        return False
    return None


def _nec_satisfied(m: MetricModel) -> bool | None:
    if m.matter.vacuum:
        return True
    role = m.metadata.get("role", "")
    if role == "energy_condition_counterexample":
        return True
    if m.id in ("morris_thorne", "alcubierre", "krasnikov"):
        return False
    return None


def _has_chronology_horizon(m: MetricModel) -> bool | None:
    return bool(m.metadata.get("chronology_horizon")) or None


def _compactly_generated_horizon(m: MetricModel) -> bool | None:
    if m.id in ("misner", "misner_covering", "grant_space"):
        return True
    if not m.metadata.get("chronology_horizon"):
        return False
    return None


def _four_dimensional(m: MetricModel) -> bool | None:
    return m.dim == 4


def _has_temporal_function(m: MetricModel) -> bool | None:
    return m.temporal_function is not None


def _traversable_wormhole(m: MetricModel) -> bool | None:
    return "wormhole" in (m.title + m.topology).lower()


# --------------------------------------------------------------------------
# Registry of theorems
# --------------------------------------------------------------------------

THEOREMS: dict[str, Theorem] = {
    "stable_causality": Theorem(
        key="stable_causality",
        title="Stable causality from a temporal function",
        reference="Minguzzi & Sanchez, The causal hierarchy of spacetimes "
                  "(arXiv:gr-qc/0609119)",
        hypotheses=[
            Hypothesis("temporal_function",
                       "a smooth function tau exists with everywhere past-directed "
                       "timelike gradient on the FULL declared domain",
                       _has_temporal_function),
        ],
        conclusion="the spacetime is stably causal and therefore contains no "
                   "closed timelike curves on that domain",
        permitted_label="NO_CTC_CERTIFIED (in the certified domain)",
        forbidden_use="a temporal function verified only on a coordinate patch, "
                      "or a function that is not single valued on a quotient, "
                      "licenses nothing",
        status_note="THEOREM"),

    "topological_censorship": Theorem(
        key="topological_censorship",
        title="Topological censorship",
        reference="Friedman, Schleich & Witt, Phys. Rev. Lett. 71, 1486 (1993)",
        hypotheses=[
            Hypothesis("globally_hyperbolic",
                       "the spacetime is globally hyperbolic",
                       _globally_hyperbolic),
            Hypothesis("asymptotically_flat",
                       "the spacetime is asymptotically flat",
                       _asymptotically_flat),
            Hypothesis("null_energy_condition",
                       "the averaged null energy condition holds on the relevant "
                       "causal curves", _nec_satisfied),
        ],
        conclusion="every causal curve between two points of the asymptotic "
                   "region is homotopic to a curve in that region: a traverser "
                   "cannot probe nontrivial topology",
        permitted_label="TOPOLOGY_CONSTRAINED",
        forbidden_use="it is not a chronology-protection theorem and says "
                      "nothing about spacetimes that are not globally hyperbolic, "
                      "which is precisely the class every CTC spacetime is in",
        status_note="THEOREM"),

    "chronology_protection": Theorem(
        key="chronology_protection",
        title="Chronology protection conjecture",
        reference="Hawking, Phys. Rev. D 46, 603 (1992)",
        hypotheses=[
            Hypothesis("chronology_horizon",
                       "a chronology horizon forms", _has_chronology_horizon),
            Hypothesis("compactly_generated",
                       "the chronology horizon is compactly generated",
                       _compactly_generated_horizon),
            Hypothesis("semiclassical_reasoning_valid",
                       "the semiclassical approximation remains valid up to the "
                       "horizon", lambda m: None),
        ],
        conclusion="classical and semiclassical effects are argued to obstruct "
                   "the formation of the chronology horizon",
        permitted_label="CHRONOLOGY_PROTECTION_SUPPORTED_IN_MODEL",
        forbidden_use="this is a CONJECTURE, not a theorem. "
                      "CHRONOLOGY_PROTECTION_PROVED is not an emittable label "
                      "anywhere in this architecture",
        status_note="CONJECTURE"),

    "kay_radzikowski_wald": Theorem(
        key="kay_radzikowski_wald",
        title="QFT on spacetimes with a compactly generated Cauchy horizon",
        reference="Kay, Radzikowski & Wald, Commun. Math. Phys. 183, 533 (1997)",
        hypotheses=[
            Hypothesis("compactly_generated",
                       "the Cauchy horizon is compactly generated",
                       _compactly_generated_horizon),
            Hypothesis("four_dimensional", "spacetime dimension is four",
                       _four_dimensional),
            Hypothesis("free_field",
                       "a free scalar field with the standard Hadamard "
                       "construction", lambda m: None),
        ],
        conclusion="the two-point function cannot be Hadamard on the whole "
                   "horizon; at base points of the horizon generators the "
                   "renormalised stress tensor is ill defined or singular",
        permitted_label="SEMICLASSICAL_QFT_BREAKDOWN (at the base points)",
        forbidden_use="it must not be generalised to horizons that are not "
                      "compactly generated, and it is not a proof that "
                      "backreaction destroys every CTC",
        status_note="THEOREM"),

    "morris_thorne_energy": Theorem(
        key="morris_thorne_energy",
        title="Energy-condition violation at a traversable wormhole throat",
        reference="Morris & Thorne, Am. J. Phys. 56, 395 (1988)",
        hypotheses=[
            Hypothesis("traversable_wormhole",
                       "a static traversable wormhole ansatz in general "
                       "relativity", _traversable_wormhole),
            Hypothesis("flare_out",
                       "the flare-out condition holds at the throat",
                       lambda m: bool(m.metadata.get("flare_out")) or None),
        ],
        conclusion="the null energy condition is violated near the throat",
        permitted_label="ENERGY_CONDITION_VIOLATION (in the applicable throat "
                        "model)",
        forbidden_use="it does not imply that the wormhole is a time machine, "
                      "and it does not apply to quantum-induced traversability",
        status_note="THEOREM"),

    "gao_jafferis_wall": Theorem(
        key="gao_jafferis_wall",
        title="Traversable wormholes via a double-trace deformation",
        reference="Gao, Jafferis & Wall, JHEP 2017:151",
        hypotheses=[
            Hypothesis("ads_btz_setup",
                       "a specific AdS/BTZ holographic construction with the "
                       "stated boundary coupling", lambda m: None),
        ],
        conclusion="a quantum interaction can render an Einstein-Rosen bridge "
                   "traversable without violating causality",
        permitted_label="TRAVERSABLE_IN_MODEL (explicitly NOT a "
                        "causality-violation certificate)",
        forbidden_use="quantum traversability is not chronology violation; this "
                      "result is a counterexample to that inference, not "
                      "support for it",
        status_note="THEOREM"),

    "ori_counterexample": Theorem(
        key="ori_counterexample",
        title="CTC formation with non-exotic matter",
        reference="Ori, Phys. Rev. Lett. 95, 021101 (2005); "
                  "Phys. Rev. D 76, 044002 (2007)",
        hypotheses=[
            Hypothesis("vacuum_core", "a compact vacuum core with a periodic "
                                      "identification", _is_vacuum),
        ],
        conclusion="closed timelike curves can form in a spacetime whose matter "
                   "satisfies the weak, dominant and strong energy conditions",
        permitted_label="ENERGY_CONDITION_COUNTEREXAMPLE",
        forbidden_use="it does not show that such a spacetime is stable, "
                      "semiclassically consistent, or physically constructible "
                      "by any known process",
        status_note="THEOREM (counterexample)"),
}


def apply(theorem_key: str, model: MetricModel) -> TheoremReport:
    """Check every hypothesis before licensing any conclusion."""
    th = THEOREMS[theorem_key]
    results: dict[str, bool | None] = {}
    for h in th.hypotheses:
        results[h.name] = h.evaluate(model)
    if any(v is False for v in results.values()):
        failed = [k for k, v in results.items() if v is False]
        return TheoremReport(theorem_key, Applicability.NOT_APPLICABLE, results,
                             None,
                             f"hypotheses {failed} are NOT satisfied by "
                             f"{model.id}; the theorem licenses nothing here. "
                             f"Forbidden use: {th.forbidden_use}")
    if any(v is None for v in results.values()):
        unknown = [k for k, v in results.items() if v is None]
        return TheoremReport(theorem_key, Applicability.HYPOTHESES_UNVERIFIED,
                             results, None,
                             f"hypotheses {unknown} are UNVERIFIED for "
                             f"{model.id}. The architecture reports the unmet "
                             f"prerequisites instead of applying the theorem's "
                             f"name as a slogan.")
    return TheoremReport(theorem_key, Applicability.LICENSED, results,
                         th.permitted_label,
                         f"all hypotheses verified. Permitted output: "
                         f"{th.permitted_label}. Status: {th.status_note}. "
                         f"Forbidden use: {th.forbidden_use}")


def applicability_matrix(model: MetricModel) -> list[dict[str, Any]]:
    return [apply(k, model).as_dict() for k in sorted(THEOREMS)]


__all__ = ["Applicability", "Hypothesis", "Theorem", "TheoremReport",
           "THEOREMS", "apply", "applicability_matrix"]
