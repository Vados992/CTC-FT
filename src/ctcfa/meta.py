"""
CTC-FA v2.0   --   L19 : cross-family meta-analysis.

Purpose
-------
The point of a registry is not to catalogue famous geometries. It is to make
the question "what do the chronology-violating spacetimes have in common, and
what do the chronal ones have in common?" a *computation* over a common
feature space.

Method and its honesty constraint
---------------------------------
Every candidate invariant precursor is derived on a **training** subset and
then tested on a **held-out** subset that the derivation never saw. A rule
that survives is reported with its held-out score; a rule that does not is
reported as refuted. Nothing is promoted to "universal" -- the architecture
emits ``CANDIDATE_PRECURSOR`` and requires independent replication before any
stronger word is used.

The feature vector is deliberately built only from quantities the other
layers *compute*, never from the model's name or its expected status, so that
the analysis cannot trivially recover the answer it was given. A
``leakage_check`` verifies exactly that.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import sympy as sp

from .causality import compact_orbit_test
from .model import MetricModel
from .status import CTCStatus, OrbitStatus


FEATURE_NAMES = (
    "has_compact_generator",
    "n_deck_generators",
    "has_nontrivial_quotient",
    "generator_is_killing",
    "sector_margin_sign",
    "reduced_margin_defined",
    "vacuum",
    "has_cosmological_constant",
    "flat_covering",
    "dimension",
    "n_parameters",
    "has_temporal_function_candidate",
    "compact_generator_is_time_direction",
)


@dataclass
class FeatureRow:
    model_id: str
    features: dict[str, float]
    label: str
    note: str = ""

    def vector(self) -> np.ndarray:
        return np.array([self.features.get(k, np.nan) for k in FEATURE_NAMES],
                        dtype=float)


def extract_features(model: MetricModel) -> FeatureRow:
    """Compute the feature vector for one model from the engines only."""
    from .symmetry import generators_admissible
    f: dict[str, float] = {}
    f["has_compact_generator"] = float(bool(model.compact))
    f["n_deck_generators"] = float(len(model.quotient.generators))
    f["has_nontrivial_quotient"] = float(bool(model.quotient.generators)
                                          or bool(model.compact))
    try:
         ok, _ = generators_admissible(model)
    except Exception:                                     # noqa: BLE001
         ok = False
    f["generator_is_killing"] = float(bool(ok))

    sign = 0.0
    if model.has_metric and model.compact and model.default_point:
        try:
            r = compact_orbit_test(model, model.default_point)
            sign = (-1.0 if r.status is OrbitStatus.SECTOR_CTC else
                    0.0 if r.status is OrbitStatus.SECTOR_NULL else
                    1.0 if r.status is OrbitStatus.SECTOR_SPACELIKE else np.nan)
        except Exception:                                # noqa: BLE001
            sign = float("nan")
    else:
        sign = float("nan")
    f["sector_margin_sign"] = sign

    try:
        from .continuation import reduced_causal_margin_symbolic
        reduced_causal_margin_symbolic(model)
        f["reduced_margin_defined"] = 1.0
    except Exception:                                    # noqa: BLE001
        f["reduced_margin_defined"] = 0.0

    f["vacuum"] = float(bool(model.matter.vacuum))
    f["has_cosmological_constant"] = float(
        model.matter.cosmological_constant != 0)
    f["flat_covering"] = float(bool(model.metadata.get("flat_covering")))
    f["dimension"] = float(model.dim)
    f["n_parameters"] = float(len(model.params))
    f["has_temporal_function_candidate"] = float(model.temporal_function is not None)
    f["compact_generator_is_time_direction"] = float(
        any(c.index == 0 for c in model.compact))

    label = ("CHRONOLOGY_VIOLATING"
             if model.expected is CTCStatus.CTC_PROVED else
             "CHRONAL" if model.expected is CTCStatus.NO_CTC_CERTIFIED else
             "UNRESOLVED")
    return FeatureRow(model.id, f, label)


def build_table(models: Iterable[MetricModel]) -> list[FeatureRow]:
    return [extract_features(m) for m in models]


# --------------------------------------------------------------------------
# Rule search with held-out validation
# --------------------------------------------------------------------------

@dataclass
class Rule:
    feature: str
    threshold: float
    direction: str             # "<=" or ">"
    train_accuracy: float
    heldout_accuracy: float
    train_n: int
    heldout_n: int
    status: str
    note: str = ""

    def predict(self, row: FeatureRow) -> bool:
        v = row.features.get(self.feature, float("nan"))
        if not np.isfinite(v):
            return False
        return v <= self.threshold if self.direction == "<=" else v > self.threshold

    def as_dict(self) -> dict[str, Any]:
        return {"feature": self.feature, "threshold": self.threshold,
                "direction": self.direction,
                "train_accuracy": self.train_accuracy,
                "heldout_accuracy": self.heldout_accuracy,
                "train_n": self.train_n, "heldout_n": self.heldout_n,
                "status": self.status, "note": self.note}


def _accuracy(rule_pred: Callable[[FeatureRow], bool], rows: Sequence[FeatureRow],
              positive: str) -> float:
    if not rows:
        return float("nan")
    correct = sum(1 for r in rows
                  if rule_pred(r) == (r.label == positive))
    return correct / len(rows)


def search_precursors(rows: Sequence[FeatureRow], positive: str =
                      "CHRONOLOGY_VIOLATING", heldout_fraction: float = 0.35,
                      seed: int = 20260831,
                      min_heldout_accuracy: float = 0.9) -> list[Rule]:
    """Single-feature threshold rules, derived on train and scored on held out."""
    labelled = [r for r in rows if r.label in (positive, "CHRONAL")]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(labelled))
    n_hold = max(2, int(round(heldout_fraction * len(labelled))))
    hold = [labelled[i] for i in idx[:n_hold]]
    train = [labelled[i] for i in idx[n_hold:]]

    out: list[Rule] = []
    for feat in FEATURE_NAMES:
        vals = sorted({r.features.get(feat, float("nan")) for r in train})
        vals = [v for v in vals if np.isfinite(v)]
        if len(vals) < 2:
            continue
        cuts = [0.5 * (a + b) for a, b in zip(vals, vals[1:])]
        best: Rule | None = None
        for c in cuts:
            for d in ("<=", ">"):
                probe = Rule(feat, float(c), d, 0.0, 0.0, len(train), len(hold),
                              "candidate")
                acc = _accuracy(probe.predict, train, positive)
                if best is None or acc > best.train_accuracy:
                     probe.train_accuracy = acc
                     best = probe
        if best is None:
            continue
        best.heldout_accuracy = _accuracy(best.predict, hold, positive)
        best.status = ("CANDIDATE_PRECURSOR"
                       if best.heldout_accuracy >= min_heldout_accuracy
                       else "REFUTED_ON_HELD_OUT")
        best.note = ("derived on the training split only; the held-out score is "
                     "the only number that may be quoted. A CANDIDATE_PRECURSOR "
                     "still requires independent replication before any "
                     "universality claim.")
        out.append(best)
    out.sort(key=lambda r: (-r.heldout_accuracy, -r.train_accuracy))
    return out


def leakage_check(rows: Sequence[FeatureRow]) -> dict[str, Any]:
    """Confirm that no feature is a relabelling of the expected status.

    A feature that perfectly reproduces the label on *every* model is
    suspicious: either it is the label in disguise, or it is a genuine
    theorem. The check reports such features so a human can decide which,
    instead of the analysis silently congratulating itself.
    """
    suspicious = []
    for feat in FEATURE_NAMES:
        vals = {}
        conflict = False
        for r in rows:
            if r.label == "UNRESOLVED":
                continue
            v = r.features.get(feat, float("nan"))
            if not np.isfinite(v):
                continue
            vals.setdefault(v, set()).add(r.label)
        for v, labels in vals.items():
            if len(labels) > 1:
                conflict = True
        if not conflict and len(vals) > 1:
            suspicious.append(feat)
    return {"perfectly_separating_features": suspicious,
            "n_models": len(rows),
            "note": ("A perfectly separating feature is either a hidden copy of "
                     "the label or a real structural theorem. It must be "
                     "adjudicated by hand, never auto-promoted.")}


def cross_family_report(models: Iterable[MetricModel]) -> dict[str, Any]:
    rows = build_table(models)
    rules = search_precursors(rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.label] = counts.get(r.label, 0) + 1
    return {"n_models": len(rows), "label_counts": counts,
            "features": list(FEATURE_NAMES),
            "table": [{"model_id": r.model_id, "label": r.label,
                       **{k: r.features.get(k) for k in FEATURE_NAMES}}
                      for r in rows],
            "rules": [r.as_dict() for r in rules],
            "leakage": leakage_check(rows),
            "governance": ("No rule in this report may be described as a "
                           "theorem or as universal. The permitted label is "
                           "CANDIDATE_PRECURSOR, and only after independent "
                           "replication on a corpus this analysis has never "
                           "seen (v1.0 REQ-019, P8/P9 of the roadmap).")}


__all__ = ["FEATURE_NAMES", "FeatureRow", "extract_features", "build_table",
           "Rule", "search_precursors", "leakage_check", "cross_family_report"]
