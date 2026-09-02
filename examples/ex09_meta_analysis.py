"""
EXAMPLE 9 -- Cross-family meta-analysis with held-out validation.

WHAT THIS UNIFIES
    This is what a registry is FOR. Every model's feature vector is built
    only from quantities the other layers compute -- never from the model's
    name and never from its expected status -- and candidate invariant
    precursors are derived on a training split, then scored on a held-out
    split the derivation never saw.

      v1.0 described this as ALG-META-001 and shipped nothing. The governance
      rule is unchanged and now enforced: the strongest emittable label is
      CANDIDATE_PRECURSOR.

DATA REQUIRED
    nothing beyond the registry; the features come from L3, L4, L5 and L9.
"""

from _common import banner, section

from ctcfa import registry as R
from ctcfa.meta import FEATURE_NAMES, cross_family_report


def main() -> None:
    banner(9, "Cross-family meta-analysis")
    rep = cross_family_report(R.models())

    section("Corpus")
    print(f" models : {rep['n_models']}")
    for k, v in sorted(rep["label_counts"].items()):
        print(f"      {k:<24} {v}")

    section("Feature table (computed, never copied from the label)")
    keys = ["has_compact_generator", "n_deck_generators",
            "generator_is_killing", "sector_margin_sign", "vacuum",
            "flat_covering", "compact_generator_is_time_direction"]
    hdr = "".join(f"{k[:11]:>13}" for k in keys)
    print(f" {'model':<26}{'label':<22}{hdr}")
    for row in rep["table"]:
        vals = "".join(
            f"{('-' if row[k] is None else f'{row[k]:g}'):>13}" for k in keys)
        print(f" {row['model_id']:<26}{row['label'][:21]:<22}{vals}")

    section("Candidate precursors, scored on the held-out split")
    print(f" {'feature':<38}{'rule':<10}{'train':>8}{'held-out':>10} status")
    for r in rep["rules"][:10]:
        rule = f"{r['direction']} {r['threshold']:g}"
        print(f" {r['feature']:<38}{rule:<10}{r['train_accuracy']:>8.3f}"
              f"{r['heldout_accuracy']:>10.3f} {r['status']}")

    section("Leakage check")
    lk = rep["leakage"]
    print(f" perfectly separating features: {lk['perfectly_separating_features']}")
    print(f" {lk['note']}")

    section("Governance")
    print(f" {rep['governance']}")

    section("Reading -- including what is WRONG with the top rule")
    print(" The top-scoring rule on this corpus is")
    print("      has_temporal_function_candidate <= 0.5 -> chronology violating")
    print(" with a perfect held-out score. It is nonetheless SCIENTIFICALLY")
    print(" WORTHLESS, and saying so is the point of running the analysis.")
    print(" A candidate temporal function is something a HUMAN registers when")
    print(" they already believe the spacetime is chronal; the feature is")
    print(" therefore a curation artefact, not a property of the geometry.")
    print(" The leakage check does not catch it because BTZ is chronal and")
    print(" carries no registered candidate, so the separation is 94.7% on")
    print(" the training split rather than exact.")
    print()
    print(" The same run also shows sector_margin_sign separating perfectly,")
    print(" which the leakage check DOES flag: that feature is the detector's")
    print(" own output, so agreement with the label is a tautology.")
    print()
    print(" What survives as a genuine, non-tautological observation is that")
    print(" every chronology-violating entry in this registry carries a")
    print(" nontrivial identification structure -- a compact Killing generator")
    print(" or a deck group -- and no chronal entry with a global temporal")
    print(" function does. That is the L3 gate stated as a corpus fact. It")
    print(" is emitted as CANDIDATE_PRECURSOR and nothing stronger, and it")
    print(" would need an independent corpus, not this one, to become a")
    print(" claim.")
    print()
    print(" Reporting a top rule that the architecture then disqualifies is")
    print(" the intended behaviour of a falsification framework.")


if __name__ == "__main__":
    main()
