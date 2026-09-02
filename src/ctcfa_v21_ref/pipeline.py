"""T0-T3 classification pipeline with refusal-first promotion logic."""

from __future__ import annotations

from collections.abc import Iterable

from .cones import common_temporal_certificate
from .model import (
    CTCWitness,
    ClaimScope,
    ConeBundle,
    FieldEquationReport,
    SolutionSpec,
    TheorySpec,
    Verdict,
    VerdictStatus,
)


def classify(
    theory: TheorySpec,
    solution: SolutionSpec,
    field_report: FieldEquationReport,
    cone_bundle: ConeBundle | None,
    witnesses: Iterable[CTCWitness] = (),
    temporal_gradient: tuple[float, float, float, float] | None = None,
) -> Verdict:
    """Classify one solution without making an unearned theory-wide claim."""

    t_ok, t_errors = theory.validate()
    s_ok, s_errors = solution.validate_for(theory)
    if not t_ok or not s_ok:
        return Verdict(
            VerdictStatus.INVALID_SPEC,
            ClaimScope.POINT,
            theory.theory_id,
            solution.solution_id,
            notes=t_errors + s_errors,
        )
    if not field_report.on_shell:
        return Verdict(
            VerdictStatus.OFF_SHELL,
            ClaimScope.DECLARED_DOMAIN,
            theory.theory_id,
            solution.solution_id,
            notes=field_report.notes + ("field-equation gate failed",),
        )
    if not field_report.validity_passed:
        return Verdict(
            VerdictStatus.OUTSIDE_EFT,
            ClaimScope.DECLARED_DOMAIN,
            theory.theory_id,
            solution.solution_id,
            notes=field_report.notes + ("validity gate failed",),
        )
    if cone_bundle is None or not cone_bundle.complete:
        return Verdict(
            VerdictStatus.NOT_IMPLEMENTED,
            ClaimScope.DECLARED_DOMAIN,
            theory.theory_id,
            solution.solution_id,
            notes=("physical characteristic bundle missing or incomplete",),
        )
    if not cone_bundle.hyperbolic:
        return Verdict(
            VerdictStatus.ILL_POSED,
            ClaimScope.DECLARED_DOMAIN,
            theory.theory_id,
            solution.solution_id,
            mode_ids=tuple(m.mode_id for m in cone_bundle.physical_modes),
            notes=("hyperbolicity/common signature gate failed",),
        )
    proved = tuple(w for w in witnesses if w.proved)
    if proved:
        return Verdict(
            VerdictStatus.CTC_PROVED,
            min((w.scope for w in proved), key=lambda s: list(ClaimScope).index(s)),
            theory.theory_id,
            solution.solution_id,
            mode_ids=tuple(sorted({w.mode_id for w in proved})),
            notes=tuple(note for w in proved for note in w.notes),
        )
    if temporal_gradient is not None:
        cert = common_temporal_certificate(
            cone_bundle,
            temporal_gradient,
            solution.topology,
        )
        if cert.proved:
            return Verdict(
                VerdictStatus.ALL_CONES_CAUSAL_CERTIFIED,
                ClaimScope.DECLARED_DOMAIN,
                theory.theory_id,
                solution.solution_id,
                mode_ids=tuple(sorted(cert.mode_margins)),
                notes=tuple(f"{k}: {v:.12g}" for k, v in sorted(cert.mode_margins.items())),
            )
    return Verdict(
        VerdictStatus.UNRESOLVED,
        ClaimScope.DECLARED_DOMAIN,
        theory.theory_id,
        solution.solution_id,
        mode_ids=tuple(m.mode_id for m in cone_bundle.physical_modes),
        notes=("no positive witness and no common temporal certificate",),
    )


__all__ = ["classify"]

