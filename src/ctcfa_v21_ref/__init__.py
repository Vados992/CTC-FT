"""CTC-FA v2.1 verified reference core.

This package implements only the T0-T3 data and gate logic described in the
complete v2.1 standard.  It deliberately does not claim to implement rotating
k-essence or EdGB background solvers.
"""

from .model import (
    ClaimScope,
    ConeBundle,
    FieldEquationReport,
    ImplementationState,
    ModeKind,
    SolutionSpec,
    TheorySpec,
    Verdict,
    VerdictStatus,
)
from .pipeline import classify

__all__ = [
    "ClaimScope",
    "ConeBundle",
    "FieldEquationReport",
    "ImplementationState",
    "ModeKind",
    "SolutionSpec",
    "TheorySpec",
    "Verdict",
    "VerdictStatus",
    "classify",
]

