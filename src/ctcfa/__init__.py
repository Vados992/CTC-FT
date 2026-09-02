"""
CTC-FA v2.1 -- Closed Timelike Curve Falsification Architecture.

A reproducible computational research standard for causal-structure
classification: CTC birth/destruction boundaries, energy conditions,
stability, constructibility, quantum admissibility and accessibility.

System architect: Vadym Tsinderhoz.

This package is research infrastructure. It is NOT evidence that a
physically realisable time machine exists, and no layer of it may be quoted
as such. Every verdict carries a scope and a claim level, and the reporting
layer refuses to print a statement stronger than the evidence recorded for
it.
"""

from __future__ import annotations

__version__ = "2.1.0"
__architect__ = "Vadym Tsinderhoz"
__document_id__ = "CTC-FA-2026-V2.1-COMPLETE-EN"

from .status import (CLAIM_TEXT, ClaimLevel, ClaimScope, CTCStatus, Gate,
                     GateRecord, GateResult, ModelKind, OrbitResult, OrbitStatus,
                     OverclaimError, Verdict)
from .model import (CompactDirection, DeckTransformation, DomainConstraint,
                    MatterSpec, MetricModel, Provenance, Quotient)

__all__ = [
    "__version__", "__architect__", "__document_id__",
    "CTCStatus", "OrbitStatus", "ModelKind", "ClaimScope", "ClaimLevel",
    "Gate", "GateRecord", "GateResult", "Verdict", "OrbitResult",
    "OverclaimError", "CLAIM_TEXT",
    "MetricModel", "CompactDirection", "DeckTransformation", "Quotient",
    "DomainConstraint", "MatterSpec", "Provenance",
]
