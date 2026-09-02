"""
CTC-FA v2.0   --   L-1 : Status algebra, claim scope and the claim ladder.

This module contains no physics. It contains the *evidence logic* that every
other layer is required to route its conclusions through.

Design rules enforced here
--------------------------
R1 Asymmetric evidence. One verified witness proves existence. Failure to
    find a witness never proves absence. ``UNRESOLVED`` is therefore an
    absorbing state for every search-type engine.
R2 Scope is a first-class part of a status. A result obtained in a compact
    sector, in a reduced model, or on a restricted domain can never be
    printed as a global result. ``ClaimScope`` is carried with the status and
    ``Verdict.promote`` is the *only* way scope can widen.
R3 A model may never be reported above its weakest satisfied gate. The
    ``ClaimLadder`` computes the permitted level from the gate record and
    ``Verdict.render`` refuses to emit anything higher.

v1.0 defect closed by this module
---------------------------------
D-05 v1.0 stated the scope rules in prose (sections 2, 9.2, 14, Appendix D)
      but the kernel had no representation of scope: ``global_model_test``
      returned bare ``CTC_PROVED`` for the *reduced* time-shifted wormhole
      predicate, which is exactly the category error section 9.2 warns about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


# --------------------------------------------------------------------------
# Global causal status
# --------------------------------------------------------------------------

class CTCStatus(str, Enum):
    """Global causal-structure verdict for a declared model + domain."""

    CTC_PROVED = "CTC_PROVED"
    """An explicit future-directed closed timelike curve, or a mathematically
    complete equivalent criterion, has been verified."""

    CRITICAL = "CRITICAL"
    """A closed-null / marginal causal boundary has been established."""

    NO_CTC_CERTIFIED = "NO_CTC_CERTIFIED"
    """A complete causal certificate (temporal function, or a theorem whose
    hypotheses are *all* verified) excludes CTCs on the full declared
    domain."""

    UNRESOLVED = "UNRESOLVED"
    """Neither existence nor absence has been proved.   No binary claim."""


class OrbitStatus(str, Enum):
    """Causal character of a *single tested compact orbit sector*.

    This is deliberately a different type from :class:`CTCStatus`. v1.0
    failure mode FM-002 is precisely the silent promotion of
    ``SECTOR_SPACELIKE`` to ``NO_CTC_CERTIFIED``; making the two enums
    non-interchangeable makes that promotion a type error.
    """

    SECTOR_SPACELIKE = "SECTOR_SPACELIKE"
    SECTOR_NULL = "SECTOR_NULL"
    SECTOR_CTC = "SECTOR_CTC"
    SECTOR_UNRESOLVED = "SECTOR_UNRESOLVED"


class ModelKind(str, Enum):
    """How the model is represented computationally."""

    LOCAL_METRIC = "LOCAL_METRIC"
    """One canonical local 4x4 (or nxn) metric in one chart."""

    QUOTIENT_GLOBAL = "QUOTIENT_GLOBAL"
    """Local metric plus a non-trivial deck group; global causal structure is
    a property of the quotient, not of the local tensor."""

    COMPOSITE_GLOBAL = "COMPOSITE_GLOBAL"
    """Several regions glued, or a construction defined by a global route
    predicate. No single canonical local tensor is asserted."""

    EXTERNAL_EXACT_SPEC = "EXTERNAL_EXACT_SPEC"
    """The published construction is global or source-specific. CTC-FA
    refuses to fabricate a local tensor for it (v1.0 REQ-016)."""


class ClaimScope(str, Enum):
    """Where a verdict is valid.   Ordered from weakest to strongest."""

    SECTOR = "SECTOR"
    """Only in the scanned compact-generator sector."""

    REDUCED_MODEL = "REDUCED_MODEL"
    """Only under an architecture-specific reduced predicate (for example the
    time-shift route-closure model of section 9.2)."""

    DECLARED_DOMAIN = "DECLARED_DOMAIN"
    """On the coordinate/parameter domain declared in the registry record."""

    GLOBAL = "GLOBAL"
    """On the maximal declared spacetime, including its global
    identifications."""


_SCOPE_ORDER = {
    ClaimScope.SECTOR: 0,
    ClaimScope.REDUCED_MODEL: 1,
    ClaimScope.DECLARED_DOMAIN: 2,
    ClaimScope.GLOBAL: 3,
}


def scope_rank(s: ClaimScope) -> int:
    return _SCOPE_ORDER[s]


def weaker_scope(a: ClaimScope, b: ClaimScope) -> ClaimScope:
    """Return the weaker of two scopes. Composition never strengthens."""
    return a if scope_rank(a) <= scope_rank(b) else b


# --------------------------------------------------------------------------
# Verification gates and the claim ladder
# --------------------------------------------------------------------------

class Gate(str, Enum):
    """The verification gates of section 15.1 (V0-V9), as machine objects."""

    V0_PARSE = "V0_PARSE"
    V1_ALGEBRA = "V1_ALGEBRA"
    V2_TENSOR = "V2_TENSOR"
    V3_FIELD_EQUATION = "V3_FIELD_EQUATION"
    V4_CTC_WITNESS = "V4_CTC_WITNESS"
    V5_BOUNDARY = "V5_BOUNDARY"
    V6_NEGATIVE_CERTIFICATE = "V6_NEGATIVE_CERTIFICATE"
    V7_DYNAMICS = "V7_DYNAMICS"
    V8_QUANTUM = "V8_QUANTUM"
    V9_REPLICATION = "V9_REPLICATION"


class GateResult(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INCONCLUSIVE = "INCONCLUSIVE"


class ClaimLevel(int, Enum):
    """Appendix D claim ladder, now machine-enforced."""

    L0_PARSED = 0
    L1_FIELD_EQUATIONS = 1
    L2_CTC_VERIFIED = 2
    L3_BOUNDARY_CHARACTERISED = 3
    L4_REGULARITY = 4
    L5_ENERGY = 5
    L6_CONSTRUCTIBILITY = 6
    L7_STABILITY = 7
    L8_SEMICLASSICAL = 8
    L9_QUANTUM_GRAVITY = 9


CLAIM_TEXT: Mapping[ClaimLevel, str] = {
    ClaimLevel.L0_PARSED: "Metric/model parsed and internally consistent.",
    ClaimLevel.L1_FIELD_EQUATIONS: "Field equations / tensor identities verified in declared scope.",
    ClaimLevel.L2_CTC_VERIFIED: "CTC mathematically verified.",
    ClaimLevel.L3_BOUNDARY_CHARACTERISED: "CTC region and causal boundary characterised.",
    ClaimLevel.L4_REGULARITY: "Regularity / geodesic-completeness status established.",
    ClaimLevel.L5_ENERGY: "Energy requirements established.",
    ClaimLevel.L6_CONSTRUCTIBILITY: "Formation from regular admissible initial data established.",
    ClaimLevel.L7_STABILITY: "Perturbative/nonlinear stability established.",
    ClaimLevel.L8_SEMICLASSICAL: "Semiclassical consistency established.",
    ClaimLevel.L9_QUANTUM_GRAVITY: "Quantum-gravity consistency established.",
}

#: Which gate must PASS before a claim level may be asserted.
LADDER_REQUIREMENT: Mapping[ClaimLevel, tuple[Gate, ...]] = {
    ClaimLevel.L0_PARSED: (Gate.V0_PARSE,),
    ClaimLevel.L1_FIELD_EQUATIONS: (Gate.V0_PARSE, Gate.V1_ALGEBRA, Gate.V2_TENSOR,
                                    Gate.V3_FIELD_EQUATION),
    ClaimLevel.L2_CTC_VERIFIED: (Gate.V0_PARSE, Gate.V1_ALGEBRA, Gate.V4_CTC_WITNESS),
    ClaimLevel.L3_BOUNDARY_CHARACTERISED: (Gate.V0_PARSE, Gate.V1_ALGEBRA, Gate.V5_BOUNDARY),
    ClaimLevel.L4_REGULARITY: (Gate.V0_PARSE, Gate.V1_ALGEBRA, Gate.V2_TENSOR),
    ClaimLevel.L5_ENERGY: (Gate.V0_PARSE, Gate.V1_ALGEBRA, Gate.V2_TENSOR,
                           Gate.V3_FIELD_EQUATION),
    ClaimLevel.L6_CONSTRUCTIBILITY: (Gate.V0_PARSE, Gate.V3_FIELD_EQUATION, Gate.V7_DYNAMICS),
    ClaimLevel.L7_STABILITY: (Gate.V0_PARSE, Gate.V3_FIELD_EQUATION, Gate.V7_DYNAMICS),
    ClaimLevel.L8_SEMICLASSICAL: (Gate.V0_PARSE, Gate.V8_QUANTUM),
    ClaimLevel.L9_QUANTUM_GRAVITY: (), # unreachable by construction, see below
}


@dataclass
class GateRecord:
    """The set of gate outcomes accumulated for one model + domain."""

    results: dict[Gate, GateResult] = field(default_factory=dict)
    notes: dict[Gate, str] = field(default_factory=dict)

    def set(self, gate: Gate, result: GateResult, note: str = "") -> "GateRecord":
        self.results[gate] = result
        if note:
            self.notes[gate] = note
        return self

    def get(self, gate: Gate) -> GateResult:
        return self.results.get(gate, GateResult.NOT_ATTEMPTED)

    def passed(self, gate: Gate) -> bool:
        return self.get(gate) is GateResult.PASS

    def highest_permitted_level(self) -> ClaimLevel:
        """Return the highest claim level whose *entire* requirement set passed.

        L9 is unreachable: no accepted complete theory of quantum gravity
        exists, so the architecture may never emit it. This is enforced, not
        merely documented.
        """
        best = ClaimLevel.L0_PARSED if self.passed(Gate.V0_PARSE) else ClaimLevel.L0_PARSED
        reached = []
        for level, req in LADDER_REQUIREMENT.items():
            if level is ClaimLevel.L9_QUANTUM_GRAVITY:
                continue
            if req and all(self.passed(g) for g in req):
                reached.append(level)
        if reached:
            best = max(reached, key=lambda lv: lv.value)
        elif not self.passed(Gate.V0_PARSE):
            best = ClaimLevel.L0_PARSED
        return best

    def as_dict(self) -> dict[str, Any]:
        return {
            "results": {g.value: r.value for g, r in sorted(
                 self.results.items(), key=lambda kv: kv[0].value)},
            "notes": {g.value: n for g, n in sorted(
                 self.notes.items(), key=lambda kv: kv[0].value)},
            "highest_permitted_level": self.highest_permitted_level().name,
        }


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------

class OverclaimError(RuntimeError):
    """Raised when code attempts to state a result above its evidence."""


@dataclass
class Verdict:
    """A causal verdict together with everything needed to bound it.

    ``Verdict`` is the only object the reporting layer accepts.   A bare
    :class:`CTCStatus` cannot be rendered.
    """

    status: CTCStatus
    scope: ClaimScope
    margin: float
    method: str
    reason: str
    model_id: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)
    gates: GateRecord = field(default_factory=GateRecord)
    evidence: Mapping[str, Any] = field(default_factory=dict)
    uncertainty: float = 0.0

    # -- scope algebra ----------------------------------------------------

    def restrict(self, scope: ClaimScope) -> "Verdict":
        """Narrow the scope. Always allowed."""
        return Verdict(self.status, weaker_scope(self.scope, scope), self.margin,
                       self.method, self.reason, self.model_id, self.parameters,
                       self.gates, self.evidence, self.uncertainty)

    def promote(self, scope: ClaimScope, justification: str) -> "Verdict":
        """Widen the scope. Requires an explicit justification string.

        Promotion of a *negative* result (``NO_CTC_CERTIFIED``) to
        ``GLOBAL`` requires a global certificate; promotion of a
        ``SECTOR``-scoped positive result to ``GLOBAL`` is permitted because
        an explicit witness in a sector of the declared spacetime *is* a
        global existence proof for that spacetime (asymmetric evidence, R1).
        """
        if scope_rank(scope) <= scope_rank(self.scope):
            return self
        if self.status is CTCStatus.NO_CTC_CERTIFIED and scope is ClaimScope.GLOBAL:
            if "certificate" not in justification.lower() and "theorem" not in justification.lower():
                raise OverclaimError(
                    "Promotion of NO_CTC_CERTIFIED to GLOBAL requires a named "
                    "temporal-function certificate or a fully verified theorem; "
                    f"got justification={justification!r}")
        if self.status is CTCStatus.UNRESOLVED:
            raise OverclaimError("UNRESOLVED can never be promoted; it is absorbing.")
        ev = dict(self.evidence)
        ev["scope_promotion"] = justification
        return Verdict(self.status, scope, self.margin, self.method, self.reason,
                        self.model_id, self.parameters, self.gates, ev, self.uncertainty)

    # -- rendering --------------------------------------------------------

    def permitted_level(self) -> ClaimLevel:
        return self.gates.highest_permitted_level()

    def render(self, requested_level: ClaimLevel | None = None) -> str:
        """Render a one-line statement, refusing to exceed the evidence."""
        allowed = self.permitted_level()
        lvl = requested_level if requested_level is not None else allowed
        if lvl.value > allowed.value:
            raise OverclaimError(
                f"{self.model_id}: requested claim level {lvl.name} exceeds the "
                f"highest level supported by the gate record ({allowed.name}).")
        return (f"{self.model_id or '<model>'}: {self.status.value} "
                f"[scope={self.scope.value}, level={lvl.name}, "
                f"margin={self.margin:+.6e}, u={self.uncertainty:.2e}] "
                f"via {self.method} -- {self.reason}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "status": self.status.value,
            "scope": self.scope.value,
            "margin": self.margin,
            "uncertainty": self.uncertainty,
            "method": self.method,
            "reason": self.reason,
            "parameters": dict(self.parameters),
            "gates": self.gates.as_dict(),
            "evidence": dict(self.evidence),
            "permitted_claim_level": self.permitted_level().name,
        }


# --------------------------------------------------------------------------
# Sector result
# --------------------------------------------------------------------------

@dataclass
class OrbitResult:
    """Result of the compact-orbit detector for one sector."""

    status: OrbitStatus
    margin: float
    winding: tuple[int, ...] | None
    point: Mapping[str, float]
    reason: str
    uncertainty: float = 0.0
    generator_names: tuple[str, ...] = ()
    orientation_consistent: bool | None = None
    isometry_verified: bool | None = None

    def to_verdict(self, model_id: str, gates: GateRecord | None = None) -> Verdict:
        """Lift a sector result to a *sector-scoped* verdict.

        This is the only bridge from :class:`OrbitStatus` to
        :class:`CTCStatus`, and it is deliberately lossy in one direction:
        ``SECTOR_SPACELIKE`` maps to ``UNRESOLVED``, never to
        ``NO_CTC_CERTIFIED`` (v1.0 REQ-005 / FM-002).
        """
        gates = gates or GateRecord()
        if self.status is OrbitStatus.SECTOR_CTC:
            st, scope = CTCStatus.CTC_PROVED, ClaimScope.SECTOR
        elif self.status is OrbitStatus.SECTOR_NULL:
            st, scope = CTCStatus.CRITICAL, ClaimScope.SECTOR
        else:
            st, scope = CTCStatus.UNRESOLVED, ClaimScope.SECTOR
        return Verdict(status=st, scope=scope, margin=self.margin,
                       method="compact_orbit", reason=self.reason,
                       model_id=model_id, parameters=dict(self.point),
                       gates=gates, uncertainty=self.uncertainty,
                       evidence={"winding": self.winding,
                                 "generators": list(self.generator_names),
                                 "orientation_consistent": self.orientation_consistent,
                                 "isometry_verified": self.isometry_verified,
                                 "orbit_status": self.status.value})


__all__ = [
    "CTCStatus", "OrbitStatus", "ModelKind", "ClaimScope", "Gate", "GateResult",
    "GateRecord", "ClaimLevel", "CLAIM_TEXT", "LADDER_REQUIREMENT", "Verdict",
    "OrbitResult", "OverclaimError", "scope_rank", "weaker_scope",
]
