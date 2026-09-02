"""Typed scientific contract for CTC-FA v2.1.

The core rule is that a theory, a candidate solution, a characteristic mode,
and a causal verdict are distinct objects.  Every verdict retains its domain,
mode set, scope, implementation state, and evidence notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

import numpy as np


class ImplementationState(str, Enum):
    VERIFIED = "VERIFIED"
    RECOVERED_REQUIRES_RERUN = "RECOVERED_REQUIRES_RERUN"
    SPECIFIED_NOT_IMPLEMENTED = "SPECIFIED_NOT_IMPLEMENTED"


class ClaimScope(str, Enum):
    POINT = "POINT"
    SECTOR = "SECTOR"
    DECLARED_DOMAIN = "DECLARED_DOMAIN"
    GLOBAL = "GLOBAL"
    THEORY_EXISTENCE = "THEORY_EXISTENCE"


class VerdictStatus(str, Enum):
    INVALID_SPEC = "INVALID_SPEC"
    OFF_SHELL = "OFF_SHELL"
    OUTSIDE_EFT = "OUTSIDE_EFT"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    ILL_POSED = "ILL_POSED"
    CTC_PROVED = "CTC_PROVED"
    MODE_CAUSAL_CERTIFIED = "MODE_CAUSAL_CERTIFIED"
    ALL_CONES_CAUSAL_CERTIFIED = "ALL_CONES_CAUSAL_CERTIFIED"
    UNRESOLVED = "UNRESOLVED"


class ModeKind(str, Enum):
    METRIC = "METRIC"
    SCALAR = "SCALAR"
    VECTOR = "VECTOR"
    TENSOR = "TENSOR"
    MIXED = "MIXED"


@dataclass(frozen=True)
class TheorySpec:
    theory_id: str
    fields: tuple[str, ...]
    coupling_dimensions: Mapping[str, str]
    signature: str = "-+++"
    x_convention: str | None = None
    field_equation_adapter: str | None = None
    principal_symbol_adapter: str | None = None
    implementation: ImplementationState = ImplementationState.SPECIFIED_NOT_IMPLEMENTED
    provenance: tuple[str, ...] = ()

    def validate(self) -> tuple[bool, tuple[str, ...]]:
        errors: list[str] = []
        if not self.theory_id.strip():
            errors.append("empty theory_id")
        if "metric:g" not in self.fields:
            errors.append("metric:g field is required")
        if self.signature not in {"-+++", "+---"}:
            errors.append("unsupported signature")
        if not self.field_equation_adapter:
            errors.append("missing field-equation adapter")
        if not self.principal_symbol_adapter:
            errors.append("missing principal-symbol adapter")
        if not self.provenance:
            errors.append("missing provenance")
        return not errors, tuple(errors)


@dataclass(frozen=True)
class TopologySpec:
    """Only the identifications needed by the verified reference core.

    ``time_period=None`` means time is unwrapped.  ``angular_period`` is the
    physical period of the axial coordinate; it is normally 2*pi.
    """

    time_period: float | None = None
    angular_period: float | None = 2.0 * np.pi

    def validate(self) -> tuple[bool, tuple[str, ...]]:
        errors: list[str] = []
        if self.time_period is not None and self.time_period <= 0:
            errors.append("time_period must be positive")
        if self.angular_period is not None and self.angular_period <= 0:
            errors.append("angular_period must be positive")
        return not errors, tuple(errors)


@dataclass(frozen=True)
class SolutionSpec:
    solution_id: str
    theory_id: str
    domain: str
    topology: TopologySpec = field(default_factory=TopologySpec)
    approximation: Mapping[str, int | float | str] = field(default_factory=dict)
    implementation: ImplementationState = ImplementationState.SPECIFIED_NOT_IMPLEMENTED
    provenance: tuple[str, ...] = ()

    def validate_for(self, theory: TheorySpec) -> tuple[bool, tuple[str, ...]]:
        errors: list[str] = []
        if not self.solution_id.strip():
            errors.append("empty solution_id")
        if self.theory_id != theory.theory_id:
            errors.append("solution/theory identity mismatch")
        if not self.domain.strip():
            errors.append("empty domain")
        ok_topology, top_errors = self.topology.validate()
        if not ok_topology:
            errors.extend(top_errors)
        if not self.provenance:
            errors.append("missing solution provenance")
        return not errors, tuple(errors)


@dataclass(frozen=True)
class FieldEquationReport:
    checked_equations: tuple[str, ...]
    max_abs_residual: float
    tolerance: float
    domain_covered: bool
    validity_passed: bool = True
    notes: tuple[str, ...] = ()

    @property
    def on_shell(self) -> bool:
        return (
            bool(self.checked_equations)
            and self.domain_covered
            and np.isfinite(self.max_abs_residual)
            and 0.0 <= self.max_abs_residual <= self.tolerance
        )


@dataclass(frozen=True)
class CharacteristicMode:
    mode_id: str
    kind: ModeKind
    inverse_metric: np.ndarray | None
    physical: bool = True
    hyperbolic: bool = True
    validity_passed: bool = True
    notes: tuple[str, ...] = ()

    def validate(self, dim: int = 4) -> tuple[bool, tuple[str, ...]]:
        errors: list[str] = []
        if not self.mode_id:
            errors.append("empty mode_id")
        if self.inverse_metric is None:
            errors.append("implicit/non-metric characteristic is not implemented by this reference core")
        else:
            a = np.asarray(self.inverse_metric, dtype=float)
            if a.shape != (dim, dim):
                errors.append(f"inverse metric must have shape {(dim, dim)}")
            elif not np.allclose(a, a.T, atol=1e-12, rtol=0.0):
                errors.append("inverse metric must be symmetric")
            elif not np.all(np.isfinite(a)):
                errors.append("inverse metric contains non-finite values")
        return not errors, tuple(errors)

    def inertia(self, tol: float = 1e-10) -> tuple[int, int, int]:
        if self.inverse_metric is None:
            raise ValueError("no quadratic inverse metric")
        eig = np.linalg.eigvalsh(np.asarray(self.inverse_metric, dtype=float))
        return (
            int(np.sum(eig < -tol)),
            int(np.sum(np.abs(eig) <= tol)),
            int(np.sum(eig > tol)),
        )

    def lorentzian_minus_plus(self, tol: float = 1e-10) -> bool:
        return self.inertia(tol) == (1, 0, 3)


@dataclass(frozen=True)
class ConeBundle:
    solution_id: str
    modes: tuple[CharacteristicMode, ...]
    expected_physical_mode_ids: tuple[str, ...]
    domain_covered: bool
    notes: tuple[str, ...] = ()

    @property
    def physical_modes(self) -> tuple[CharacteristicMode, ...]:
        return tuple(m for m in self.modes if m.physical)

    @property
    def complete(self) -> bool:
        actual = {m.mode_id for m in self.physical_modes}
        expected = set(self.expected_physical_mode_ids)
        return bool(expected) and actual == expected and self.domain_covered

    @property
    def hyperbolic(self) -> bool:
        return self.complete and all(
            m.hyperbolic and m.validity_passed and m.lorentzian_minus_plus()
            for m in self.physical_modes
        )


@dataclass(frozen=True)
class CTCWitness:
    mode_id: str
    closed: bool
    future_directed: bool
    timelike_margin: float
    scope: ClaimScope
    notes: tuple[str, ...] = ()

    @property
    def proved(self) -> bool:
        return self.closed and self.future_directed and self.timelike_margin < 0.0


@dataclass(frozen=True)
class TemporalCertificate:
    gradient_covector: tuple[float, ...]
    single_valued: bool
    mode_margins: Mapping[str, float]
    epsilon: float
    domain_covered: bool

    @property
    def proved(self) -> bool:
        return (
            self.single_valued
            and self.domain_covered
            and bool(self.mode_margins)
            and all(v <= -self.epsilon for v in self.mode_margins.values())
        )


@dataclass(frozen=True)
class Verdict:
    status: VerdictStatus
    scope: ClaimScope
    theory_id: str
    solution_id: str
    mode_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def theory_existence_claim(self) -> "Verdict":
        """Only a positive solution witness can widen to theory existence.

        The widened statement means "this theory admits at least one solution
        sector with a proved CTC".  It never means all solutions are acausal.
        """

        if self.status is not VerdictStatus.CTC_PROVED:
            raise ValueError("negative or unresolved solution verdict cannot become a theory claim")
        return Verdict(
            status=self.status,
            scope=ClaimScope.THEORY_EXISTENCE,
            theory_id=self.theory_id,
            solution_id=self.solution_id,
            mode_ids=self.mode_ids,
            notes=self.notes + ("existential theory claim only",),
        )

