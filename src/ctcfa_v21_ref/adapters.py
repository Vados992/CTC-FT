"""Verified analytic adapters used by the v2.1 reference tests.

Only constant-background Minkowski and quadratic k-essence characteristic
relations are implemented.  Rotating k-essence and EdGB solvers remain
SPECIFIED_NOT_IMPLEMENTED in the complete standard.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import CharacteristicMode, ConeBundle, ModeKind


MINKOWSKI_INVERSE = np.diag([-1.0, 1.0, 1.0, 1.0])


@dataclass(frozen=True)
class EFTValidityReport:
    background_to_cutoff_ratio: float
    derivative_expansion_ratio: float
    minimum_background_ratio: float = 10.0
    maximum_derivative_ratio: float = 0.1

    @property
    def passed(self) -> bool:
        return (
            np.isfinite(self.background_to_cutoff_ratio)
            and np.isfinite(self.derivative_expansion_ratio)
            and self.background_to_cutoff_ratio >= self.minimum_background_ratio
            and 0.0 <= self.derivative_expansion_ratio <= self.maximum_derivative_ratio
        )


def metric_mode(inverse_metric: np.ndarray = MINKOWSKI_INVERSE) -> CharacteristicMode:
    return CharacteristicMode(
        mode_id="metric",
        kind=ModeKind.METRIC,
        inverse_metric=np.asarray(inverse_metric, dtype=float),
        physical=True,
        hyperbolic=True,
        validity_passed=True,
        notes=("quadratic metric null cone",),
    )


def kessence_characteristic(
    alpha: float,
    beta: float,
    dphi_covector: np.ndarray,
    inverse_metric: np.ndarray = MINKOWSKI_INVERSE,
) -> tuple[np.ndarray, float, float, float]:
    """Return Z^{mu nu}, X, P_X and c_s^2 for P=alpha X+beta X^2.

    The convention is X=-g^{mu nu} phi_mu phi_nu / 2 and signature -+++.
    Z^{mu nu}=P_X g^{mu nu}-P_XX nabla^mu phi nabla^nu phi.
    """

    g_inv = np.asarray(inverse_metric, dtype=float)
    q = np.asarray(dphi_covector, dtype=float)
    if g_inv.shape != (4, 4) or q.shape != (4,):
        raise ValueError("expected a 4D inverse metric and scalar covector")
    x = float(-0.5 * q @ g_inv @ q)
    p_x = float(alpha + 2.0 * beta * x)
    p_xx = float(2.0 * beta)
    grad_up = g_inv @ q
    z = p_x * g_inv - p_xx * np.outer(grad_up, grad_up)
    denominator = p_x + 2.0 * x * p_xx
    sound_speed_sq = float(p_x / denominator) if denominator != 0.0 else float("nan")
    return z, x, p_x, sound_speed_sq


def kessence_constant_bundle(
    solution_id: str,
    alpha: float,
    beta: float,
    dphi_covector: np.ndarray,
    eft: EFTValidityReport,
) -> ConeBundle:
    z, x, p_x, c_s2 = kessence_characteristic(alpha, beta, dphi_covector)
    scalar = CharacteristicMode(
        mode_id="scalar",
        kind=ModeKind.SCALAR,
        inverse_metric=z,
        physical=True,
        hyperbolic=bool(np.isfinite(c_s2) and c_s2 > 0.0 and abs(np.linalg.det(z)) > 1e-12),
        validity_passed=eft.passed,
        notes=(f"X={x:.12g}", f"P_X={p_x:.12g}", f"c_s^2={c_s2:.12g}"),
    )
    return ConeBundle(
        solution_id=solution_id,
        modes=(metric_mode(), scalar),
        expected_physical_mode_ids=("metric", "scalar"),
        domain_covered=True,
        notes=("constant-background analytic bundle",),
    )


__all__ = [
    "EFTValidityReport",
    "MINKOWSKI_INVERSE",
    "kessence_characteristic",
    "kessence_constant_bundle",
    "metric_mode",
]

