"""Quadratic characteristic cones and common temporal certificates."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .model import ConeBundle, TemporalCertificate, TopologySpec


def mode_margin(inverse_metric: np.ndarray, gradient_covector: np.ndarray) -> float:
    h = np.asarray(inverse_metric, dtype=float)
    q = np.asarray(gradient_covector, dtype=float)
    return float(q @ h @ q)


def temporal_single_valued(gradient_covector: np.ndarray, topology: TopologySpec, tol: float = 1e-12) -> bool:
    q = np.asarray(gradient_covector, dtype=float)
    if q.shape != (4,):
        return False
    if topology.time_period is not None and abs(q[0]) > tol:
        return False
    if topology.angular_period is not None and abs(q[3]) > tol:
        return False
    return True


def common_temporal_certificate(
    bundle: ConeBundle,
    gradient_covector: Iterable[float],
    topology: TopologySpec,
    epsilon: float = 1e-10,
) -> TemporalCertificate:
    q = np.asarray(tuple(gradient_covector), dtype=float)
    margins: dict[str, float] = {}
    for mode in bundle.physical_modes:
        if mode.inverse_metric is None:
            continue
        margins[mode.mode_id] = mode_margin(mode.inverse_metric, q)
    return TemporalCertificate(
        gradient_covector=tuple(float(v) for v in q),
        single_valued=temporal_single_valued(q, topology),
        mode_margins=margins,
        epsilon=epsilon,
        domain_covered=bundle.domain_covered and bundle.complete,
    )


__all__ = ["common_temporal_certificate", "mode_margin", "temporal_single_valued"]

