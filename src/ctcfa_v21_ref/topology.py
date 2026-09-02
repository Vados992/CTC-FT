"""Topology-aware closure and stationary-axisymmetric sector diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, pi

import numpy as np

from .model import ClaimScope, CTCWitness, TopologySpec


def _integer_multiple(value: float, period: float, tol: float) -> bool:
    ratio = value / period
    return isclose(ratio, round(ratio), abs_tol=tol, rel_tol=0.0)


@dataclass(frozen=True)
class KillingOrbitClosure:
    parameter_period: float
    delta_t: float
    delta_phi: float
    closed: bool
    reason: str


def killing_orbit_closure(
    time_component: float,
    angular_component: float,
    topology: TopologySpec,
    windings: int = 1,
    tol: float = 1e-10,
) -> KillingOrbitClosure:
    """Test the natural axial period of ``a*d_t + b*d_phi``.

    If ``b`` is nonzero, the candidate parameter period is chosen to produce
    ``windings`` angular revolutions.  With unwrapped time, closure additionally
    requires zero time displacement.  With periodic time it requires a deck
    transformation by an integer number of time periods.
    """

    if windings == 0:
        raise ValueError("windings must be nonzero")
    if topology.angular_period is None:
        return KillingOrbitClosure(0.0, 0.0, 0.0, False, "axial coordinate is not periodic")
    if isclose(angular_component, 0.0, abs_tol=tol):
        if topology.time_period is None:
            return KillingOrbitClosure(0.0, 0.0, 0.0, False, "no compact generator")
        parameter = topology.time_period / abs(time_component)
    else:
        parameter = abs(windings) * topology.angular_period / abs(angular_component)
    delta_t = time_component * parameter
    delta_phi = angular_component * parameter
    phi_closed = _integer_multiple(delta_phi, topology.angular_period, tol)
    if topology.time_period is None:
        time_closed = isclose(delta_t, 0.0, abs_tol=tol, rel_tol=0.0)
        reason = "unwrapped time requires zero delta_t"
    else:
        time_closed = _integer_multiple(delta_t, topology.time_period, tol)
        reason = "time displacement must be a deck period"
    return KillingOrbitClosure(parameter, delta_t, delta_phi, phi_closed and time_closed, reason)


def killing_block(g_tt: float, g_tphi: float, g_phiphi: float) -> dict[str, float | bool]:
    determinant = g_tt * g_phiphi - g_tphi * g_tphi
    return {
        "g_phiphi": float(g_phiphi),
        "determinant": float(determinant),
        "axial_ctc_candidate": bool(g_phiphi < 0.0),
        "lorentzian_killing_plane": bool(determinant < 0.0),
    }


def axial_witness(
    g_phiphi: float,
    topology: TopologySpec,
    mode_id: str = "metric",
    tol: float = 1e-12,
) -> CTCWitness:
    closed = topology.angular_period is not None
    margin = float(g_phiphi)
    return CTCWitness(
        mode_id=mode_id,
        closed=closed,
        future_directed=closed and margin < -tol,
        timelike_margin=margin,
        scope=ClaimScope.SECTOR,
        notes=("closed axial orbit", "orientation convention fixed for compact orbit"),
    )


def periodic_time_witness(
    g_tt: float,
    topology: TopologySpec,
    mode_id: str = "metric",
) -> CTCWitness:
    closed = topology.time_period is not None
    return CTCWitness(
        mode_id=mode_id,
        closed=closed,
        future_directed=closed and g_tt < 0.0,
        timelike_margin=float(g_tt),
        scope=ClaimScope.GLOBAL,
        notes=("periodic-time deck orbit",),
    )


__all__ = [
    "KillingOrbitClosure",
    "axial_witness",
    "killing_block",
    "killing_orbit_closure",
    "periodic_time_witness",
]

