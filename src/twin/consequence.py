"""Physical deviation as a measured outcome (CLAUDE.md layer [0]/[4], claim C1).

Pure: `GridState -> PhysicalObservation`. No SimPy, no pandapower, no DBN. Power
flow lives in `grid.py`; this module only interprets an already-solved state.

The point of this module is that `UnstablePS` stops being an attack-graph
assertion and becomes a measurement: `classify` never looks at the attack
graph or at any attack-step completion time, only at `GridState`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping

from src.attack_graph.graph import PHYS_LOCAL_DER, PHYS_WIDE_AREA
from src.twin.grid import GridState


class DeviationClass(Enum):
    NONE = "none"
    LOCALIZED = "localized"
    WIDESPREAD = "widespread"
    UNCLASSIFIED = "unclassified"  # violated buses fit no zone pattern
    NONCONVERGED = "nonconverged"  # geography undefined


@dataclass(frozen=True)
class DerZone:
    der_id: str
    bus: int
    buses: frozenset[int]


@dataclass(frozen=True)
class ZoneMap:
    zones: tuple[DerZone, ...]
    unassigned_buses: frozenset[int]
    sensitivity_dv_dp: Mapping[str, Mapping[int, float]]
    dominance_tau: float
    delta_p_mw: float
    rule: str  # provenance string

    def zone_for(self, der_id: str) -> DerZone:
        for zone in self.zones:
            if zone.der_id == der_id:
                return zone
        raise KeyError(f"no zone for {der_id!r}")


def build_zone_map(
    sensitivity: Mapping[str, Mapping[int, float]],
    *,
    dominance_tau: float,
    delta_p_mw: float,
    der_buses: Mapping[str, int],
) -> ZoneMap:
    """Derive DER-to-bus zones from sensitivity shares.

    share_d(b) = S[d][b] / sum_d' S[d'][b], defined only where the denominator
    is positive. zone(d) = {b : share_d(b) >= dominance_tau}.

    tau=2/3 is the recommended default: the midpoint of the interval over
    which zone membership was found to be invariant on case33bw
    (LAB_NOTEBOOK.md 2026-08-01) -- a number backed by a logged sweep
    (experiments/exp04_closed_loop_c1.py stage 0), not asserted here.
    """
    if not 0.5 < dominance_tau <= 1.0:
        raise ValueError(
            f"dominance_tau must be in (0.5, 1.0] for zones to be well-defined "
            f"(a bus cannot dominantly belong to two DERs at once), got {dominance_tau}"
        )

    der_ids = list(sensitivity)
    all_buses: set[int] = set()
    for buses in sensitivity.values():
        all_buses |= set(buses)

    zone_buses: dict[str, set[int]] = {d: set() for d in der_ids}
    unassigned: set[int] = set()
    for bus in all_buses:
        total = sum(sensitivity[d].get(bus, 0.0) for d in der_ids)
        if total <= 0:
            unassigned.add(bus)
            continue
        shares = {d: sensitivity[d].get(bus, 0.0) / total for d in der_ids}
        dominant = max(shares, key=shares.get)
        if shares[dominant] >= dominance_tau:
            zone_buses[dominant].add(bus)
        else:
            unassigned.add(bus)

    zones = tuple(
        DerZone(der_id=d, bus=der_buses[d], buses=frozenset(zone_buses[d]))
        for d in der_ids
    )
    return ZoneMap(
        zones=zones,
        unassigned_buses=frozenset(unassigned),
        sensitivity_dv_dp=sensitivity,
        dominance_tau=dominance_tau,
        delta_p_mw=delta_p_mw,
        rule="dv_dp dominance share >= dominance_tau",
    )


@dataclass(frozen=True)
class PhysicalObservation:
    t_units: float
    converged: bool
    deviation: DeviationClass
    violated_buses: tuple[int, ...]
    zones_violated: tuple[str, ...]
    vm_pu_max: float | None
    vm_pu_min: float | None
    exceeds_limit: bool  # the declared-limit ground truth for instability
    exceeds_prealarm: bool  # sensitivity-arm only; == exceeds_limit at the primary band
    prealarm_max_vm_pu: float


def classify(
    state: GridState | None,
    zones: ZoneMap,
    *,
    prealarm_max_vm_pu: float | None = None,
) -> PhysicalObservation:
    """Map a solved grid state to a physical observation.

    `state is None` means no solve has occurred yet at this instant (before
    the first slice, or a gap in the trace) -- distinct from a solve that
    happened and did not converge. Both cases mean deviation is unobserved
    (`NONE`-shaped but the caller should treat this slice as unobserved for
    physical evidence -- see runner.discretize), never coerced to 0.
    """
    if state is None:
        return PhysicalObservation(
            t_units=float("nan"),
            converged=False,
            deviation=DeviationClass.NONCONVERGED,
            violated_buses=(),
            zones_violated=(),
            vm_pu_max=None,
            vm_pu_min=None,
            exceeds_limit=False,
            exceeds_prealarm=False,
            prealarm_max_vm_pu=prealarm_max_vm_pu or 0.0,
        )

    band = prealarm_max_vm_pu if prealarm_max_vm_pu is not None else state.limits.max_vm_pu

    if not state.converged:
        # exceeds_limit mirrors state.unstable exactly, rather than hardcoding
        # True: GridState.unstable for a non-converged solve is already the
        # config-driven decision (GridConfig.nonconvergence_is_unstable), and
        # record.grid_unstable == obs.exceeds_limit must hold unconditionally
        # (pinned by a test) -- hardcoding here would silently break that if
        # nonconvergence_is_unstable were ever set False.
        return PhysicalObservation(
            t_units=state.t_units,
            converged=False,
            deviation=DeviationClass.NONCONVERGED,
            violated_buses=(),
            zones_violated=(),
            vm_pu_max=None,
            vm_pu_min=None,
            exceeds_limit=state.unstable,
            exceeds_prealarm=True,
            prealarm_max_vm_pu=band,
        )

    if not state.violated_buses:
        return PhysicalObservation(
            t_units=state.t_units,
            converged=True,
            deviation=DeviationClass.NONE,
            violated_buses=(),
            zones_violated=(),
            vm_pu_max=state.vm_pu_max,
            vm_pu_min=state.vm_pu_min,
            exceeds_limit=False,
            exceeds_prealarm=(state.vm_pu_max is not None and state.vm_pu_max > band),
            prealarm_max_vm_pu=band,
        )

    violated = set(state.violated_buses)
    hit_zones = tuple(
        sorted(zone.der_id for zone in zones.zones if violated & zone.buses)
    )
    outside_any_zone = violated - {b for zone in zones.zones for b in zone.buses}

    if outside_any_zone or not hit_zones:
        deviation = DeviationClass.UNCLASSIFIED
    elif len(hit_zones) == 1:
        deviation = DeviationClass.LOCALIZED
    else:
        deviation = DeviationClass.WIDESPREAD

    return PhysicalObservation(
        t_units=state.t_units,
        converged=True,
        deviation=deviation,
        violated_buses=state.violated_buses,
        zones_violated=hit_zones,
        vm_pu_max=state.vm_pu_max,
        vm_pu_min=state.vm_pu_min,
        exceeds_limit=True,
        exceeds_prealarm=True,
        prealarm_max_vm_pu=band,
    )


def observe_local(obs: PhysicalObservation) -> int | None:
    """PhysLocalDER: 1 iff the deviation is confined to exactly one DER zone.

    Returns None when the observation itself is unobserved (no solve, or
    non-converged -- geography is undefined, not "no").
    """
    if obs.deviation is DeviationClass.NONCONVERGED and not obs.converged:
        return None
    return int(obs.deviation is DeviationClass.LOCALIZED)


def observe_wide(obs: PhysicalObservation) -> int | None:
    """PhysWideArea: 1 iff the deviation spans more than one DER zone."""
    if obs.deviation is DeviationClass.NONCONVERGED and not obs.converged:
        return None
    return int(obs.deviation is DeviationClass.WIDESPREAD)


# Node names are owned by src.attack_graph.graph (the authoritative source of
# every node name in the model); imported here rather than redeclared, so this
# registry cannot silently name a node the graph doesn't actually have.
PHYSICAL_OBSERVERS: Mapping[str, Callable[[PhysicalObservation], "int | None"]] = {
    PHYS_LOCAL_DER: observe_local,
    PHYS_WIDE_AREA: observe_wide,
}
