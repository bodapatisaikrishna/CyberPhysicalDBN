"""Tests for src/twin/consequence.py (claim C1's measured-not-asserted layer).

Two tiers: hand-built synthetic sensitivity/state fixtures pin `classify`'s
branch logic exactly (fast, no pandapower); a smaller real-grid tier uses
`GridModel`/`voltage_sensitivity` to check the zone map is sane against the
actual case33bw feeder and stable across the measured tau-invariance interval
(LAB_NOTEBOOK.md 2026-08-01: real interval [0.51, 0.685), not the ad-hoc
[0.55, 0.70] originally guessed).
"""

from __future__ import annotations

import itertools

import pytest

from src.attack_graph.graph import PHYS_LOCAL_DER, PHYS_WIDE_AREA
from src.twin.consequence import (
    PHYSICAL_OBSERVERS,
    DerZone,
    DeviationClass,
    ZoneMap,
    build_zone_map,
    classify,
    observe_local,
    observe_wide,
)
from src.twin.grid import (
    ActionOrigin,
    ControlAction,
    GridConfig,
    GridModel,
    GridState,
    VoltageLimits,
    voltage_sensitivity,
)

LIMITS = VoltageLimits(min_vm_pu=0.90, max_vm_pu=1.10, source="test fixture")


def _state(
    *,
    converged: bool = True,
    vm_pu: dict[int, float] | None = None,
    violated_buses: tuple[int, ...] = (),
    unstable: bool = False,
    vm_pu_max: float | None = 1.0,
    vm_pu_min: float | None = 1.0,
) -> GridState:
    return GridState(
        t_units=0.0,
        converged=converged,
        vm_pu=vm_pu or {},
        vm_pu_min=vm_pu_min,
        vm_pu_max=vm_pu_max,
        argmin_bus=None,
        argmax_bus=None,
        violated_buses=violated_buses,
        setpoints_mw={},
        limits=LIMITS,
        unstable=unstable,
    )


# Synthetic two-DER sensitivity: DER_A dominates buses 1-3, DER_B dominates
# buses 8-10, bus 5 is contested (no dominant DER at tau=0.6).
_SYNTH_SENSITIVITY = {
    "DER_A": {1: 0.09, 2: 0.08, 3: 0.07, 5: 0.05},
    "DER_B": {8: 0.09, 9: 0.08, 10: 0.07, 5: 0.05},
}
_SYNTH_DER_BUSES = {"DER_A": 1, "DER_B": 8}


@pytest.fixture
def synthetic_zones() -> ZoneMap:
    return build_zone_map(
        _SYNTH_SENSITIVITY, dominance_tau=0.6, delta_p_mw=0.1, der_buses=_SYNTH_DER_BUSES
    )


class TestBuildZoneMap:
    def test_dominant_buses_assigned_to_owning_der(self, synthetic_zones):
        zone_a = synthetic_zones.zone_for("DER_A")
        zone_b = synthetic_zones.zone_for("DER_B")
        assert zone_a.buses == frozenset({1, 2, 3})
        assert zone_b.buses == frozenset({8, 9, 10})

    def test_contested_bus_is_unassigned_not_dropped(self, synthetic_zones):
        assert 5 in synthetic_zones.unassigned_buses
        assert 5 not in synthetic_zones.zone_for("DER_A").buses
        assert 5 not in synthetic_zones.zone_for("DER_B").buses

    def test_zone_for_unknown_der_raises(self, synthetic_zones):
        with pytest.raises(KeyError):
            synthetic_zones.zone_for("DER_NOPE")

    def test_rejects_tau_at_or_below_half(self):
        with pytest.raises(ValueError):
            build_zone_map(
                _SYNTH_SENSITIVITY, dominance_tau=0.5, delta_p_mw=0.1, der_buses=_SYNTH_DER_BUSES
            )

    def test_rejects_tau_above_one(self):
        with pytest.raises(ValueError):
            build_zone_map(
                _SYNTH_SENSITIVITY, dominance_tau=1.01, delta_p_mw=0.1, der_buses=_SYNTH_DER_BUSES
            )

    def test_accepts_tau_equal_one(self):
        zm = build_zone_map(
            _SYNTH_SENSITIVITY, dominance_tau=1.0, delta_p_mw=0.1, der_buses=_SYNTH_DER_BUSES
        )
        assert isinstance(zm, ZoneMap)


class TestClassify:
    def test_state_none_is_unobserved_not_coerced(self, synthetic_zones):
        obs = classify(None, synthetic_zones)
        assert obs.converged is False
        assert obs.deviation is DeviationClass.NONCONVERGED
        assert observe_local(obs) is None
        assert observe_wide(obs) is None

    def test_nonconverged_mirrors_state_unstable_true(self, synthetic_zones):
        state = _state(converged=False, unstable=True)
        obs = classify(state, synthetic_zones)
        assert obs.deviation is DeviationClass.NONCONVERGED
        assert obs.exceeds_limit is True

    def test_nonconverged_mirrors_state_unstable_false(self, synthetic_zones):
        """Pins the fix: exceeds_limit must track state.unstable, never be
        hardcoded True, so it stays consistent with a config where
        nonconvergence_is_unstable=False."""
        state = _state(converged=False, unstable=False)
        obs = classify(state, synthetic_zones)
        assert obs.deviation is DeviationClass.NONCONVERGED
        assert obs.exceeds_limit is False

    def test_no_violated_buses_is_none_not_unobserved(self, synthetic_zones):
        state = _state(violated_buses=(), unstable=False)
        obs = classify(state, synthetic_zones)
        assert obs.deviation is DeviationClass.NONE
        assert obs.exceeds_limit is False
        assert observe_local(obs) == 0
        assert observe_wide(obs) == 0

    def test_violation_confined_to_one_zone_is_localized(self, synthetic_zones):
        state = _state(violated_buses=(1, 2), unstable=True)
        obs = classify(state, synthetic_zones)
        assert obs.deviation is DeviationClass.LOCALIZED
        assert obs.zones_violated == ("DER_A",)
        assert observe_local(obs) == 1
        assert observe_wide(obs) == 0

    def test_violation_spanning_two_zones_is_widespread(self, synthetic_zones):
        state = _state(violated_buses=(1, 8), unstable=True)
        obs = classify(state, synthetic_zones)
        assert obs.deviation is DeviationClass.WIDESPREAD
        assert set(obs.zones_violated) == {"DER_A", "DER_B"}
        assert observe_local(obs) == 0
        assert observe_wide(obs) == 1

    def test_violation_on_unassigned_bus_is_unclassified_not_none(self, synthetic_zones):
        state = _state(violated_buses=(5,), unstable=True)
        obs = classify(state, synthetic_zones)
        assert obs.deviation is DeviationClass.UNCLASSIFIED
        assert obs.zones_violated == ()

    def test_violation_partly_outside_any_zone_is_unclassified(self, synthetic_zones):
        state = _state(violated_buses=(1, 5), unstable=True)
        obs = classify(state, synthetic_zones)
        assert obs.deviation is DeviationClass.UNCLASSIFIED

    def test_localized_and_widespread_mutually_exclusive(self, synthetic_zones):
        for buses in [(1,), (1, 2, 3), (8, 9), (1, 8), (1, 2, 8, 9)]:
            obs = classify(_state(violated_buses=buses, unstable=True), synthetic_zones)
            assert not (obs.deviation is DeviationClass.LOCALIZED and observe_wide(obs) == 1)
            assert not (obs.deviation is DeviationClass.WIDESPREAD and observe_local(obs) == 1)

    def test_exceeds_prealarm_true_at_declared_limit_band(self, synthetic_zones):
        state = _state(violated_buses=(1,), unstable=True, vm_pu_max=1.15)
        obs = classify(state, synthetic_zones, prealarm_max_vm_pu=1.15)
        assert obs.exceeds_prealarm is True

    def test_exceeds_prealarm_can_trigger_below_declared_limit(self, synthetic_zones):
        """Sensitivity-arm-only: a tighter prealarm band can fire even when
        the declared-limit ground truth (exceeds_limit) has not."""
        state = _state(violated_buses=(), unstable=False, vm_pu_max=1.05)
        obs = classify(state, synthetic_zones, prealarm_max_vm_pu=1.02)
        assert obs.exceeds_limit is False
        assert obs.exceeds_prealarm is True


class TestObservers:
    def test_registry_maps_declared_node_names(self):
        assert set(PHYSICAL_OBSERVERS) == {PHYS_LOCAL_DER, PHYS_WIDE_AREA}
        assert PHYSICAL_OBSERVERS[PHYS_LOCAL_DER] is observe_local
        assert PHYSICAL_OBSERVERS[PHYS_WIDE_AREA] is observe_wide


# --- Real-grid tier: actual case33bw feeder, actual pandapower solves. -----


@pytest.fixture(scope="module")
def real_zone_map() -> ZoneMap:
    model = GridModel(GridConfig())
    sensitivity = voltage_sensitivity(model, delta_p_mw=0.5)
    der_buses = dict(zip(model.der_ids, model.der_buses))
    return build_zone_map(sensitivity, dominance_tau=0.6, delta_p_mw=0.5, der_buses=der_buses)


class TestRealGridZoneMap:
    def test_every_der_gets_a_nonempty_zone(self, real_zone_map):
        for zone in real_zone_map.zones:
            assert len(zone.buses) > 0

    def test_zones_are_pairwise_disjoint(self, real_zone_map):
        seen: set[int] = set()
        for zone in real_zone_map.zones:
            assert not (zone.buses & seen)
            seen |= zone.buses

    def test_every_reachable_ladder_state_classifies_non_unclassified(self, real_zone_map):
        """Sweeps every combination of the twin's configured p_mw_levels
        across both DERs (the full dispatch ladder) and asserts the zone map
        never leaves a converged, violating state UNCLASSIFIED."""
        config = GridConfig()
        unclassified = []
        for levels in itertools.product(config.p_mw_levels, repeat=config.n_der):
            model = GridModel(config)
            for der_id, p_mw in zip(model.der_ids, levels):
                model.apply_control_action(
                    ControlAction(der_id, p_mw, ActionOrigin.OPERATOR, 0.0)
                )
            state = model.solve(0.0)
            if not state.converged or not state.violated_buses:
                continue
            obs = classify(state, real_zone_map)
            if obs.deviation is DeviationClass.UNCLASSIFIED:
                unclassified.append((levels, state.violated_buses))
        assert unclassified == []

    def test_tau_invariant_over_measured_interval(self):
        """Freshly re-derived (this test, not carried over from any prior
        session's orientation-only exploration -- CLAUDE.md rule 2/4): with
        delta_p_mw=0.5 on case33bw, dominance shares have breakpoints at
        buses 27 (0.6457) and 9 (0.6884), so zone membership is invariant on
        tau in [0.6457, 0.6884) -- a band that contains the module
        docstring's recommended default of 2/3. Swept and confirmed via
        `python -c` before writing this assertion, not assumed."""
        model = GridModel(GridConfig())
        sensitivity = voltage_sensitivity(model, delta_p_mw=0.5)
        der_buses = dict(zip(model.der_ids, model.der_buses))

        reference = build_zone_map(
            sensitivity, dominance_tau=0.65, delta_p_mw=0.5, der_buses=der_buses
        )
        ref_buses = {z.der_id: z.buses for z in reference.zones}

        for tau in (0.65, 0.66, 2 / 3, 0.68):
            zm = build_zone_map(sensitivity, dominance_tau=tau, delta_p_mw=0.5, der_buses=der_buses)
            buses = {z.der_id: z.buses for z in zm.zones}
            assert buses == ref_buses, f"zone membership changed at tau={tau}"

    def test_tau_outside_interval_changes_membership_control(self):
        """Control for the invariance test above: confirms it isn't vacuous
        by showing membership DOES change just outside [0.6457, 0.6884) --
        at 0.64 (below) and 0.69 (above)."""
        model = GridModel(GridConfig())
        sensitivity = voltage_sensitivity(model, delta_p_mw=0.5)
        der_buses = dict(zip(model.der_ids, model.der_buses))

        inside = build_zone_map(sensitivity, dominance_tau=0.65, delta_p_mw=0.5, der_buses=der_buses)
        below = build_zone_map(sensitivity, dominance_tau=0.64, delta_p_mw=0.5, der_buses=der_buses)
        above = build_zone_map(sensitivity, dominance_tau=0.69, delta_p_mw=0.5, der_buses=der_buses)

        inside_buses = {z.der_id: z.buses for z in inside.zones}
        below_buses = {z.der_id: z.buses for z in below.zones}
        above_buses = {z.der_id: z.buses for z in above.zones}
        assert below_buses != inside_buses
        assert above_buses != inside_buses
