"""Tests for src/perception/features.py.

The leak guard (`test_features_bitwise_invariant_to_label_field_perturbation`)
is the load-bearing test in this file: it proves the perception pipeline is
STRUCTURALLY incapable of training on the labels it is meant to predict, not
merely that it doesn't happen to today.
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest
import torch

from src.attack_graph.graph import build_attack_graph
from src.perception.asset_graph import (
    CyberAsset,
    CyberOverlayConfig,
    build_asset_graph,
    observed_endpoints,
)
from src.perception.features import (
    DynamicFeatureConfig,
    SanitizedMessage,
    SliceObservation,
    build_dynamic_features,
    build_slice_observations,
    fit_feature_scaler,
    ied_dynamic_features,
    sanitize_bus_log,
    slice_of,
)
from src.twin.comms import (
    ActionOrigin,
    DeliveryRecord,
    Message,
    MessageType,
    PayloadCategory,
)
from src.twin.grid import GridConfig, GridModel
from src.twin.runner import TwinConfig, TwinRunner

DELTA_T = 166.13 / 600
N_SLICES = 722  # matches T_TIME_UNITS=200 at this delta_t, per exp03/exp04 convention


def _overlay() -> CyberOverlayConfig:
    return CyberOverlayConfig(
        assets=(
            CyberAsset("ControlCentre", "host", "ControlCentre"),
            CyberAsset("IED_17", "IED", "DER_17", controls=("DER_17",)),
            CyberAsset("IED_32", "IED", "DER_32", controls=("DER_32",)),
        ),
        source="test fixture",
    )


@pytest.fixture(scope="module")
def real_trace():
    ag = build_attack_graph()
    config = TwinConfig(horizon_time_units=200.0)
    return TwinRunner(ag, config, np.random.SeedSequence(5)).run()


@pytest.fixture(scope="module")
def asset_graph(real_trace):
    grid = GridModel(GridConfig())
    obs = observed_endpoints(real_trace.messages)
    return build_asset_graph(grid, _overlay(), observed_endpoints_set=obs)


class TestSliceObservationWhitelist:
    def test_fields_are_exactly_whitelisted(self):
        fields = {f.name for f in dataclasses.fields(SliceObservation)}
        assert fields == {"slice_index", "t_units", "grid", "messages"}

    def test_sanitized_message_carries_no_origin_or_tampered_by(self):
        fields = {f.name for f in dataclasses.fields(SanitizedMessage)}
        assert "origin" not in fields
        assert "tampered_by" not in fields


class TestSanitizeBusLog:
    def test_drops_origin_tampered_by_and_spoofed_payload_key(self):
        msg = Message(
            MessageType.MEASUREMENT, "DER_17", "ControlCentre",
            PayloadCategory.VOLTAGE_REPORT,
            {"vm_pu_min": 0.85, "spoofed": "true"},
            timestamp=1.0, origin=ActionOrigin.ATTACKER, tampered_by=("SpoofRepMsg",),
        )
        log = [DeliveryRecord(sent=msg, delivered=msg)]
        (sanitized,) = sanitize_bus_log(log, viewpoint="ControlCentre")
        assert sanitized.numeric_payload == {"vm_pu_min": 0.85}
        assert "spoofed" not in sanitized.numeric_payload
        assert not hasattr(sanitized, "origin")
        assert not hasattr(sanitized, "tampered_by")

    def test_viewpoint_never_takes_both_ends_of_one_message(self):
        """A tampered measurement: sent (DER's true value) differs from
        delivered (spoofed). From ControlCentre's viewpoint, ONLY delivered
        is observable -- sent's true value must not leak through."""
        true_msg = Message(
            MessageType.MEASUREMENT, "DER_17", "ControlCentre",
            PayloadCategory.VOLTAGE_REPORT, {"vm_pu_min": 1.05}, timestamp=2.0,
        )
        spoofed_msg = dataclasses.replace(
            true_msg, payload={"vm_pu_min": 0.85, "spoofed": "true"}, tampered_by=("SpoofRepMsg",)
        )
        log = [DeliveryRecord(sent=true_msg, delivered=spoofed_msg)]
        (sanitized,) = sanitize_bus_log(log, viewpoint="ControlCentre")
        assert sanitized.numeric_payload["vm_pu_min"] == pytest.approx(0.85)

        # from the DER's own viewpoint, only the (unmodified in this case)
        # SENT copy is observable -- delivered is never consulted for a
        # message this endpoint originated.
        (sanitized_der,) = sanitize_bus_log(log, viewpoint="DER_17")
        assert sanitized_der.numeric_payload["vm_pu_min"] == pytest.approx(1.05)

    def test_message_not_involving_viewpoint_is_dropped(self):
        msg = Message(
            MessageType.MEASUREMENT, "DER_32", "ControlCentre",
            PayloadCategory.VOLTAGE_REPORT, {"vm_pu_min": 1.0}, timestamp=1.0,
        )
        log = [DeliveryRecord(sent=msg, delivered=msg)]
        assert sanitize_bus_log(log, viewpoint="DER_17") == ()

    def test_dropped_message_is_absent_not_reconstructed(self):
        """delivered=None (a hook dropped it) must yield NO SanitizedMessage
        for the receiving viewpoint -- never falling back to `sent`, which
        would leak the ORIGINAL value past whatever dropped it."""
        msg = Message(
            MessageType.COMMAND, "ControlCentre", "DER_17",
            PayloadCategory.SETPOINT, {"p_mw": 5.0}, timestamp=3.0,
        )
        log = [DeliveryRecord(sent=msg, delivered=None)]
        assert sanitize_bus_log(log, viewpoint="DER_17") == ()
        # the SENDER's own viewpoint still sees its own outbound log though
        # (it does not know delivery failed at the payload level).
        (sanitized,) = sanitize_bus_log(log, viewpoint="ControlCentre")
        assert sanitized.numeric_payload["p_mw"] == pytest.approx(5.0)


class TestSliceOf:
    def test_message_at_exactly_slice_boundary_lands_in_that_slice(self):
        """Matches discretize()'s slice-end convention exactly."""
        assert slice_of(3 * DELTA_T, DELTA_T, N_SLICES) == 3

    def test_message_just_after_boundary_lands_in_next_slice(self):
        assert slice_of(3 * DELTA_T + 1e-6, DELTA_T, N_SLICES) == 4

    def test_message_at_t_zero_lands_in_slice_one(self):
        assert slice_of(0.0, DELTA_T, N_SLICES) == 1

    def test_clamped_to_n_slices(self):
        assert slice_of(1e9, DELTA_T, N_SLICES) == N_SLICES

    def test_rejects_nonpositive_delta_t(self):
        with pytest.raises(ValueError):
            slice_of(1.0, 0.0, 10)


class TestBuildSliceObservations:
    def test_length_equals_n_slices(self, real_trace):
        observations = build_slice_observations(real_trace, DELTA_T, N_SLICES)
        assert len(observations) == N_SLICES
        assert [o.slice_index for o in observations] == list(range(1, N_SLICES + 1))

    def test_empty_slice_fraction_derived_from_config_not_a_literal(self, real_trace):
        """~72% empty is a DERIVED fact of (dispatch_period=1.0, delta_t),
        not asserted as a magic number. 200 report instants (t=0..199) fall
        into distinct slices at this delta_t (delta_t < 1.0 time unit), so
        the empty fraction is computed from the config, then checked against
        the measured trace within a loose statistical band (commands add
        extra non-empty slices depending on the run, so this cannot be
        exact)."""
        dispatch_period = 1.0
        expected_report_slices = int(200 / dispatch_period)  # one report burst per period
        expected_empty_frac_upper_bound = 1.0 - (expected_report_slices / N_SLICES) + 0.02

        observations = build_slice_observations(real_trace, DELTA_T, N_SLICES)
        n_empty = sum(1 for o in observations if len(o.messages) == 0)
        measured_frac = n_empty / N_SLICES
        assert measured_frac < expected_empty_frac_upper_bound
        assert measured_frac > 0.5  # a real, non-trivial sparsity, not "basically dense"


class TestDynamicFeatures:
    def test_no_nan_or_inf_anywhere(self, real_trace, asset_graph):
        feats = build_dynamic_features(
            real_trace, DELTA_T, N_SLICES, asset_graph, _overlay(), GridConfig(),
            dispatch_period_time_units=1.0,
        )
        for name, tensor in feats.items():
            assert torch.isfinite(tensor).all(), f"{name} has non-finite values"

    def test_no_nan_on_nonconverged_slice(self, asset_graph):
        """A synthetic trace with zero grid solves (grid_state_at always
        returns None) must still produce finite bus features."""
        from src.twin.runner import ContinuousTrace

        empty_trace = ContinuousTrace(horizon_time_units=10.0)
        feats = build_dynamic_features(
            empty_trace, DELTA_T, 5, asset_graph, _overlay(), GridConfig(),
            dispatch_period_time_units=1.0,
        )
        assert torch.isfinite(feats["bus"]).all()
        assert (feats["bus"][:, :, 1] == 0.0).all()  # converged bit always 0

    def test_shapes_match_asset_graph_counts(self, real_trace, asset_graph):
        feats = build_dynamic_features(
            real_trace, DELTA_T, N_SLICES, asset_graph, _overlay(), GridConfig(),
            dispatch_period_time_units=1.0,
        )
        assert feats["bus"].shape == (N_SLICES, 33, 4)
        assert feats["IED"].shape == (N_SLICES, 2, 10)
        assert feats["host"].shape == (N_SLICES, 1, 6)
        assert feats["DER"].shape == (N_SLICES, 2, 3)
        for t in ("line", "transformer", "RTU", "relay"):
            assert feats[t].shape[-1] == 0
            assert feats[t].shape[1] == asset_graph.counts[t]

    def test_voltage_only_excludes_setpoint_telemetry(self, real_trace, asset_graph):
        """Perturbing setpoints_mw in every recorded GridState must not move
        a single bit of the voltage_only feature tensors -- the actual
        invariant, not merely that DER happens to be all-zero."""
        base = build_dynamic_features(
            real_trace, DELTA_T, N_SLICES, asset_graph, _overlay(), GridConfig(),
            dispatch_period_time_units=1.0,
            config=DynamicFeatureConfig(observability="voltage_only"),
        )
        perturbed_trace = copy.deepcopy(real_trace)
        for i, (t, state) in enumerate(perturbed_trace.grid_solves):
            new_setpoints = {k: v + 999.0 for k, v in state.setpoints_mw.items()}
            perturbed_trace.grid_solves[i] = (t, dataclasses.replace(state, setpoints_mw=new_setpoints))
        perturbed = build_dynamic_features(
            perturbed_trace, DELTA_T, N_SLICES, asset_graph, _overlay(), GridConfig(),
            dispatch_period_time_units=1.0,
            config=DynamicFeatureConfig(observability="voltage_only"),
        )
        for name in base:
            assert torch.equal(base[name], perturbed[name]), f"{name} changed under voltage_only"

    def test_full_telemetry_exposes_setpoint_and_differs_from_voltage_only(self, real_trace, asset_graph):
        voltage_only = build_dynamic_features(
            real_trace, DELTA_T, N_SLICES, asset_graph, _overlay(), GridConfig(),
            dispatch_period_time_units=1.0,
            config=DynamicFeatureConfig(observability="voltage_only"),
        )
        full = build_dynamic_features(
            real_trace, DELTA_T, N_SLICES, asset_graph, _overlay(), GridConfig(),
            dispatch_period_time_units=1.0,
            config=DynamicFeatureConfig(observability="full_telemetry"),
        )
        assert not torch.equal(voltage_only["DER"], full["DER"])
        assert voltage_only["DER"].shape == full["DER"].shape  # width constant across arms
        assert (full["DER"][:, :, 0] > 0).any()  # telemetry_available fires somewhere

    def test_se_noise_sigma_zero_is_noiseless(self, real_trace, asset_graph):
        a = build_dynamic_features(
            real_trace, DELTA_T, N_SLICES, asset_graph, _overlay(), GridConfig(),
            dispatch_period_time_units=1.0, config=DynamicFeatureConfig(se_noise_sigma=0.0),
        )
        b = build_dynamic_features(
            real_trace, DELTA_T, N_SLICES, asset_graph, _overlay(), GridConfig(),
            dispatch_period_time_units=1.0, config=DynamicFeatureConfig(se_noise_sigma=0.0),
        )
        assert torch.equal(a["bus"], b["bus"])

    def test_se_noise_sigma_positive_requires_rng(self, real_trace, asset_graph):
        with pytest.raises(ValueError, match="rng"):
            build_dynamic_features(
                real_trace, DELTA_T, N_SLICES, asset_graph, _overlay(), GridConfig(),
                dispatch_period_time_units=1.0,
                config=DynamicFeatureConfig(se_noise_sigma=0.01),
            )

    def test_se_noise_sigma_positive_perturbs_voltage_only(self, real_trace, asset_graph):
        clean = build_dynamic_features(
            real_trace, DELTA_T, N_SLICES, asset_graph, _overlay(), GridConfig(),
            dispatch_period_time_units=1.0,
            config=DynamicFeatureConfig(se_noise_sigma=0.02), rng=np.random.default_rng(0),
        )
        noisy = build_dynamic_features(
            real_trace, DELTA_T, N_SLICES, asset_graph, _overlay(), GridConfig(),
            dispatch_period_time_units=1.0,
            config=DynamicFeatureConfig(se_noise_sigma=0.02), rng=np.random.default_rng(1),
        )
        assert not torch.equal(clean["bus"][:, :, 0], noisy["bus"][:, :, 0])


class TestZohAndStaleness:
    def _fixture_assets(self):
        return [CyberAsset("IED_17", "IED", "DER_17")]

    def _obs(self, messages_by_slice: dict[int, list[SanitizedMessage]], n_slices: int):
        return [
            SliceObservation(slice_index=s, t_units=float(s), grid=None,
                              messages=tuple(messages_by_slice.get(s, [])))
            for s in range(1, n_slices + 1)
        ]

    def test_zoh_carries_forward_and_staleness_grows(self):
        msg = SanitizedMessage("measurement", "DER_17", "ControlCentre", "voltage_report",
                                {"vm_pu_min": 0.95}, timestamp=1.0)
        observations = self._obs({1: [msg]}, n_slices=4)
        x = ied_dynamic_features(observations, self._fixture_assets(), p_mw_levels=[0.0, 0.8], max_staleness_slices=50.0)
        # slice 1: has_report=1, zoh=0.95, staleness=0
        assert x[0, 0, 0] == 1.0
        assert x[0, 0, 2] == pytest.approx(0.95)
        assert x[0, 0, 3] == 0.0
        # slices 2-4: has_report=0, zoh STILL 0.95 (carried forward), staleness grows
        for s in range(1, 4):
            assert x[s, 0, 0] == 0.0
            assert x[s, 0, 2] == pytest.approx(0.95)
            assert x[s, 0, 3] == pytest.approx(float(s))

    def test_staleness_capped(self):
        msg = SanitizedMessage("measurement", "DER_17", "ControlCentre", "voltage_report",
                                {"vm_pu_min": 1.0}, timestamp=1.0)
        observations = self._obs({1: [msg]}, n_slices=10)
        x = ied_dynamic_features(observations, self._fixture_assets(), p_mw_levels=[0.0], max_staleness_slices=3.0)
        assert x[-1, 0, 3] == pytest.approx(3.0)

    def test_before_first_observation_zoh_is_default_not_leaked(self):
        observations = self._obs({}, n_slices=3)
        x = ied_dynamic_features(observations, self._fixture_assets(), p_mw_levels=[0.0, 0.8], max_staleness_slices=50.0)
        assert (x[:, 0, 0] == 0.0).all()  # has_report always 0
        assert torch.allclose(x[:, 0, 2], torch.ones_like(x[:, 0, 2]))  # default nominal, never NaN

    def test_command_ladder_index_tracks_rung(self):
        cmd = SanitizedMessage("command", "ControlCentre", "DER_17", "setpoint",
                                {"p_mw": 2.0}, timestamp=1.0)
        observations = self._obs({1: [cmd]}, n_slices=2)
        levels = [0.0, 0.8, 2.0, 3.0, 5.0]
        x = ied_dynamic_features(observations, self._fixture_assets(), p_mw_levels=levels, max_staleness_slices=50.0)
        assert x[0, 0, 8] == pytest.approx(2.0)  # index of 2.0 in levels
        assert x[0, 0, 9] == pytest.approx(1.0)  # n_cmds_so_far

    def test_ladder_index_distinguishes_gradual_climb_from_a_jump(self):
        """The genuine CommandCoherence signature: a legitimate climb visits
        every intermediate rung; UnauthCommand jumps straight to the top."""
        levels = [0.0, 0.8, 2.0, 3.0, 5.0]
        gradual = self._obs(
            {1: [SanitizedMessage("command", "ControlCentre", "DER_17", "setpoint", {"p_mw": 0.8}, 1.0)],
             2: [SanitizedMessage("command", "ControlCentre", "DER_17", "setpoint", {"p_mw": 2.0}, 2.0)],
             3: [SanitizedMessage("command", "ControlCentre", "DER_17", "setpoint", {"p_mw": 3.0}, 3.0)]},
            n_slices=3,
        )
        jump = self._obs(
            {1: [SanitizedMessage("command", "ControlCentre", "DER_17", "setpoint", {"p_mw": 5.0}, 1.0)]},
            n_slices=3,
        )
        x_gradual = ied_dynamic_features(gradual, self._fixture_assets(), p_mw_levels=levels, max_staleness_slices=50.0)
        x_jump = ied_dynamic_features(jump, self._fixture_assets(), p_mw_levels=levels, max_staleness_slices=50.0)
        gradual_indices = x_gradual[:, 0, 8].tolist()
        jump_indices = x_jump[:, 0, 8].tolist()
        assert gradual_indices == [1.0, 2.0, 3.0]
        assert jump_indices == [4.0, 4.0, 4.0]
        # the max single-step jump size is what a model needs to see:
        assert max(abs(b - a) for a, b in zip(gradual_indices, gradual_indices[1:])) == 1.0
        assert jump_indices[0] - 0.0 == 4.0  # straight to the top rung from nothing


class TestFeatureScaler:
    def test_fit_and_transform_zero_means_unit_std(self):
        x = torch.randn(100, 5, 3) * 2.0 + 10.0
        scaler = fit_feature_scaler(x)
        transformed = scaler.transform(x)
        assert transformed.mean(dim=(0, 1)).abs().max() < 0.05
        assert (transformed.std(dim=(0, 1)) - 1.0).abs().max() < 0.05

    def test_frozen_scaler_applied_to_new_data_does_not_refit(self):
        train = torch.randn(50, 3) * 5.0 + 1.0
        scaler = fit_feature_scaler(train)
        held_out = torch.zeros(10, 3)  # wildly different distribution
        transformed = scaler.transform(held_out)
        # transform must use TRAIN stats, not re-center held_out to mean 0
        assert not torch.allclose(transformed.mean(dim=0), torch.zeros(3), atol=1e-3)

    def test_constant_feature_does_not_divide_by_zero(self):
        x = torch.zeros(20, 4)
        scaler = fit_feature_scaler(x)
        transformed = scaler.transform(x)
        assert torch.isfinite(transformed).all()

    def test_rejects_width_mismatch(self):
        scaler = fit_feature_scaler(torch.randn(10, 3))
        with pytest.raises(ValueError):
            scaler.transform(torch.randn(10, 4))


class TestLeakGuard:
    """The load-bearing test: no perturbation of any LABEL field anywhere in
    `ContinuousTrace` may move a single bit of the dynamic feature tensors."""

    def test_features_bitwise_invariant_to_label_field_perturbation(self, real_trace, asset_graph):
        baseline = build_dynamic_features(
            real_trace, DELTA_T, N_SLICES, asset_graph, _overlay(), GridConfig(),
            dispatch_period_time_units=1.0,
        )

        def _rebuild(trace):
            return build_dynamic_features(
                trace, DELTA_T, N_SLICES, asset_graph, _overlay(), GridConfig(),
                dispatch_period_time_units=1.0,
            )

        # 1. flip every ground-truth-adjacent trace field this module must
        #    never read (step_completion_times, events, reaction_outcomes).
        t1 = copy.deepcopy(real_trace)
        t1.step_completion_times = {k: 0.0 for k in t1.step_completion_times}
        t1.reaction_outcomes = {k: not v for k, v in t1.reaction_outcomes.items()}
        t1.events = []
        assert_equal = lambda a, b, name: [  # noqa: E731
            torch.testing.assert_close(a[k], b[k], rtol=0, atol=0, msg=f"{name}:{k}") for k in a
        ]
        assert_equal(baseline, _rebuild(t1), "ground_truth_fields")

        # 2. flip origin/tampered_by on every message.
        t2 = copy.deepcopy(real_trace)
        t2.messages = tuple(
            DeliveryRecord(
                sent=dataclasses.replace(
                    r.sent, origin=ActionOrigin.ATTACKER, tampered_by=("MITM", "SpoofRepMsg")
                ),
                delivered=(
                    dataclasses.replace(
                        r.delivered, origin=ActionOrigin.ATTACKER,
                        tampered_by=("MITM", "SpoofRepMsg"),
                    )
                    if r.delivered is not None else None
                ),
            )
            for r in t2.messages
        )
        assert_equal(baseline, _rebuild(t2), "origin_tampered_by")

        # 3. inject payload["spoofed"] = "true" into every message.
        t3 = copy.deepcopy(real_trace)
        t3.messages = tuple(
            DeliveryRecord(
                sent=dataclasses.replace(r.sent, payload={**r.sent.payload, "spoofed": "true"}),
                delivered=(
                    dataclasses.replace(r.delivered, payload={**r.delivered.payload, "spoofed": "true"})
                    if r.delivered is not None else None
                ),
            )
            for r in t3.messages
        )
        assert_equal(baseline, _rebuild(t3), "spoofed_payload_key")

    def test_structural_guard_new_leaky_field_fails_ci(self):
        """If a future edit adds a field to SliceObservation, this test
        fails immediately rather than silently widening the audit surface."""
        assert {f.name for f in dataclasses.fields(SliceObservation)} == {
            "slice_index", "t_units", "grid", "messages"
        }
