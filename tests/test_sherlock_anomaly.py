"""Tests for src/perception/sherlock_anomaly.py."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.baselines.lstm_ae import (
    LSTMAETrialConfig,
    LSTMAutoencoder,
    build_causal_windows,
    score_trajectory,
)
from src.perception.sherlock_anomaly import (
    apply_standardizer,
    blocked_split,
    causal_windows_for_endpoints,
    endpoints_in_ranges,
    score_ae_ranges,
    event_runs,
    fit_standardizer,
    score_ae_batched,
    windows_ending_at,
    zscore_mean_abs,
    _padded,
)


class TestStandardizer:
    def test_hand_computed_mean_and_population_std(self):
        x = torch.tensor([[1.0, 10.0], [3.0, 10.0], [5.0, 10.0]])
        s = fit_standardizer(x, fit_split="unit")
        assert s.mean.tolist() == pytest.approx([3.0, 10.0])
        # population std of [1,3,5] = sqrt(((-2)^2+0+2^2)/3) = sqrt(8/3)
        assert s.std[0].item() == pytest.approx((8.0 / 3.0) ** 0.5, rel=1e-6)
        assert s.fit_split == "unit"

    def test_constant_column_standardizes_to_zero_not_nan(self):
        x = torch.tensor([[1.0, 10.0], [3.0, 10.0]])
        z = apply_standardizer(x, fit_standardizer(x, fit_split="unit"))
        assert torch.isfinite(z).all()
        assert z[:, 1].tolist() == [0.0, 0.0]

    def test_standardized_fit_chunk_has_zero_mean_unit_std(self):
        torch.manual_seed(0)
        x = torch.randn(500, 4) * torch.tensor([1.0, 1e3, 1e-3, 5.0]) + torch.tensor([0.0, 1e6, 3.0, -2.0])
        z = apply_standardizer(x, fit_standardizer(x, fit_split="unit"))
        assert z.mean(dim=0).abs().max().item() < 1e-3
        assert (z.std(dim=0, unbiased=False) - 1.0).abs().max().item() < 1e-3

    def test_rejects_empty_and_wrong_rank(self):
        with pytest.raises(ValueError):
            fit_standardizer(torch.zeros(0, 3), fit_split="unit")
        with pytest.raises(ValueError):
            fit_standardizer(torch.zeros(3), fit_split="unit")


class TestWindows:
    def test_windows_ending_at_matches_build_causal_windows(self):
        torch.manual_seed(1)
        x = torch.randn(40, 3)
        ref = build_causal_windows(x, 7)
        got = windows_ending_at(_padded(x, 7), torch.arange(40), 7)
        assert torch.equal(got, ref)

    def test_public_endpoint_form_equals_reference_rows(self):
        x = torch.randn(30, 2)
        ends = torch.tensor([5, 11, 29])
        assert torch.equal(causal_windows_for_endpoints(x, ends, 6), build_causal_windows(x, 6)[ends])

    def test_left_zero_pad_only_at_sequence_start(self):
        x = torch.arange(1.0, 11.0).unsqueeze(1)  # slices 1..10, F=1
        w = windows_ending_at(_padded(x, 4), torch.tensor([0, 2, 9]), 4)
        assert w[0, :, 0].tolist() == [0.0, 0.0, 0.0, 1.0]
        assert w[1, :, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
        assert w[2, :, 0].tolist() == [7.0, 8.0, 9.0, 10.0]  # true context, no padding


class TestScoreAeBatched:
    def _model(self, f):
        torch.manual_seed(2)
        return LSTMAutoencoder(f, LSTMAETrialConfig(hidden_dim=6, latent_dim=3, n_layers=1, dropout=0.0, learning_rate=1e-3))

    def test_identical_to_score_trajectory_on_full_window_tensor(self):
        torch.manual_seed(3)
        x = torch.randn(60, 4)
        model = self._model(4)
        ref = score_trajectory(model, build_causal_windows(x, 9))
        got = score_ae_batched(model, x, 9, batch_size=7)  # batch size does not divide S
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)

    def test_sub_range_uses_true_preceding_context(self):
        torch.manual_seed(4)
        x = torch.randn(50, 3)
        model = self._model(3)
        ref = score_trajectory(model, build_causal_windows(x, 6))
        got = score_ae_batched(model, x, 6, batch_size=8, start=20, end=35)
        np.testing.assert_allclose(got, ref[20:35], rtol=1e-5, atol=1e-6)
        assert got.shape == (15,)


class TestZscoreMeanAbs:
    def test_hand_computed(self):
        z = torch.tensor([[1.0, -3.0], [0.0, 0.0], [-2.0, 2.0]])
        assert zscore_mean_abs(z).tolist() == [2.0, 0.0, 2.0]


class TestEventRuns:
    def test_contiguous_runs_and_none_breaks(self):
        ids = [None, "14", "14", None, "14", "20", "20", None]
        assert event_runs(ids) == [("14", 1, 3), ("14", 4, 5), ("20", 5, 7)]

    def test_run_touching_end_is_closed(self):
        assert event_runs([None, "9", "9"]) == [("9", 1, 3)]

    def test_no_events(self):
        assert event_runs([None, None]) == []
        assert event_runs([]) == []

    def test_run_lengths_sum_to_positive_slice_count(self):
        ids = [None, "1", "1", "2", None, "2", "2", "2"]
        runs = event_runs(ids)
        assert sum(e - s for _, s, e in runs) == sum(i is not None for i in ids)


class TestZClip:
    def test_clip_caps_a_frozen_feature_that_later_moves(self):
        fit = torch.tensor([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])  # column 1 constant in the fit data
        s = fit_standardizer(fit, eps=1e-6, fit_split="unit", z_clip=10.0)
        later = torch.tensor([[2.0, 5.0], [2.0, 132.5]])  # column 1 jumps by 127.5
        z = apply_standardizer(later, s)
        assert z[1, 1].item() == 10.0  # not 1.275e8
        assert z[0, 1].item() == 0.0

    def test_no_clip_is_the_identity_on_the_old_behaviour(self):
        x = torch.tensor([[1.0], [3.0]])
        a = apply_standardizer(x, fit_standardizer(x, fit_split="unit"))
        b = apply_standardizer(x, fit_standardizer(x, fit_split="unit", z_clip=None))
        assert torch.equal(a, b) and a.abs().max().item() == pytest.approx(1.0)

    def test_clip_is_symmetric_and_leaves_small_values_alone(self):
        x = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]])
        s = fit_standardizer(x, fit_split="unit", z_clip=1.0)
        z = apply_standardizer(torch.tensor([[-100.0], [2.2], [100.0]]), s)
        assert z[:, 0].tolist()[0] == -1.0 and z[:, 0].tolist()[2] == 1.0
        assert abs(z[1, 0].item()) < 1.0

    def test_rejects_non_positive_clip(self):
        with pytest.raises(ValueError):
            fit_standardizer(torch.ones(3, 1), fit_split="unit", z_clip=0.0)


class TestBlockedSplit:
    def test_cyclic_assignment_hand_computed(self):
        got = blocked_split(10, 2, ("a", "a", "b"))
        assert got == {"a": [(0, 2), (2, 4), (6, 8), (8, 10)], "b": [(4, 6)]}

    def test_ranges_are_disjoint_and_cover_everything(self):
        got = blocked_split(103, 7, ("fit", "fit", "fit", "val", "calib"))
        covered = sorted(r for rs in got.values() for r in rs)
        assert covered[0][0] == 0 and covered[-1][1] == 103
        assert all(covered[i][1] == covered[i + 1][0] for i in range(len(covered) - 1))

    def test_every_split_spans_the_whole_recording(self):
        got = blocked_split(43204, 1800, ("fit", "fit", "fit", "val", "calib"))
        for name, rs in got.items():
            assert rs[0][0] < 10000 and rs[-1][1] > 33000, name  # early and late blocks in each

    def test_rejects_degenerate_arguments(self):
        with pytest.raises(ValueError):
            blocked_split(10, 0, ("a",))
        with pytest.raises(ValueError):
            blocked_split(10, 2, ())


class TestEndpointsAndRangeScoring:
    def test_endpoints_hand_computed_with_min_end(self):
        e = endpoints_in_ranges([(0, 10), (20, 26)], stride=3, min_end=4)
        assert e.tolist() == [4, 7, 20, 23]

    def test_endpoints_empty_when_every_range_precedes_min_end(self):
        assert endpoints_in_ranges([(0, 3)], stride=1, min_end=5).numel() == 0

    def test_score_ae_ranges_concatenates_in_order(self):
        torch.manual_seed(5)
        x = torch.randn(40, 3)
        model = LSTMAutoencoder(3, LSTMAETrialConfig(hidden_dim=5, latent_dim=2, n_layers=1, dropout=0.0, learning_rate=1e-3))
        ref = score_trajectory(model, build_causal_windows(x, 6))
        got = score_ae_ranges(model, x, 6, [(30, 38), (2, 9)], batch_size=4)
        np.testing.assert_allclose(got, np.concatenate([ref[30:38], ref[2:9]]), rtol=1e-5, atol=1e-6)
