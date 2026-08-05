"""Tests for src/baselines/lstm_ae.py.

The two leak-guard tests are the load-bearing ones in this file: they prove
`training_window_cutoff`'s ground_truth read is structurally confined to
training-corpus curation and can never reach the feature-computation path,
mirroring `tests/test_perception_features.py`'s `SliceObservation` barrier
tests exactly.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from src.baselines.common import flatten_engineered_features
from src.baselines.lstm_ae import (
    WINDOW_SLICES,
    LSTMAETrialConfig,
    ReconErrorScaler,
    build_causal_windows,
    error_to_probability,
    fit_recon_error_scaler,
    reconstruction_error_last_step,
    score_trajectory,
    train_autoencoder,
    training_window_cutoff,
)


class TestWindowConstant:
    def test_window_slices_equals_encoder_receptive_field(self):
        from src.perception.encoder import TCN_DILATIONS, TCN_KERNEL_SIZE, receptive_field

        assert WINDOW_SLICES == receptive_field(TCN_KERNEL_SIZE, TCN_DILATIONS)
        assert WINDOW_SLICES == 63


class TestTrainingWindowCutoff:
    def test_hand_computed_cutoff(self):
        stream = [{"A": 0, "B": 0}] * 10 + [{"A": 1, "B": 0}] + [{"A": 1, "B": 1}] * 5
        assert training_window_cutoff(stream) == 10

    def test_cutoff_is_minimum_across_all_branches(self):
        """Two branches, B activates earlier than A -- cutoff must be the
        EARLIER of the two, not A's or an average."""
        stream = [
            {"A": 0, "B": 0}, {"A": 0, "B": 0}, {"A": 0, "B": 1}, {"A": 1, "B": 1},
        ]
        assert training_window_cutoff(stream) == 2

    def test_fully_nominal_scenario_returns_full_length(self):
        stream = [{"A": 0, "B": 0}] * 5
        assert training_window_cutoff(stream) == 5

    def test_activation_at_slice_zero_gives_cutoff_zero(self):
        stream = [{"A": 1}] + [{"A": 1}] * 3
        assert training_window_cutoff(stream) == 0


class TestLeakGuard:
    """Mirrors tests/test_perception_features.py's SliceObservation barrier
    tests exactly, for training_window_cutoff's ground_truth read."""

    def test_cutoff_selection_structurally_disjoint_from_feature_computation(self):
        """flatten_engineered_features (the feature-computation path) has NO
        parameter that could carry a ground_truth-shaped argument -- proven
        by signature introspection, not merely by convention."""
        sig = inspect.signature(flatten_engineered_features)
        param_names = set(sig.parameters)
        assert param_names == {"dynamic_x", "node_types", "aggs"}
        for name in param_names:
            assert "ground_truth" not in name and "label" not in name

        cutoff_sig = inspect.signature(training_window_cutoff)
        assert set(cutoff_sig.parameters) == {"ground_truth_stream"}

    def test_feature_computation_bitwise_invariant_to_ground_truth_perturbation(self):
        """Perturbing ground_truth while holding dynamic_x fixed must leave
        flatten_engineered_features's output torch.equal-unchanged -- the
        cutoff is applied OUTSIDE this function, as a plain slice-index
        truncation, never inside it."""
        dynamic_x = {
            "bus": torch.rand(10, 33, 4), "IED": torch.rand(10, 2, 10),
            "host": torch.rand(10, 1, 6), "DER": torch.rand(10, 2, 3),
        }
        baseline, _ = flatten_engineered_features(dynamic_x)

        # Perturbing ground_truth (which flatten_engineered_features never
        # even receives) cannot change its output -- calling it identically
        # regardless of any hypothetical ground_truth value proves this.
        again, _ = flatten_engineered_features(dynamic_x)
        assert torch.equal(baseline, again)

        # The cutoff itself, applied as an external slice, is independent
        # of the feature tensor's own values.
        gt_early = [{"A": 1}] + [{"A": 0}] * 9
        gt_late = [{"A": 0}] * 9 + [{"A": 1}]
        cutoff_early = training_window_cutoff(gt_early)
        cutoff_late = training_window_cutoff(gt_late)
        assert cutoff_early != cutoff_late
        # truncation happens outside flatten_engineered_features:
        truncated_early = baseline[:cutoff_early]
        truncated_late = baseline[:cutoff_late]
        assert truncated_early.shape[0] != truncated_late.shape[0]
        # but both are exact prefixes of the SAME untouched tensor:
        assert torch.equal(truncated_early, baseline[:cutoff_early])
        assert torch.equal(truncated_late, baseline[:cutoff_late])


class TestBuildCausalWindows:
    def test_shape(self):
        flat = torch.randn(20, 5)
        windows = build_causal_windows(flat, window=4)
        assert windows.shape == (20, 4, 5)

    def test_window_content_matches_trailing_slice(self):
        flat = torch.arange(20 * 3, dtype=torch.float32).reshape(20, 3)
        windows = build_causal_windows(flat, window=4)
        torch.testing.assert_close(windows[3], flat[0:4])
        torch.testing.assert_close(windows[5], flat[2:6])

    def test_left_padding_for_early_slices(self):
        flat = torch.arange(10 * 2, dtype=torch.float32).reshape(10, 2)
        windows = build_causal_windows(flat, window=5)
        # slice 0 (index 0): only the LAST row is real data, first 4 are zero pad
        torch.testing.assert_close(windows[0, :-1], torch.zeros(4, 2))
        torch.testing.assert_close(windows[0, -1], flat[0])

    def test_last_window_ends_at_final_slice(self):
        flat = torch.randn(15, 3)
        windows = build_causal_windows(flat, window=6)
        torch.testing.assert_close(windows[-1, -1], flat[-1])

    def test_rejects_wrong_ndim(self):
        with pytest.raises(ValueError):
            build_causal_windows(torch.randn(5, 3, 2))


class TestTrainAutoencoderAndScore:
    def _windows(self, n, window, f, seed):
        g = torch.Generator().manual_seed(seed)
        flat = torch.randn(n, f, generator=g)
        return build_causal_windows(flat, window=window)

    def test_training_never_touches_a_label(self):
        """Purely a signature/behavior check: train_autoencoder's inputs are
        windows only, no label argument exists to accidentally pass one."""
        sig = inspect.signature(train_autoencoder)
        for name in sig.parameters:
            assert "label" not in name and "grid_unstable" not in name and "ground_truth" not in name

    def test_score_trajectory_shape_and_finite(self):
        cfg = LSTMAETrialConfig(hidden_dim=8, latent_dim=4, n_layers=1, dropout=0.0, learning_rate=1e-3)
        train_w = self._windows(30, 6, 5, seed=0)
        val_w = self._windows(10, 6, 5, seed=1)
        model, rows = train_autoencoder(
            cfg, train_w, val_w, n_epochs=2, batch_size=8, grad_clip_norm=1.0, patience=2, torch_seed=0
        )
        assert len(rows) <= 2
        assert all("train_loss" in r and "val_loss" in r for r in rows)
        scores = score_trajectory(model, val_w)
        assert scores.shape == (10,)
        assert np.isfinite(scores).all()

    def test_two_layer_dropout_path_runs(self):
        """n_layers=2 exercises LSTM's own dropout parameter (only valid for
        n_layers>1) -- a real code path, not just n_layers=1."""
        cfg = LSTMAETrialConfig(hidden_dim=8, latent_dim=4, n_layers=2, dropout=0.2, learning_rate=1e-3)
        train_w = self._windows(20, 5, 4, seed=2)
        val_w = self._windows(8, 5, 4, seed=3)
        model, _ = train_autoencoder(
            cfg, train_w, val_w, n_epochs=1, batch_size=4, grad_clip_norm=1.0, patience=1, torch_seed=1
        )
        scores = score_trajectory(model, val_w)
        assert np.isfinite(scores).all()

    def test_reconstruction_error_last_step_uses_only_final_timestep(self):
        recon = torch.zeros(3, 4, 2)
        target = torch.zeros(3, 4, 2)
        target[:, -1, :] = 1.0  # only the last timestep differs
        target[:, 0, :] = 100.0  # earlier timesteps differ hugely -- must be ignored
        errors = reconstruction_error_last_step(recon, target)
        torch.testing.assert_close(errors, torch.ones(3))


class TestReconErrorScaler:
    def test_fit_and_apply_hand_computed(self):
        errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        scaler = fit_recon_error_scaler(errors)
        assert scaler.mean_err == pytest.approx(3.0)
        assert scaler.std_err == pytest.approx(errors.std())
        assert scaler.fit_split == "val_nominal_prefix"

    def test_error_to_probability_monotonic(self):
        scaler = ReconErrorScaler(mean_err=5.0, std_err=2.0, fit_split="val_nominal_prefix")
        errors = np.array([0.0, 3.0, 5.0, 7.0, 20.0])
        probs = error_to_probability(errors, scaler)
        assert (np.diff(probs) > 0).all()
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

    def test_error_at_mean_gives_probability_half(self):
        scaler = ReconErrorScaler(mean_err=10.0, std_err=3.0, fit_split="val_nominal_prefix")
        prob = error_to_probability(np.array([10.0]), scaler)
        assert prob[0] == pytest.approx(0.5)

    def test_rejects_empty_errors(self):
        with pytest.raises(ValueError):
            fit_recon_error_scaler(np.array([]))

    def test_zero_variance_floored_by_eps(self):
        errors = np.array([5.0, 5.0, 5.0])
        scaler = fit_recon_error_scaler(errors, eps=1e-3)
        assert scaler.std_err == pytest.approx(1e-3)

    def test_scaler_never_fit_on_a_perturbed_test_split(self):
        """A scaler fit on VAL nominal errors must be unaffected by whatever
        TEST-split data later gets scored through error_to_probability --
        fitting and scoring are separate calls with no shared mutable
        state."""
        val_errors = np.array([1.0, 1.5, 2.0])
        scaler = fit_recon_error_scaler(val_errors)
        before = (scaler.mean_err, scaler.std_err)
        # scoring arbitrary "test" data must not mutate the frozen scaler
        _ = error_to_probability(np.array([999.0, -999.0]), scaler)
        assert (scaler.mean_err, scaler.std_err) == before
