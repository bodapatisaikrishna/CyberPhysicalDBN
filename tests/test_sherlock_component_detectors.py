"""Hand-computed checks for the label-free component detectors."""

import numpy as np
import pytest
import torch

from src.perception.sherlock_component_detectors import component_scores, fit_component_model, standardized


def _fit(x, **kw):
    cfg = dict(eps=1e-6, z_clip=10.0, explained_variance_target=0.99, max_components=8, fit_split="test")
    cfg.update(kw)
    return fit_component_model(x, **cfg)


def test_rank_one_data_spe_zero_in_subspace_and_positive_off_it():
    g = torch.Generator().manual_seed(0)
    t = torch.randn(500, 1, generator=g)
    x = torch.cat([t, 2 * t, -t], dim=1)  # exactly rank 1
    m = _fit(x)
    assert m.components.shape[1] == 1
    on = torch.tensor([[1.0, 2.0, -1.0]])
    off = torch.tensor([[1.0, -2.0, -1.0]])  # violates the column relation
    s_on = component_scores(m, on, top_k=2)["pca_spe"][0]
    s_off = component_scores(m, off, top_k=2)["pca_spe"][0]
    assert s_on < 1e-2 and s_off > 1.0


def test_nan_filled_with_fit_median_not_zero_signal():
    x = torch.tensor([[1.0, 5.0], [3.0, 5.0], [2.0, float("nan")], [2.0, 5.0]])
    m = _fit(x)
    assert float(m.fill[1]) == 5.0  # median of finite values only
    z = standardized(m, torch.tensor([[2.0, float("nan")]]))
    assert torch.isfinite(z).all()


def test_constant_in_fit_column_moving_later_is_clipped_not_exploding():
    x = torch.cat([torch.randn(200, 1), torch.zeros(200, 1)], dim=1)
    m = _fit(x)
    assert m.n_constant_in_fit == 1
    s = component_scores(m, torch.tensor([[0.0, 1e6]]), top_k=1)
    assert s["max_abs_z"][0] == pytest.approx(10.0)  # z_clip, not 1e12


def test_max_and_top_k_hand_computed():
    x = torch.tensor([[0.0, 0.0], [2.0, 2.0]] * 50)  # mean 1, std 1 per column
    m = _fit(x)
    s = component_scores(m, torch.tensor([[4.0, 1.0]]), top_k=2)  # z = (3, 0)
    assert s["max_abs_z"][0] == pytest.approx(3.0, abs=1e-5)
    assert s["top_k_mean_abs_z"][0] == pytest.approx(1.5, abs=1e-5)


def test_rolling_robust_z_hand_computed_and_causal():
    from src.perception.sherlock_component_detectors import causal_rolling_robust_z

    x = torch.tensor([[1.0], [1.0], [1.0], [1.0], [9.0], [1.0]])
    z, _ = causal_rolling_robust_z(x, window=4, min_history=2, scale_floor=torch.tensor([1.0]), z_clip=10.0)
    assert z[0, 0] == 0.0 and z[1, 0] == 0.0          # < min_history past values -> 0
    assert z[4, 0] == pytest.approx((9.0 - 1.0) / 1.4826, rel=1e-5)  # past median 1, scale = floor 1
    # causal: changing a FUTURE value must not change an earlier z
    x2 = x.clone(); x2[5, 0] = 100.0
    z2, _ = causal_rolling_robust_z(x2, window=4, min_history=2, scale_floor=torch.tensor([1.0]), z_clip=10.0)
    assert torch.equal(z[:5], z2[:5])


def test_rolling_robust_z_clips_and_handles_nan():
    from src.perception.sherlock_component_detectors import causal_rolling_robust_z

    x = torch.tensor([[1.0], [1.0], [1.0], [float("nan")], [1000.0]])
    z, _ = causal_rolling_robust_z(x, window=4, min_history=2, scale_floor=torch.tensor([0.001]), z_clip=10.0)
    assert torch.isfinite(z).all() and z[4, 0] == 10.0 and z[3, 0] == 0.0
