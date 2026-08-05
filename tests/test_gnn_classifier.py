"""Tests for src/baselines/gnn_classifier.py.

The block-diagonal-replication-equals-per-slice-loop test is run for BOTH
conv types deliberately: it is the direct verification of this session's own
flagged risk (`GATConv`'s default `add_self_loops=True` raises on a
heterogeneous edge type; `(-1,-1)` lazy in_channels is required for both
conv types) -- a wiring mistake here would silently corrupt every subsequent
number, not raise an exception.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.baselines.gnn_classifier import GNNBaselineConfig, GNNClassifier, HeteroSpatialLayer
from src.perception.encoder import flatten_for_spatial, replicate_static_graph, unflatten_from_spatial

NODE_TYPES = ["bus", "line", "DER", "IED", "host"]
COUNTS = {"bus": 33, "line": 37, "DER": 2, "IED": 2, "host": 1}
F_DIMS = {"bus": 10, "line": 5, "DER": 6, "IED": 11, "host": 7}


def _rand_edges(rng, n_src, n_dst, m):
    return torch.tensor(
        np.stack([rng.integers(0, n_src, m), rng.integers(0, n_dst, m)]), dtype=torch.long
    )


def _edge_index_dict(rng):
    ei = {
        ("bus", "electrical_coupling", "bus"): _rand_edges(rng, 33, 33, 32),
        ("bus", "electrical_coupling", "line"): _rand_edges(rng, 33, 37, 74),
        ("bus", "electrical_coupling", "DER"): _rand_edges(rng, 33, 2, 2),
        ("host", "network_reachability", "IED"): _rand_edges(rng, 1, 2, 2),
        ("IED", "control_authority", "DER"): _rand_edges(rng, 2, 2, 2),
    }
    ei[("line", "rev_electrical_coupling", "bus")] = ei[("bus", "electrical_coupling", "line")].flip(0)
    ei[("DER", "rev_electrical_coupling", "bus")] = ei[("bus", "electrical_coupling", "DER")].flip(0)
    ei[("IED", "rev_network_reachability", "host")] = ei[("host", "network_reachability", "IED")].flip(0)
    ei[("DER", "rev_control_authority", "IED")] = ei[("IED", "control_authority", "DER")].flip(0)
    return ei


def _random_inputs(S: int, B: int = 1, seed: int = 0):
    torch.manual_seed(seed)
    x_dict = {t: torch.randn(B, S, COUNTS[t], F_DIMS[t], requires_grad=True) for t in NODE_TYPES}
    globals_ = torch.randn(B, S, 6, requires_grad=True)
    ei = _edge_index_dict(np.random.default_rng(seed))
    return x_dict, ei, globals_


def _model(conv_type: str, seed: int = 1, **overrides) -> GNNClassifier:
    torch.manual_seed(seed)
    ei = _edge_index_dict(np.random.default_rng(0))
    cfg_kwargs = dict(conv_type=conv_type, n_layers=2, hidden=16, heads=2, temporal_kernel_size=5)
    cfg_kwargs.update(overrides)
    cfg = GNNBaselineConfig(**cfg_kwargs)
    model = GNNClassifier(tuple(NODE_TYPES), tuple(ei.keys()), cfg)
    model.eval()
    return model


class TestGNNBaselineConfig:
    def test_rejects_unknown_conv_type(self):
        with pytest.raises(ValueError):
            GNNBaselineConfig(conv_type="bogus", n_layers=1, hidden=8)

    def test_rejects_nonpositive_n_layers(self):
        with pytest.raises(ValueError):
            GNNBaselineConfig(conv_type="gat", n_layers=0, hidden=8)

    def test_rejects_nonpositive_temporal_kernel(self):
        with pytest.raises(ValueError):
            GNNBaselineConfig(conv_type="sage", n_layers=1, hidden=8, temporal_kernel_size=0)


@pytest.mark.parametrize("conv_type", ["gat", "sage"])
class TestOutputShape:
    def test_output_shape_is_B_S(self, conv_type):
        model = _model(conv_type)
        x_dict, ei, globals_ = _random_inputs(S=12)
        with torch.no_grad():
            out = model(x_dict, ei, globals_)
        assert out.shape == (1, 12)
        assert torch.isfinite(out).all()

    def test_batched_output_shape(self, conv_type):
        model = _model(conv_type)
        x_dict, ei, globals_ = _random_inputs(S=8, B=3, seed=2)
        with torch.no_grad():
            out = model(x_dict, ei, globals_)
        assert out.shape == (3, 8)

    def test_single_layer_runs(self, conv_type):
        """n_layers=1 is a deliberate ablation point (per the plan: it
        should fail to reach the 3-hop cyber-shortcut signal), but it must
        still run without error."""
        model = _model(conv_type, n_layers=1)
        x_dict, ei, globals_ = _random_inputs(S=6, seed=3)
        with torch.no_grad():
            out = model(x_dict, ei, globals_)
        assert out.shape == (1, 6)


@pytest.mark.parametrize("conv_type", ["gat", "sage"])
class TestCausality:
    def test_causality_by_gradient(self, conv_type):
        model = _model(conv_type)
        x_dict, ei, globals_ = _random_inputs(S=20)
        out = model(x_dict, ei, globals_)
        t_check = 10
        out[0, t_check].backward()
        assert x_dict["bus"].grad[0, t_check + 1:].abs().max().item() == 0.0
        assert globals_.grad[0, t_check + 1:].abs().max().item() == 0.0
        assert x_dict["bus"].grad[0, :t_check + 1].abs().max().item() > 0.0

    def test_causality_by_perturbation(self, conv_type):
        model = _model(conv_type)
        x_dict, ei, globals_ = _random_inputs(S=20)
        x_dict = {t: x.detach() for t, x in x_dict.items()}
        globals_ = globals_.detach()
        with torch.no_grad():
            baseline = model(x_dict, ei, globals_)
        t_check = 10
        perturbed_x = {t: x.clone() for t, x in x_dict.items()}
        perturbed_x["bus"][0, t_check + 3] += 1000.0
        with torch.no_grad():
            perturbed = model(perturbed_x, ei, globals_)
        before = (perturbed[0, :t_check + 1] - baseline[0, :t_check + 1]).abs().max().item()
        after = (perturbed[0, t_check + 3:] - baseline[0, t_check + 3:]).abs().max().item()
        assert before < 1e-4  # small BLAS-noise tolerance, see test_perception_encoder.py
        assert after > 1e-3


@pytest.mark.parametrize("conv_type", ["gat", "sage"])
class TestBlockDiagonalReplication:
    def test_replication_equals_per_slice_loop(self, conv_type):
        """The direct verification of this module's flagged risk: a wrong
        add_self_loops/lazy-in_channels wiring would silently corrupt this
        comparison rather than raise, for either conv type."""
        torch.manual_seed(5)
        ei = _edge_index_dict(np.random.default_rng(1))
        cfg = GNNBaselineConfig(conv_type=conv_type, n_layers=2, hidden=8, heads=2)
        layer = HeteroSpatialLayer(tuple(NODE_TYPES), tuple(ei.keys()), cfg)
        layer.eval()

        S = 5
        x_dict = {t: torch.randn(1, S, COUNTS[t], F_DIMS[t]) for t in NODE_TYPES}
        flat, num_nodes, b, s = flatten_for_spatial(x_dict)
        rep = replicate_static_graph(ei, num_nodes, b * s)
        with torch.no_grad():
            h_flat = layer(flat, rep)
        batched = unflatten_from_spatial(h_flat, num_nodes, b, s)

        per_slice = {t: [] for t in NODE_TYPES}
        with torch.no_grad():
            for slice_idx in range(S):
                xs = {t: x_dict[t][0, slice_idx] for t in NODE_TYPES}
                out_s = layer(xs, ei)
                for t in NODE_TYPES:
                    per_slice[t].append(out_s[t])
        per_slice = {t: torch.stack(v, dim=0).unsqueeze(0) for t, v in per_slice.items()}

        for t in NODE_TYPES:
            torch.testing.assert_close(batched[t], per_slice[t], atol=1e-5, rtol=1e-5)


class TestDeterminism:
    @pytest.mark.parametrize("conv_type", ["gat", "sage"])
    def test_deterministic_given_torch_seed(self, conv_type):
        x_dict, ei, globals_ = _random_inputs(S=10, seed=9)
        x_dict = {t: x.detach() for t, x in x_dict.items()}
        globals_ = globals_.detach()
        model_a = _model(conv_type, seed=42)
        with torch.no_grad():
            out_a = model_a(x_dict, ei, globals_)
        model_b = _model(conv_type, seed=42)
        with torch.no_grad():
            out_b = model_b(x_dict, ei, globals_)
        torch.testing.assert_close(out_a, out_b)
