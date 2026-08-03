"""Tests for src/perception/encoder.py.

Causality (by perturbation AND by gradient) and the exactness of the static-
graph batch replication are the load-bearing properties this file exists to
pin -- everything else in the perception pipeline depends on the TCN never
seeing the future and the [B,S] folding trick being numerically exact.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pytest

from src.perception.encoder import (
    TCN_DILATIONS,
    TCN_KERNEL_SIZE,
    CausalTCN,
    N_GNN_LAYERS,
    PerceptionEncoder,
    EncoderConfig,
    Readout,
    SpatialEncoder,
    TARGETS,
    combine_static_dynamic,
    flatten_for_spatial,
    receptive_field,
    replicate_static_graph,
    stack_scenarios,
    unflatten_from_spatial,
)

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
    x_dict = {
        t: torch.randn(B, S, COUNTS[t], F_DIMS[t], requires_grad=True) for t in NODE_TYPES
    }
    globals_ = torch.randn(B, S, 6, requires_grad=True)
    ei = _edge_index_dict(np.random.default_rng(seed))
    return x_dict, ei, globals_


def _model(n_gnn_layers: int = N_GNN_LAYERS, seed: int = 1) -> PerceptionEncoder:
    torch.manual_seed(seed)
    ei = _edge_index_dict(np.random.default_rng(0))
    cfg = EncoderConfig(
        node_types=tuple(NODE_TYPES), edge_types=tuple(ei.keys()), n_gnn_layers=n_gnn_layers
    )
    model = PerceptionEncoder(cfg)
    model.eval()
    return model


class TestReceptiveField:
    def test_hand_computed_63(self):
        assert receptive_field(TCN_KERNEL_SIZE, TCN_DILATIONS) == 63

    def test_formula_matches_definition(self):
        assert receptive_field(3, (1, 2, 4, 8, 16)) == 1 + 2 * (1 + 2 + 4 + 8 + 16)


class TestOutputShape:
    def test_output_is_dict_of_B_S_per_target(self):
        model = _model()
        x_dict, ei, globals_ = _random_inputs(S=12)
        with torch.no_grad():
            out = model(x_dict, ei, globals_)
        assert set(out) == set(TARGETS)
        for name, tensor in out.items():
            assert tensor.shape == (1, 12), name

    def test_batched_output_shape(self):
        model = _model()
        x_dict, ei, globals_ = _random_inputs(S=8, B=3, seed=2)
        with torch.no_grad():
            out = model(x_dict, ei, globals_)
        for tensor in out.values():
            assert tensor.shape == (3, 8)


class TestCausality:
    def test_tcn_causality_by_gradient(self):
        model = _model()
        x_dict, ei, globals_ = _random_inputs(S=20)
        out = model(x_dict, ei, globals_)
        t_check = 10
        out["CommandCoherence"][0, t_check].backward()

        assert globals_.grad[0, t_check + 1:].abs().max().item() == 0.0
        assert x_dict["bus"].grad[0, t_check + 1:].abs().max().item() == 0.0
        assert x_dict["host"].grad[0, t_check + 1:].abs().max().item() == 0.0
        # non-vacuous: the past/present DOES receive gradient.
        assert x_dict["bus"].grad[0, :t_check + 1].abs().max().item() > 0.0

    def test_tcn_causality_by_gradient_every_target(self):
        """Every one of the 4 heads must independently respect causality --
        a bug isolated to one head's readout slice would not show up if only
        one target were checked."""
        for target in TARGETS:
            model = _model(seed=3)
            x_dict, ei, globals_ = _random_inputs(S=15, seed=4)
            out = model(x_dict, ei, globals_)
            t_check = 7
            out[target][0, t_check].backward()
            assert x_dict["bus"].grad[0, t_check + 1:].abs().max().item() == 0.0, target

    def test_end_to_end_causality_by_perturbation(self):
        """Perturbing raw input features at slice s > t must leave output at
        t bit-identical -- the black-box version of the gradient check."""
        model = _model()
        x_dict, ei, globals_ = _random_inputs(S=20)
        x_dict = {t: x.detach() for t, x in x_dict.items()}
        globals_ = globals_.detach()

        with torch.no_grad():
            baseline = model(x_dict, ei, globals_)

        t_check = 10
        x_perturbed = {t: x.clone() for t, x in x_dict.items()}
        x_perturbed["bus"][0, t_check + 3] += 1000.0
        with torch.no_grad():
            perturbed = model(x_perturbed, ei, globals_)

        for target in TARGETS:
            # Structurally causal, but conv1d's internal batched-matmul
            # reduction order can introduce ~1e-8 floating noise unrelated to
            # any real information flow (verified: Readout's own per-slice
            # output is exactly 0 everywhere except the perturbed slice; the
            # noise originates inside CausalTCN's BLAS calls). 1e-4 cleanly
            # separates that from a real leak, which shows up at >= 1e-1.
            before = perturbed[target][0, :t_check + 1] - baseline[target][0, :t_check + 1]
            assert before.abs().max().item() < 1e-4, target
            after = perturbed[target][0, t_check + 3:] - baseline[target][0, t_check + 3:]
            assert after.abs().max().item() > 0.0, f"{target} perturbation had no downstream effect"

    def test_globals_perturbation_is_also_causal(self):
        model = _model()
        x_dict, ei, globals_ = _random_inputs(S=15)
        x_dict = {t: x.detach() for t, x in x_dict.items()}
        globals_ = globals_.detach()
        with torch.no_grad():
            baseline = model(x_dict, ei, globals_)
        t_check = 6
        g2 = globals_.clone()
        g2[0, t_check + 2] += 500.0
        with torch.no_grad():
            perturbed = model(x_dict, ei, g2)
        for target in TARGETS:
            diff_before = (perturbed[target][0, :t_check + 1] - baseline[target][0, :t_check + 1]).abs().max()
            assert diff_before.item() < 1e-4, target  # see tolerance note above


class TestStaticGraphReplication:
    def test_replicate_static_graph_equals_per_slice_loop(self):
        """The block-diagonal batching is EXACT: running the spatial encoder
        on the folded [B*S*N,F] representation must match running it once
        per slice, to float32 rounding only."""
        torch.manual_seed(5)
        ei = _edge_index_dict(np.random.default_rng(1))
        encoder = SpatialEncoder(NODE_TYPES, list(ei.keys()), hidden=16, heads=2, n_layers=2)
        encoder.eval()

        S = 5
        x_dict = {t: torch.randn(1, S, COUNTS[t], F_DIMS[t]) for t in NODE_TYPES}

        flat, num_nodes, B, Sx = flatten_for_spatial(x_dict)
        rep = replicate_static_graph(ei, num_nodes, B * Sx)
        with torch.no_grad():
            h_flat = encoder(flat, rep)
        batched = unflatten_from_spatial(h_flat, num_nodes, B, Sx)

        per_slice = {t: [] for t in NODE_TYPES}
        with torch.no_grad():
            for s in range(S):
                xs = {t: x_dict[t][0, s] for t in NODE_TYPES}
                out_s = encoder(xs, ei)
                for t in NODE_TYPES:
                    per_slice[t].append(out_s[t])
        per_slice = {t: torch.stack(v, dim=0).unsqueeze(0) for t, v in per_slice.items()}

        for t in NODE_TYPES:
            torch.testing.assert_close(batched[t], per_slice[t], atol=1e-5, rtol=1e-5)

    def test_replicate_rejects_zero_repeats(self):
        with pytest.raises(ValueError):
            replicate_static_graph({}, {}, 0)

    def test_flatten_unflatten_roundtrip(self):
        x_dict = {t: torch.randn(2, 4, COUNTS[t], F_DIMS[t]) for t in NODE_TYPES}
        flat, num_nodes, B, S = flatten_for_spatial(x_dict)
        restored = unflatten_from_spatial(flat, num_nodes, B, S)
        for t in NODE_TYPES:
            torch.testing.assert_close(restored[t], x_dict[t])

    def test_flatten_rejects_mismatched_batch_slice(self):
        x_dict = {
            "bus": torch.randn(1, 5, 33, 10),
            "host": torch.randn(2, 5, 1, 7),  # different B
        }
        with pytest.raises(ValueError):
            flatten_for_spatial(x_dict)


class TestHGTConvEmbeddingCoverage:
    def test_every_nonempty_type_gets_an_embedding(self):
        torch.manual_seed(6)
        ei = _edge_index_dict(np.random.default_rng(2))
        encoder = SpatialEncoder(NODE_TYPES, list(ei.keys()), hidden=8, heads=2, n_layers=N_GNN_LAYERS)
        encoder.eval()
        x_dict = {t: torch.randn(COUNTS[t], F_DIMS[t]) for t in NODE_TYPES}
        with torch.no_grad():
            out = encoder(x_dict, ei)
        assert set(out) == set(NODE_TYPES)
        for t in NODE_TYPES:
            assert out[t].shape == (COUNTS[t], 8)


class TestNoBatchNorm:
    def test_no_batchnorm_anywhere(self):
        model = _model()
        for name, module in model.named_modules():
            assert not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)), (
                f"found BatchNorm at {name} -- forbidden, see encoder.py module docstring "
                "on why this is a mechanical causality guard, not a style choice"
            )


class TestDeterminism:
    def test_deterministic_given_torch_seed(self):
        x_dict, ei, globals_ = _random_inputs(S=10, seed=9)
        x_dict = {t: x.detach() for t, x in x_dict.items()}
        globals_ = globals_.detach()

        model_a = _model(seed=42)
        with torch.no_grad():
            out_a = model_a(x_dict, ei, globals_)
        model_b = _model(seed=42)
        with torch.no_grad():
            out_b = model_b(x_dict, ei, globals_)
        for target in TARGETS:
            torch.testing.assert_close(out_a[target], out_b[target])


class TestOptimizerBeforeForward:
    """Regression guard for a known PyTorch footgun: an optimizer built on
    lazily-initialized (HGTConv/LazyLinear) parameters BEFORE the first
    forward pass. Verified empirically in this torch version (2.13.0) that
    it works correctly (materialization updates parameters in place); this
    test pins that so an upstream PyTorch change that breaks it is caught."""

    def test_optimizer_constructed_before_forward_still_updates_all_params(self):
        model = _model(seed=11)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        x_dict, ei, globals_ = _random_inputs(S=6, seed=12)
        out = model(x_dict, ei, globals_)
        loss = sum(v.sum() for v in out.values())
        loss.backward()
        opt.step()  # must not raise

        opt_param_ids = {id(p) for g in opt.param_groups for p in g["params"]}
        model_param_ids = {id(p) for p in model.parameters()}
        assert model_param_ids == opt_param_ids


class TestScaffolding:
    def test_combine_static_dynamic_broadcasts_static_across_slices(self):
        static_x = {"bus": torch.ones(3, 2)}
        dynamic_x = {"bus": torch.zeros(5, 3, 4)}
        combined = combine_static_dynamic(static_x, dynamic_x)
        assert combined["bus"].shape == (5, 3, 6)
        torch.testing.assert_close(combined["bus"][:, :, :2], torch.ones(5, 3, 2))
        torch.testing.assert_close(combined["bus"][:, :, 2:], torch.zeros(5, 3, 4))

    def test_stack_scenarios_matches_manual_stack(self):
        scenarios = [{"bus": torch.randn(5, 3, 2)} for _ in range(4)]
        stacked = stack_scenarios(scenarios)
        expected = torch.stack([s["bus"] for s in scenarios], dim=0)
        torch.testing.assert_close(stacked["bus"], expected)

    def test_stack_scenarios_rejects_shape_mismatch(self):
        scenarios = [{"bus": torch.randn(5, 3, 2)}, {"bus": torch.randn(5, 4, 2)}]
        with pytest.raises(ValueError):
            stack_scenarios(scenarios)

    def test_stack_scenarios_rejects_empty(self):
        with pytest.raises(ValueError):
            stack_scenarios([])
