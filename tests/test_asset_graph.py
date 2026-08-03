"""Tests for src/perception/asset_graph.py.

Uses a real (short-horizon) twin run to obtain `observed_endpoints`, matching
how experiments/exp05_perception.py's stage 0 will actually call this --
the fabrication guard is tested against real telemetry, not a hand-built log.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import torch
from torch_geometric.nn import HGTConv

from src.attack_graph.graph import build_attack_graph
from src.perception.asset_graph import (
    DECLARED_EDGE_TYPES,
    DECLARED_NODE_TYPES,
    CyberAsset,
    CyberOverlayConfig,
    build_asset_graph,
    filtered_input,
    hops_between,
    nonempty_metadata,
    observed_endpoints,
)
from src.twin.comms import DeliveryRecord, Message, MessageType, PayloadCategory
from src.twin.grid import GridConfig, GridModel
from src.twin.runner import TwinConfig, TwinRunner

N_GNN_LAYERS = 3  # pinned load-bearing constant; must match encoder.py


@pytest.fixture(scope="module")
def twin_log() -> tuple[DeliveryRecord, ...]:
    ag = build_attack_graph()
    config = TwinConfig(horizon_time_units=200.0)
    trace = TwinRunner(ag, config, np.random.SeedSequence(1)).run()
    return trace.messages


@pytest.fixture(scope="module")
def obs(twin_log) -> frozenset[str]:
    return observed_endpoints(twin_log)


@pytest.fixture(scope="module")
def grid() -> GridModel:
    return GridModel(GridConfig())


def _overlay(controls_der: bool = True) -> CyberOverlayConfig:
    return CyberOverlayConfig(
        assets=(
            CyberAsset("ControlCentre", "host", "ControlCentre"),
            CyberAsset("IED_17", "IED", "DER_17", controls=("DER_17",) if controls_der else ()),
            CyberAsset("IED_32", "IED", "DER_32", controls=("DER_32",) if controls_der else ()),
        ),
        source="test fixture",
    )


class TestCounts:
    def test_bus_count(self, grid, obs):
        ag = build_asset_graph(grid, _overlay(), observed_endpoints_set=obs)
        assert ag.counts["bus"] == 33

    def test_line_count_and_in_service_split(self, grid, obs):
        ag = build_asset_graph(grid, _overlay(), observed_endpoints_set=obs)
        assert ag.counts["line"] == 37
        in_service = int(ag.data["line"].x[:, 4].sum().item())
        assert in_service == 32

    def test_transformer_type_is_empty_on_case33bw(self, grid, obs):
        ag = build_asset_graph(grid, _overlay(), observed_endpoints_set=obs)
        assert ag.counts["transformer"] == 0
        assert "transformer" in ag.empty_node_types
        assert ag.data["transformer"].x.shape[0] == 0

    def test_rtu_and_relay_are_empty(self, grid, obs):
        ag = build_asset_graph(grid, _overlay(), observed_endpoints_set=obs)
        assert ag.counts["RTU"] == 0
        assert ag.counts["relay"] == 0
        assert "RTU" in ag.empty_node_types
        assert "relay" in ag.empty_node_types

    def test_der_nodes_derived_from_gridmodel_not_hardcoded(self, obs):
        """Vary n_der and confirm DER node count and identity track
        GridModel.der_ids exactly -- never a literal ["DER_17","DER_32"]."""
        grid3 = GridModel(GridConfig(n_der=3))
        overlay = CyberOverlayConfig(
            assets=tuple(
                CyberAsset(f"IED_{b}", "IED", der_id, controls=(der_id,))
                for der_id, b in zip(grid3.der_ids, grid3.der_buses)
            )
            + (CyberAsset("ControlCentre", "host", "ControlCentre"),),
            source="n_der=3 test",
        )
        obs3 = obs | frozenset(grid3.der_ids) | frozenset({"ControlCentre"})
        ag = build_asset_graph(grid3, overlay, observed_endpoints_set=obs3)
        assert ag.counts["DER"] == 3
        assert set(ag.node_index["DER"]) == set(grid3.der_ids)

    def test_declared_types_are_exactly_stated(self):
        assert len(DECLARED_NODE_TYPES) == 8
        assert len(DECLARED_EDGE_TYPES) == 3


class TestFabricationGuard:
    def test_every_cyber_asset_endpoint_must_appear_in_twin_log(self, grid, obs):
        bad = CyberOverlayConfig(
            assets=(CyberAsset("Ghost", "host", "NeverSeenEndpoint"),),
            source="test",
        )
        with pytest.raises(ValueError, match="never appeared"):
            build_asset_graph(grid, bad, observed_endpoints_set=obs)

    def test_real_overlay_endpoints_all_pass(self, grid, obs):
        """The positive case: ControlCentre/DER_17/DER_32 all genuinely
        appear in a real twin run's comms log."""
        ag = build_asset_graph(grid, _overlay(), observed_endpoints_set=obs)
        assert ag.counts["host"] == 1
        assert ag.counts["IED"] == 2

    def test_host_may_not_directly_control_a_der(self, grid, obs):
        """control_authority is IED-only; a host declaring `controls` is
        rejected structurally, not just by convention -- see the module
        docstring on why a direct host->DER edge would break the 3-hop
        cyber-shortcut claim."""
        bad = CyberOverlayConfig(
            assets=(CyberAsset("ControlCentre", "host", "ControlCentre", controls=("DER_17",)),),
            source="test",
        )
        with pytest.raises(ValueError, match="IED"):
            build_asset_graph(grid, bad, observed_endpoints_set=obs)

    def test_controls_unknown_der_rejected(self, grid, obs):
        bad = CyberOverlayConfig(
            assets=(CyberAsset("IED_X", "IED", "DER_17", controls=("DER_999",)),),
            source="test",
        )
        with pytest.raises(ValueError, match="DER_999"):
            build_asset_graph(grid, bad, observed_endpoints_set=obs)


class TestEdgesAndMetadata:
    @pytest.fixture
    def ag(self, grid, obs):
        return build_asset_graph(grid, _overlay(), observed_endpoints_set=obs)

    def test_declared_edge_types_are_exactly_three(self):
        assert DECLARED_EDGE_TYPES == (
            "electrical_coupling", "network_reachability", "control_authority",
        )

    def test_every_nonempty_node_type_has_incoming_edges(self, ag):
        """The verified HGTConv constraint: a type with only outgoing edges
        is silently dropped from HGTConv's output. Structural, not a forward-
        pass smoke test."""
        nt, _ = nonempty_metadata(ag.data)
        dst_types = {et[2] for et in ag.data.edge_types}
        for t in nt:
            assert t in dst_types, f"{t} has no incoming edges after ToUndirected"

    def test_metadata_excludes_empty_node_types(self, ag):
        nt, et = nonempty_metadata(ag.data)
        assert "transformer" not in nt
        assert "RTU" not in nt
        assert "relay" not in nt
        for src, _, dst in et:
            assert src in nt and dst in nt

    def test_electrical_coupling_matches_networkx_adjacency_of_in_service_lines(self, ag, grid):
        """ag.graph is the module's own in-service adjacency; the bus<->bus
        electrical_coupling edge count must equal 2x its edge count (directed
        pairs, undirected graph)."""
        assert ag.edge_counts[("bus", "electrical_coupling", "bus")] == ag.graph.number_of_edges()

    def test_control_authority_targets_only_der_nodes(self, ag):
        for (src, rel, dst), count in ag.edge_counts.items():
            if rel == "control_authority":
                assert dst == "DER"
                assert src == "IED"
                assert count > 0

    def test_control_authority_absent_when_no_controls_declared(self, grid, obs):
        ag = build_asset_graph(grid, _overlay(controls_der=False), observed_endpoints_set=obs)
        assert ag.edge_counts[("IED", "control_authority", "DER")] == 0

    def test_hops_from_der_bus_to_host_equals_n_gnn_layers(self, ag, grid):
        """The load-bearing fact this module's architecture depends on: the
        cyber overlay makes a DER's bus exactly N_GNN_LAYERS hops from host,
        independent of the 20-hop electrical distance. If this test starts
        failing, encoder.py's n_layers must change to match, not the other
        way around."""
        for bus, der_id in zip(grid.der_buses, grid.der_ids):
            h = hops_between(
                ag.data,
                ("bus", ag.node_index["bus"][str(bus)]),
                ("host", ag.node_index["host"]["ControlCentre"]),
            )
            assert h == N_GNN_LAYERS, f"{der_id} bus {bus}: {h} hops, expected {N_GNN_LAYERS}"

    def test_electrical_distance_between_der_buses_is_20_hops(self, ag, grid):
        """Control for the above: proves the cyber shortcut is doing real
        work, not just restating a short electrical path."""
        d = nx.shortest_path_length(ag.graph, grid.der_buses[0], grid.der_buses[1])
        assert d == 20

    def test_in_service_line_graph_is_a_tree(self, ag):
        assert nx.is_tree(ag.graph)

    def test_topology_identical_across_repeated_builds(self, grid, obs):
        """No hidden randomness in the static structure: two independent
        builds from the same inputs must agree exactly on edges."""
        a = build_asset_graph(grid, _overlay(), observed_endpoints_set=obs)
        b = build_asset_graph(grid, _overlay(), observed_endpoints_set=obs)
        assert a.edge_counts == b.edge_counts
        for et in a.data.edge_types:
            torch.testing.assert_close(a.data[et].edge_index, b.data[et].edge_index)


class TestHGTConvCompatibility:
    """The actual downstream contract encoder.py depends on."""

    def test_hgtconv_runs_on_filtered_input(self, grid, obs):
        ag = build_asset_graph(grid, _overlay(), observed_endpoints_set=obs)
        nt, et = nonempty_metadata(ag.data)
        conv = HGTConv(-1, 8, (nt, et), heads=2)
        x_dict, edge_index_dict = filtered_input(ag.data, nt, et)
        out = conv(x_dict, edge_index_dict)
        assert set(out) == set(nt)
        assert out["host"].shape == (1, 8)
        assert out["bus"].shape == (33, 8)

    def test_hgtconv_raises_on_unfiltered_input(self, grid, obs):
        """Documents the footgun `filtered_input` exists to avoid -- passing
        the raw x_dict/edge_index_dict (with empty types included) breaks."""
        ag = build_asset_graph(grid, _overlay(), observed_endpoints_set=obs)
        nt, et = nonempty_metadata(ag.data)
        conv = HGTConv(-1, 8, (nt, et), heads=2)
        with pytest.raises((KeyError, IndexError)):
            conv(ag.data.x_dict, ag.data.edge_index_dict)
