"""Tests for src/attack_graph/family.py (Session 8, claim C2)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from src.attack_graph.graph import technique_table3_ttc
from src.dbn.compiler import compile_to_2tbn
from src.dbn.inference import DBNInference, InferenceConfig, _interface_nodes, fully_factorized_clustering
from src.dbn.parameterization import attach_cpds
from src.attack_graph.family import (
    FamilyGeneratorConfig,
    FamilyGraphSpec,
    family_graph_rows,
    generate_family,
)
from src.parameterization.amortized import known_techniques


def _small_config(**overrides) -> FamilyGeneratorConfig:
    base = dict(
        n_graphs=6, n_train=3, n_val=1, n_test=2,
        depth_range=(2, 3), branching_factor_range=(1, 2), n_root_processes_range=(1, 2),
        analytic_coverage_range=(0.3, 1.0), asset_context_range=(0.0, 1.0),
        defensive_posture_range=(0.5, 4.0), attacker_capability_range=(0.25, 4.0),
    )
    base.update(overrides)
    return FamilyGeneratorConfig(**base)


class TestFamilyGeneratorConfig:
    def test_rejects_split_mismatch(self):
        with pytest.raises(ValueError, match="n_train"):
            _small_config(n_train=99)


class TestGenerationDeterminism:
    def test_same_seed_identical_graphs(self):
        config = _small_config()
        g1 = generate_family(config, np.random.SeedSequence(42))
        g2 = generate_family(config, np.random.SeedSequence(42))
        assert [fg.graph_id for fg in g1] == [fg.graph_id for fg in g2]
        for a, b in zip(g1, g2):
            assert nx.utils.graphs_equal(a.ag, b.ag)
            assert a.spec == b.spec

    def test_different_seed_can_diverge(self):
        config = _small_config()
        g1 = generate_family(config, np.random.SeedSequence(1))
        g2 = generate_family(config, np.random.SeedSequence(2))
        specs1 = [(fg.spec.depth, fg.spec.branching_factor, fg.spec.n_root_processes) for fg in g1]
        specs2 = [(fg.spec.depth, fg.spec.branching_factor, fg.spec.n_root_processes) for fg in g2]
        assert specs1 != specs2


class TestPerformanceCap:
    """Verified pgmpy VariableElimination blowup (module docstring): never
    let n_root_processes>1 and branching_factor>1 co-occur."""

    def test_never_both_axes_exceed_one(self):
        config = _small_config(
            n_graphs=30, n_train=15, n_val=5, n_test=10,
            depth_range=(2, 4), branching_factor_range=(1, 3), n_root_processes_range=(1, 3),
        )
        graphs = generate_family(config, np.random.SeedSequence(21))
        for fg in graphs:
            assert not (fg.spec.n_root_processes > 1 and fg.spec.branching_factor > 1), (
                fg.graph_id, fg.spec.n_root_processes, fg.spec.branching_factor
            )


class TestSplitDisjointness:
    def test_exact_sizes_and_no_overlap(self):
        config = _small_config()
        graphs = generate_family(config, np.random.SeedSequence(0))
        by_split = {"train": [], "val": [], "test": []}
        for fg in graphs:
            by_split[fg.split].append(fg.graph_id)
        assert len(by_split["train"]) == 3
        assert len(by_split["val"]) == 1
        assert len(by_split["test"]) == 2
        all_ids = by_split["train"] + by_split["val"] + by_split["test"]
        assert len(all_ids) == len(set(all_ids))


class TestTechniqueVocabularyContainment:
    def test_every_technique_is_known(self):
        config = _small_config()
        graphs = generate_family(config, np.random.SeedSequence(3))
        known = set(known_techniques())
        for fg in graphs:
            for node, data in fg.ag.nodes(data=True):
                if data.get("ttc") is not None:
                    assert data["mitre_technique"] in known


class TestGroundTruthFormula:
    def test_exact_formula_per_node(self):
        config = _small_config()
        graphs = generate_family(config, np.random.SeedSequence(5))
        table3 = technique_table3_ttc()
        for fg in graphs:
            for node, data in fg.ag.nodes(data=True):
                if data.get("ttc") is None:
                    continue
                expected = table3[data["mitre_technique"]] * data["defensive_posture"] / data["attacker_capability"]
                assert data["ttc"] == pytest.approx(expected)


class TestStructuralValidity:
    """Load-bearing: the first time any graph besides the one paper graph
    goes through the real compiler/parameterization/inference stack."""

    def test_every_graph_compiles_and_runs(self):
        """Ranges match configs/transfer_c2.yaml's real ones -- verified
        tractable (14s/20 graphs) only AFTER the n_root_processes>1-and-
        branching_factor>1 cap was added (see family.py's module
        docstring for the measured pgmpy VariableElimination blowup this
        cap avoids)."""
        config = _small_config(
            n_graphs=20, n_train=10, n_val=3, n_test=7,
            depth_range=(2, 5), branching_factor_range=(1, 3), n_root_processes_range=(1, 3),
        )
        graphs = generate_family(config, np.random.SeedSequence(7))
        for fg in graphs:
            dbn = compile_to_2tbn(fg.ag)
            dbn = attach_cpds(dbn, fg.ag, m=1.0, p_pos=1e-4, p_neg=1e-4)
            interface = _interface_nodes(fg.ag)
            clustering = fully_factorized_clustering(interface)
            engine = DBNInference(fg.ag, InferenceConfig(clustering=clustering, m=1.0, p_pos=1e-4, p_neg=1e-4))
            trajectory = engine.run({}, T=2)
            assert len(trajectory.marginals) == 2
            for marginals in trajectory.marginals:
                for node in fg.ag.nodes():
                    assert node in marginals
                    assert 0.0 <= marginals[node] <= 1.0

    def test_root_nodes_have_no_precondition_parents(self):
        """Sanity on the generator's own topology: a depth-1 sub-chain's
        single node is a genuine root (no incoming precondition edge other
        than its own self-loop)."""
        config = _small_config(depth_range=(1, 1))
        graphs = generate_family(config, np.random.SeedSequence(9))
        for fg in graphs:
            for node, data in fg.ag.nodes(data=True):
                if data["node_type"] != "attack_step" or data.get("ttc") is None:
                    continue
                if "_s0" not in node:
                    continue
                parents = [p for p in fg.ag.predecessors(node) if p != node]
                assert parents == []


class TestFamilyGraphRows:
    def test_columns_and_row_count(self):
        config = _small_config()
        graphs = generate_family(config, np.random.SeedSequence(11))
        df = family_graph_rows(graphs)
        assert set(df.columns) >= {
            "graph_id", "split", "node", "technique", "asset_context",
            "defensive_posture", "attacker_capability", "true_ttc",
        }
        expected_n_rows = sum(
            1 for fg in graphs for _, d in fg.ag.nodes(data=True) if d.get("ttc") is not None
        )
        assert len(df) == expected_n_rows

    def test_split_column_matches_graph_split(self):
        config = _small_config()
        graphs = generate_family(config, np.random.SeedSequence(13))
        df = family_graph_rows(graphs)
        split_by_graph = {fg.graph_id: fg.split for fg in graphs}
        for _, row in df.iterrows():
            assert row["split"] == split_by_graph[row["graph_id"]]
