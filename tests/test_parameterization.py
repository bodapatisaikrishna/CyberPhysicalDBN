"""Tests for uniformization and CPT construction (Cerotti et al. Eq. 3, Tables 1-3)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.attack_graph.graph import PHYS_LOCAL_DER, PHYS_WIDE_AREA, SensorModel, build_attack_graph
from src.dbn.compiler import ANTERIOR, ULTERIOR, compile_to_2tbn
from src.dbn.parameterization import (
    analytic_error_rates,
    attach_cpds,
    build_attack_step_cpt,
    build_analytic_cpt,
    build_latch_cpt,
    build_latched_reaction_cpt,
    collect_uniformization_ttcs,
    compute_delta_t,
    compute_ps,
    precondition_parents,
)

# Paper experiment settings (Cerotti et al. Sec. IV).
PAPER_M = 1.0
PAPER_P_POS = 1e-4
PAPER_P_NEG = 1e-4


@pytest.fixture
def ag():
    return build_attack_graph()


def _column(cpd, evidence_states: tuple[int, ...]) -> np.ndarray:
    """Column of `cpd` for a given assignment, indexed by evidence order."""
    n_evidence = len(evidence_states)
    index = 0
    for state in evidence_states:
        index = index * 2 + state
    assert cpd.get_values().shape[1] == 2**n_evidence
    return cpd.get_values()[:, index]


class TestUniformization:
    def test_delta_t_and_ps_hand_computed(self):
        """Two concurrent steps, T_bar = {2, 4}, m = 1.

        By hand:
            sum(1/T_bar) = 1/2 + 1/4 = 3/4
            delta_t      = 1 / (1 * 3/4) = 4/3
            p_a          = (4/3) / 2 = 2/3
            p_b          = (4/3) / 4 = 1/3
        """
        ttcs = {"a": 2.0, "b": 4.0}
        delta_t = compute_delta_t(ttcs, m=1.0)

        assert delta_t == pytest.approx(4.0 / 3.0)
        assert compute_ps(2.0, delta_t) == pytest.approx(2.0 / 3.0)
        assert compute_ps(4.0, delta_t) == pytest.approx(1.0 / 3.0)

    def test_m_scales_delta_t_inversely(self):
        ttcs = {"a": 2.0, "b": 4.0}
        assert compute_delta_t(ttcs, m=2.0) == pytest.approx(
            compute_delta_t(ttcs, m=1.0) / 2.0
        )

    def test_delta_t_from_paper_ttcs(self, ag):
        """Full Table 3 TTC set at m = 1.

        By hand, summing 1/T_bar over the eleven timed steps:
            3 x (1 / (1/3))  = 9            [UnsecCred1, UnsecCred2, UnsecCred]
            2 x (1 / (1/2))  = 4            [ModAuthProc1, ModAuthProc]
            1/2 + 1/2 + 1/2  = 3/2          [MITM, ModifyProgram, Masquerade]
            1/15 + 1/40 + 1/50 = 67/600     [SpoofRepMsg, UnauthCommand, ModCtrlLogic]
        Total = 8767/600, so delta_t = 600/8767.
        """
        ttcs = collect_uniformization_ttcs(ag)
        assert len(ttcs) == 11

        delta_t = compute_delta_t(ttcs, m=PAPER_M)
        assert delta_t == pytest.approx(600.0 / 8767.0)

    def test_rejects_nonpositive_inputs(self):
        with pytest.raises(ValueError):
            compute_delta_t({"a": 2.0}, m=0.0)
        with pytest.raises(ValueError):
            compute_delta_t({"a": 0.0}, m=1.0)
        with pytest.raises(ValueError):
            compute_delta_t({}, m=1.0)
        with pytest.raises(ValueError):
            compute_ps(0.0, 1.0)


class TestTable1:
    """The attack-step CPT must reproduce Cerotti et al. Table 1 exactly."""

    def test_spoofrepmsg_matches_table_1(self, ag):
        delta_t = compute_delta_t(collect_uniformization_ttcs(ag), m=PAPER_M)
        p_srm = compute_ps(float(ag.nodes["SpoofRepMsg"]["ttc"]), delta_t)
        assert p_srm == pytest.approx(40.0 / 8767.0)

        cpd = build_attack_step_cpt("SpoofRepMsg", ["MITM"], p_srm, ag)

        assert cpd.variables == [
            ("SpoofRepMsg", ULTERIOR),
            ("MITM", ANTERIOR),
            ("SpoofRepMsg", ANTERIOR),
        ]

        # Columns ordered (MITM(t-1), SpoofRepMsg(t-1)); rows are
        # SpoofRepMsg(t) = 0 then 1. This is Table 1's eight rows.
        expected = np.array(
            [
                [1.0, 1.0, 1.0 - p_srm, 0.0],
                [0.0, 0.0, p_srm, 1.0],
            ]
        )
        np.testing.assert_allclose(cpd.get_values(), expected)

    def test_rows_1_to_4_precondition_unsatisfied(self, ag):
        """MITM inactive forces SpoofRepMsg to 0, even when it was active."""
        cpd = build_attack_step_cpt("SpoofRepMsg", ["MITM"], 0.25, ag)

        np.testing.assert_allclose(_column(cpd, (0, 0)), [1.0, 0.0])
        np.testing.assert_allclose(_column(cpd, (0, 1)), [1.0, 0.0])

    def test_rows_5_to_6_bernoulli(self, ag):
        cpd = build_attack_step_cpt("SpoofRepMsg", ["MITM"], 0.25, ag)
        np.testing.assert_allclose(_column(cpd, (1, 0)), [0.75, 0.25])

    def test_rows_7_to_8_persistence(self, ag):
        cpd = build_attack_step_cpt("SpoofRepMsg", ["MITM"], 0.25, ag)
        np.testing.assert_allclose(_column(cpd, (1, 1)), [0.0, 1.0])


class TestMultiParentGeneralization:
    """UnauthCommand is the real two-parent case: MITM and Masquerade."""

    def test_unauthcommand_has_two_parents(self, ag):
        assert precondition_parents(ag, "UnauthCommand") == ["MITM", "Masquerade"]

    def test_two_parent_cpt_structure(self, ag):
        p_s = 0.25
        parents = precondition_parents(ag, "UnauthCommand")
        cpd = build_attack_step_cpt("UnauthCommand", parents, p_s, ag)

        assert cpd.get_values().shape == (2, 8)

        for mitm, masq, self_prev in itertools.product([0, 1], repeat=3):
            column = _column(cpd, (mitm, masq, self_prev))
            precondition_met = bool(mitm or masq)

            if not precondition_met:
                # Precondition-false outranks self-persistence: forced to 0
                # even when the node was already active.
                expected_active = 0.0
            elif self_prev:
                expected_active = 1.0
            else:
                expected_active = p_s

            assert column[1] == pytest.approx(expected_active), (
                f"MITM={mitm} Masquerade={masq} self={self_prev}"
            )
            assert column[0] == pytest.approx(1.0 - expected_active)

    def test_root_node_precondition_vacuously_satisfied(self, ag):
        """Roots have no parents and must still be able to activate."""
        assert precondition_parents(ag, "UnsecCred1") == []

        cpd = build_attack_step_cpt("UnsecCred1", [], 0.25, ag)

        assert cpd.get_values().shape == (2, 2)
        np.testing.assert_allclose(_column(cpd, (0,)), [0.75, 0.25])
        np.testing.assert_allclose(_column(cpd, (1,)), [0.0, 1.0])


class TestCompiledModel:
    def test_canonical_form_no_anterior_intra_slice_arcs(self, ag):
        dbn = compile_to_2tbn(ag)
        anterior_intra = [
            (u, v) for u, v in dbn.edges() if u[1] == ANTERIOR and v[1] == ANTERIOR
        ]
        assert anterior_intra == []

    def test_analytics_are_untimed(self, ag):
        """No self-loop and no anterior-layer copy for analytic nodes."""
        dbn = compile_to_2tbn(ag)
        analytics = [
            n for n, d in ag.nodes(data=True) if d["node_type"] == "analytic"
        ]
        for name in analytics:
            assert not ag.has_edge(name, name)
            assert (name, ANTERIOR) not in dbn.nodes()

    def test_attack_steps_have_temporal_arc(self, ag):
        """The 11 attack steps persist; nothing else does.

        Reactions and gates carry TTC=0 in Table 3 -- they resolve within a
        slice rather than racing to complete -- so they get no temporal arc.
        """
        dbn = compile_to_2tbn(ag)
        persistent = [n for n, d in ag.nodes(data=True) if d["self_loop"]]
        assert len(persistent) == 11
        assert all(ag.nodes[n]["node_type"] == "attack_step" for n in persistent)
        assert all(ag.nodes[n]["gate"] is None for n in persistent)
        for name in persistent:
            assert dbn.has_edge((name, ANTERIOR), (name, ULTERIOR))

    def test_reactions_are_untimed(self, ag):
        for name in ["CorrReact", "WrongLogicExec"]:
            assert not ag.nodes[name]["self_loop"]
            assert not ag.has_edge(name, name)

    def test_mitm_depends_on_credaccess_within_slice(self, ag):
        """CredAccess is untimed, so MITM sees it at t, not t-1."""
        dbn = compile_to_2tbn(ag)
        assert dbn.has_edge(("CredAccess", ULTERIOR), ("MITM", ULTERIOR))
        assert not dbn.has_edge(("CredAccess", ANTERIOR), ("MITM", ULTERIOR))

    def test_every_cpd_normalized(self, ag):
        """Every generated CPT's columns sum to 1."""
        dbn = attach_cpds(
            compile_to_2tbn(ag), ag, m=PAPER_M, p_pos=PAPER_P_POS, p_neg=PAPER_P_NEG
        )
        cpds = dbn.get_cpds()
        assert len(cpds) == ag.number_of_nodes()

        for cpd in cpds:
            column_sums = cpd.get_values().sum(axis=0)
            np.testing.assert_allclose(
                column_sums, np.ones_like(column_sums), err_msg=str(cpd.variable)
            )

    def test_gate_cpds_are_deterministic(self, ag):
        dbn = attach_cpds(
            compile_to_2tbn(ag), ag, m=PAPER_M, p_pos=PAPER_P_POS, p_neg=PAPER_P_NEG
        )

        cred = dbn.get_cpds(("CredAccess", ULTERIOR))
        # AND over two parents: only the all-active column activates.
        np.testing.assert_allclose(cred.get_values()[1], [0.0, 0.0, 0.0, 1.0])

        unstable = dbn.get_cpds(("UnstablePS", ULTERIOR))
        # OR over three parents: only the all-inactive column stays inactive.
        np.testing.assert_allclose(
            unstable.get_values()[1], [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )

    def test_analytic_cpd_matches_table_2(self, ag):
        dbn = attach_cpds(
            compile_to_2tbn(ag), ag, m=PAPER_M, p_pos=PAPER_P_POS, p_neg=PAPER_P_NEG
        )
        cpd = dbn.get_cpds(("FileAccess", ULTERIOR))
        np.testing.assert_allclose(
            cpd.get_values(),
            [[1.0 - PAPER_P_POS, PAPER_P_NEG], [PAPER_P_POS, 1.0 - PAPER_P_NEG]],
        )


class TestLatchedReactions:
    """One-shot latched reactions (graph.build_attack_graph reaction_mode).

    The paper's own figures disagree on reaction semantics: Fig. 5a's flat-0.7
    plateau requires no memory, while Fig. 6c's non-converging KL requires
    memory. The latched reading satisfies both -- it plateaus at exactly the
    fixed success probability while carrying state across slices. See
    LAB_NOTEBOOK.md 2026-07-31.
    """

    @pytest.fixture
    def latched_ag(self):
        return build_attack_graph(reaction_mode="latched")

    def test_adds_one_latch_per_reaction(self, latched_ag):
        latches = [
            n for n, d in latched_ag.nodes(data=True) if d["node_type"] == "latch"
        ]
        assert sorted(latches) == ["CorrReact__Seen", "WrongLogicExec__Seen"]
        assert latched_ag.number_of_nodes() == 25  # 23 + 2 latches

    def test_reactions_become_persistent(self, latched_ag):
        for reaction in ["CorrReact", "WrongLogicExec"]:
            assert latched_ag.nodes[reaction]["self_loop"] is True
            assert latched_ag.nodes[reaction]["latch"] == f"{reaction}__Seen"

    def test_memoryless_mode_is_unchanged_default(self):
        default_ag = build_attack_graph()
        assert default_ag.number_of_nodes() == 23
        assert not any(
            d["node_type"] == "latch" for _, d in default_ag.nodes(data=True)
        )
        for reaction in ["CorrReact", "WrongLogicExec"]:
            assert default_ag.nodes[reaction]["self_loop"] is False

    def test_latch_cpt_is_deterministic_or(self, latched_ag):
        cpd = build_latch_cpt("CorrReact__Seen", "SpoofRepMsg", latched_ag)
        # columns ordered (SpoofRepMsg(t-1), self(t-1))
        np.testing.assert_allclose(cpd.get_values()[1], [0.0, 1.0, 1.0, 1.0])

    def test_latched_reaction_fires_only_on_first_chance(self, latched_ag):
        p = 0.7
        cpd = build_latched_reaction_cpt(
            "CorrReact", "SpoofRepMsg", "CorrReact__Seen", p, latched_ag
        )
        active = cpd.get_values()[1]

        for i, (latch, precondition, self_prev) in enumerate(
            itertools.product([0, 1], repeat=3)
        ):
            if self_prev:
                expected = 1.0  # persistence
            elif not precondition:
                expected = 0.0  # precondition never held
            elif latch:
                expected = 0.0  # the one chance was used and failed
            else:
                expected = p  # first and only chance
            assert active[i] == pytest.approx(expected), (
                f"latch={latch} precondition={precondition} self={self_prev}"
            )

        np.testing.assert_allclose(cpd.get_values().sum(axis=0), np.ones(8))

    def test_all_latched_cpds_normalized(self, latched_ag):
        dbn = attach_cpds(
            compile_to_2tbn(latched_ag),
            latched_ag,
            m=PAPER_M,
            p_pos=PAPER_P_POS,
            p_neg=PAPER_P_NEG,
        )
        cpds = dbn.get_cpds()
        assert len(cpds) == latched_ag.number_of_nodes()
        for cpd in cpds:
            sums = cpd.get_values().sum(axis=0)
            np.testing.assert_allclose(sums, np.ones_like(sums), err_msg=str(cpd.variable))


class TestPhysicalEvidenceNodes:
    """Session 4 (LAB_NOTEBOOK.md 2026-08-01): PhysLocalDER/PhysWideArea are
    ordinary analytics distinguished only by observable_kind and an optional
    SensorModel override, threaded through analytic_error_rates. The whole
    point is that the 8 existing cyber analytics must not move by a single
    bit when these are added."""

    @pytest.fixture
    def physical_ag(self):
        return build_attack_graph(physical_evidence=True)

    def test_rejects_sensor_models_without_physical_evidence(self):
        with pytest.raises(ValueError):
            build_attack_graph(
                sensor_models={PHYS_LOCAL_DER: SensorModel(0.1, 0.1, "test")}
            )

    def test_sensor_model_requires_nonempty_source(self):
        with pytest.raises(ValueError):
            SensorModel(0.1, 0.1, "")

    def test_node_counts_across_reaction_mode_and_physical_evidence_axes(self):
        assert build_attack_graph().number_of_nodes() == 23
        assert build_attack_graph(physical_evidence=True).number_of_nodes() == 25
        assert build_attack_graph(reaction_mode="latched").number_of_nodes() == 25
        assert (
            build_attack_graph(reaction_mode="latched", physical_evidence=True).number_of_nodes()
            == 27
        )

    def test_physical_nodes_have_declared_single_parent_edges(self, physical_ag):
        assert list(physical_ag.predecessors(PHYS_LOCAL_DER)) == ["WrongLogicExec"]
        assert list(physical_ag.predecessors(PHYS_WIDE_AREA)) == ["UnstablePS"]
        for node in (PHYS_LOCAL_DER, PHYS_WIDE_AREA):
            assert physical_ag.nodes[node]["node_type"] == "analytic"
            assert physical_ag.nodes[node]["observable_kind"] == "physical"
            assert physical_ag.nodes[node]["self_loop"] is False
            assert physical_ag.nodes[node]["mitre_technique"] is None

    def test_analytic_error_rates_no_override_passes_through_global(self, physical_ag):
        """No SensorModel attached -> the global rates flow through unchanged."""
        assert analytic_error_rates(physical_ag, "FileAccess", 1e-4, 1e-4) == (1e-4, 1e-4)
        assert analytic_error_rates(physical_ag, PHYS_LOCAL_DER, 1e-4, 1e-4) == (1e-4, 1e-4)

    def test_analytic_error_rates_override_takes_precedence(self):
        ag = build_attack_graph(
            physical_evidence=True,
            sensor_models={PHYS_WIDE_AREA: SensorModel(0.35, 0.4, "hand-written test override")},
        )
        assert analytic_error_rates(ag, PHYS_WIDE_AREA, 1e-4, 1e-4) == (0.35, 0.4)
        # The other physical node and all cyber analytics are unaffected.
        assert analytic_error_rates(ag, PHYS_LOCAL_DER, 1e-4, 1e-4) == (1e-4, 1e-4)
        assert analytic_error_rates(ag, "FileAccess", 1e-4, 1e-4) == (1e-4, 1e-4)

    def test_overridden_sensor_model_produces_hand_written_cpt(self):
        ag = build_attack_graph(
            physical_evidence=True,
            sensor_models={PHYS_WIDE_AREA: SensorModel(0.35, 0.4, "hand-written test override")},
        )
        cpd = build_analytic_cpt(PHYS_WIDE_AREA, "UnstablePS", 1e-4, 1e-4, ag)
        # P(PhysWideArea=1 | UnstablePS=0) = p_pos = 0.35
        # P(PhysWideArea=1 | UnstablePS=1) = 1 - p_neg = 0.6
        np.testing.assert_allclose(cpd.get_values(), [[0.65, 0.4], [0.35, 0.6]])

    def test_existing_eight_analytics_byte_identical_with_physical_evidence_added(self):
        """The regression this module exists to prevent: adding the two
        physical nodes must not perturb any of the 8 Session-1 analytic CPDs."""
        cyber_ag = build_attack_graph()
        physical_ag = build_attack_graph(physical_evidence=True)
        cyber_names = [
            n for n, d in cyber_ag.nodes(data=True) if d["node_type"] == "analytic"
        ]
        assert len(cyber_names) == 8

        dbn_cyber = attach_cpds(
            compile_to_2tbn(cyber_ag), cyber_ag, m=PAPER_M, p_pos=PAPER_P_POS, p_neg=PAPER_P_NEG
        )
        dbn_physical = attach_cpds(
            compile_to_2tbn(physical_ag),
            physical_ag,
            m=PAPER_M,
            p_pos=PAPER_P_POS,
            p_neg=PAPER_P_NEG,
        )
        for name in cyber_names:
            a = dbn_cyber.get_cpds((name, ULTERIOR)).get_values()
            b = dbn_physical.get_cpds((name, ULTERIOR)).get_values()
            np.testing.assert_array_equal(a, b, err_msg=name)

    def test_total_cpd_count_with_physical_evidence(self, physical_ag):
        dbn = attach_cpds(
            compile_to_2tbn(physical_ag),
            physical_ag,
            m=PAPER_M,
            p_pos=PAPER_P_POS,
            p_neg=PAPER_P_NEG,
        )
        assert len(dbn.get_cpds()) == 25
