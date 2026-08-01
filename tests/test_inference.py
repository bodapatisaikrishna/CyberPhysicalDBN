"""Cross-checks of DBNInference against pgmpy VariableElimination run directly.

These are the actual correctness gate for src/dbn/inference.py. Both tests use
the *real* CPT-building code (compile_to_2tbn, attach_cpds) restricted to a
small subgraph, with a hand-chosen non-degenerate anterior prior injected
directly (bypassing DBNInference.initial_belief()'s point mass) -- a
degenerate all-zero prior would make FF and EX (and a correct vs. a buggy
independence assumption) agree by accident on the first step, which is exactly
the failure mode a design review caught in this project's first draft: MITM's
only parent, CredAccess, is intra-slice (see
test_parameterization.py::test_mitm_depends_on_credaccess_within_slice), so
MITM is correlated with UnsecCred/ModAuthProc through a same-slice gate, not
only through history. A hand-rolled implementation that assumed per-interface-
node independence given the anterior layer would compute this case wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

from src.attack_graph.graph import build_attack_graph
from src.dbn.compiler import ANTERIOR, ULTERIOR, compile_to_2tbn
from src.dbn.parameterization import attach_cpds, compute_delta_t, compute_ps
from src.dbn.inference import DBNInference, InferenceConfig

M = 1.0
P_POS = 1e-4
P_NEG = 1e-4
# Cerotti et al. Table 5, m=1 (166.13 s), in Table 3 time units (10 min).
DELTA_T = 166.13 / 600


@pytest.fixture
def ag():
    return build_attack_graph()


def _ve_marginal(dbn, node: str, evidence: dict | None = None) -> float:
    infer = VariableElimination(dbn)
    result = infer.query(
        [(node, ULTERIOR)], evidence=evidence or {}, joint=False, show_progress=False
    )
    return float(result[(node, ULTERIOR)].values[1])


class TestBaselineChain:
    """UnsecCred1 -> UnsecCred2: two persistent attack steps, pure inter-slice.

    Both nodes self-loop, so this is the plain temporal case with no gate and
    no same-slice mediator -- the baseline that isolates "does the per-step
    machinery work at all" from the harder CredAccess/MITM structure below.
    Isolated as its own subgraph, which preserves both nodes' semantics
    exactly: UnsecCred1 is a root in the full graph too, and UnsecCred2's only
    precondition there is UnsecCred1.
    """

    @pytest.fixture
    def sub(self, ag):
        return ag.subgraph(["UnsecCred1", "UnsecCred2"]).copy()

    def test_matches_direct_pgmpy(self, sub):
        dbn = attach_cpds(compile_to_2tbn(sub), sub, m=M, p_pos=P_POS, p_neg=P_NEG)
        dbn.add_cpds(TabularCPD(("UnsecCred1", ANTERIOR), 2, [[0.7], [0.3]]))
        dbn.add_cpds(TabularCPD(("UnsecCred2", ANTERIOR), 2, [[0.6], [0.4]]))
        expected = _ve_marginal(dbn, "UnsecCred2")

        config = InferenceConfig(
            clustering=frozenset(
                {frozenset(["UnsecCred1"]), frozenset(["UnsecCred2"])}
            ),
            m=M,
            p_pos=P_POS,
            p_neg=P_NEG,
        )
        engine = DBNInference(sub, config)
        belief = {
            frozenset(["UnsecCred1"]): np.array([0.7, 0.3]),
            frozenset(["UnsecCred2"]): np.array([0.6, 0.4]),
        }
        result = engine.step(belief, {})

        assert result.marginals["UnsecCred2"] == pytest.approx(expected)


class TestReactionIsMemoryless:
    """ModCtrlLogic -> WrongLogicExec: reaction, intra-slice, no persistence.

    Reactions carry TTC=0 in Table 3, so they resolve within the slice rather
    than racing to complete. WrongLogicExec therefore has no anterior copy and
    no temporal arc: its value is a fresh Bernoulli(0.8) draw gated on its
    precondition's *current* value, not a function of its own past. This is a
    documented deviation from Fig. 2 (which draws self-loops on the two
    reaction nodes) -- see LAB_NOTEBOOK.md 2026-07-31 for the numerical
    evidence that decided it.
    """

    @pytest.fixture
    def sub(self, ag):
        return ag.subgraph(["ModCtrlLogic", "WrongLogicExec"]).copy()

    def test_no_anterior_copy(self, sub):
        dbn = compile_to_2tbn(sub)
        assert ("WrongLogicExec", ANTERIOR) not in dbn.nodes()
        assert ("ModCtrlLogic", ANTERIOR) in dbn.nodes()

    def test_tracks_precondition_times_fixed_probability(self, sub):
        """P(WrongLogicExec) == 0.8 * P(ModCtrlLogic), same slice.

        Hand-computed: WrongLogicExec is active iff its precondition is active
        (probability p) and the 0.8 reaction coin lands, and it has no other
        route to activation. So the marginal is exactly 0.8p.
        """
        config = InferenceConfig(
            clustering=frozenset({frozenset(["ModCtrlLogic"])}),
            m=M,
            p_pos=P_POS,
            p_neg=P_NEG,
        )
        engine = DBNInference(sub, config)
        result = engine.step({frozenset(["ModCtrlLogic"]): np.array([0.55, 0.45])}, {})

        p_mcl = result.marginals["ModCtrlLogic"]
        assert result.marginals["WrongLogicExec"] == pytest.approx(0.8 * p_mcl)


class TestCredAccessMitmStructure:
    """UnsecCred/ModAuthProc -> CredAccess (AND) -> MITM, plus FileAccess.

    This is the structure a per-node-independence assumption gets wrong: MITM
    depends on CredAccess at the SAME ulterior slice, and CredAccess itself is
    a same-slice AND of UnsecCred and ModAuthProc. In this isolated subgraph,
    UnsecCred and ModAuthProc have no precondition of their own (their real
    parents, UnsecCred2 and ModAuthProc1, are outside the subgraph), so they
    behave as roots here -- this only affects THEIR OWN CPT shape, not the
    CredAccess/MITM structure under test, and is called out so these numbers
    are never mistaken for the production experiment's.
    """

    @pytest.fixture
    def sub(self, ag):
        return ag.subgraph(
            ["UnsecCred", "ModAuthProc", "CredAccess", "MITM", "FileAccess"]
        ).copy()

    @pytest.fixture
    def priors(self):
        # Deliberately non-degenerate: a point-mass-at-0 prior would make a
        # buggy independence assumption agree with the correct computation by
        # accident on the first step.
        return {"UnsecCred": 0.3, "ModAuthProc": 0.6, "MITM": 0.1}

    def test_no_evidence_matches_hand_derived_closed_form(self, ag, sub, priors):
        """P(MITM=1) = P(CredAccess=1) * (m + (1-m) * p_s_MITM), where
        P(CredAccess=1) = P(UnsecCred=1) * P(ModAuthProc=1) (AND, independent
        roots) and P(X=1) = prior_X + (1-prior_X) * p_s_X for a root X
        (Table 1's root case: forced 1 if already active, else Bernoulli(p_s)).
        This is Table 1's own row structure written out algebraically, not a
        second call into the code under test.
        """
        delta_t = compute_delta_t(
            {"UnsecCred": float(sub.nodes["UnsecCred"]["ttc"]),
             "ModAuthProc": float(sub.nodes["ModAuthProc"]["ttc"]),
             "MITM": float(sub.nodes["MITM"]["ttc"])},
            m=M,
        )
        p_s_unseccred = compute_ps(float(sub.nodes["UnsecCred"]["ttc"]), delta_t)
        p_s_modauthproc = compute_ps(float(sub.nodes["ModAuthProc"]["ttc"]), delta_t)
        p_s_mitm = compute_ps(float(sub.nodes["MITM"]["ttc"]), delta_t)

        p_unseccred = priors["UnsecCred"] + (1 - priors["UnsecCred"]) * p_s_unseccred
        p_modauthproc = (
            priors["ModAuthProc"] + (1 - priors["ModAuthProc"]) * p_s_modauthproc
        )
        p_credaccess = p_unseccred * p_modauthproc
        expected_mitm = p_credaccess * (
            priors["MITM"] + (1 - priors["MITM"]) * p_s_mitm
        )

        config = InferenceConfig(
            clustering=frozenset(
                frozenset([n]) for n in ["UnsecCred", "ModAuthProc", "MITM"]
            ),
            m=M,
            p_pos=P_POS,
            p_neg=P_NEG,
        )
        engine = DBNInference(sub, config)
        belief = {
            frozenset([name]): np.array([1 - p, p])
            for name, p in priors.items()
        }
        result = engine.step(belief, {})

        assert result.marginals["MITM"] == pytest.approx(expected_mitm)

        # An independence-assumption bug (computing MITM from its own
        # anterior parents directly, ignoring the intra-slice CredAccess
        # mediator) would instead treat P(CredAccess=1) as informationless
        # for MITM's own P(MITM_prev) blend and drop the p_credaccess factor
        # entirely -- i.e. it would compute something close to
        # `priors["MITM"] + (1 - priors["MITM"]) * p_s_mitm` with no
        # p_credaccess multiplier. Assert the two differ by more than
        # floating-point noise, so this test cannot pass against that bug by
        # coincidence.
        buggy_value = priors["MITM"] + (1 - priors["MITM"]) * p_s_mitm
        assert abs(expected_mitm - buggy_value) > 1e-3

    def test_evidence_matches_direct_pgmpy(self, sub, priors):
        """Same structure, with FileAccess=1 evidence, checked against a
        directly-built pgmpy network rather than a hand closed form (the
        evidence-conditioning Bayes update is not hand-tractable here, but
        agreement with an independently-constructed exact-inference call
        still exercises the same CredAccess/MITM path end to end)."""
        dbn = attach_cpds(compile_to_2tbn(sub), sub, m=M, p_pos=P_POS, p_neg=P_NEG)
        dbn.add_cpds(TabularCPD(("UnsecCred", ANTERIOR), 2, [[0.7], [0.3]]))
        dbn.add_cpds(TabularCPD(("ModAuthProc", ANTERIOR), 2, [[0.4], [0.6]]))
        dbn.add_cpds(TabularCPD(("MITM", ANTERIOR), 2, [[0.9], [0.1]]))
        expected = _ve_marginal(dbn, "MITM", evidence={("FileAccess", ULTERIOR): 1})

        config = InferenceConfig(
            clustering=frozenset(
                frozenset([n]) for n in ["UnsecCred", "ModAuthProc", "MITM"]
            ),
            m=M,
            p_pos=P_POS,
            p_neg=P_NEG,
        )
        engine = DBNInference(sub, config)
        belief = {frozenset([name]): np.array([1 - p, p]) for name, p in priors.items()}
        result = engine.step(belief, {"FileAccess": 1})

        assert result.marginals["MITM"] == pytest.approx(expected)


class TestFFExAgreeOnDegenerateStep:
    """A sanity check that isn't in the plan but is cheap and worth having:
    FF and EX must agree exactly on the very first step from the point-mass
    initial belief, since there is no accumulated correlation yet to drop."""

    def test_first_step_agrees(self, ag):
        from src.dbn.inference import (
            exact_clustering,
            fully_factorized_clustering,
        )

        interface = sorted(n for n, d in ag.nodes(data=True) if d["self_loop"])
        ff = DBNInference(
            ag,
            InferenceConfig(fully_factorized_clustering(interface), M, P_POS, P_NEG),
        )
        ex = DBNInference(
            ag, InferenceConfig(exact_clustering(interface), M, P_POS, P_NEG)
        )

        ff_result = ff.step(ff.initial_belief(), {})
        ex_result = ex.step(ex.initial_belief(), {})

        for node in ag.nodes():
            assert ff_result.marginals[node] == pytest.approx(
                ex_result.marginals[node], abs=1e-9
            ), node


class TestLatchedReactionInference:
    """One-shot latched reaction, end-to-end through DBNInference.

    Uses the ModCtrlLogic -> WrongLogicExec__Seen -> WrongLogicExec subgraph
    (3 interface nodes, 8 joint states) so the exact-clustering path is cheap
    enough for the unit suite. The full 15-node latched model costs ~7.5 s per
    slice under EX and is measured in experiments/exp02_latched_kl.py instead.
    """

    @pytest.fixture
    def sub(self):
        ag = build_attack_graph(reaction_mode="latched")
        return ag.subgraph(
            ["ModCtrlLogic", "WrongLogicExec", "WrongLogicExec__Seen"]
        ).copy()

    def _run(self, sub, clustering, n_steps):
        from src.dbn.inference import exact_clustering, fully_factorized_clustering

        interface = sorted(n for n, d in sub.nodes(data=True) if d["self_loop"])
        cl = (
            exact_clustering(interface)
            if clustering == "EX"
            else fully_factorized_clustering(interface)
        )
        engine = DBNInference(
            sub,
            InferenceConfig(
                clustering=cl,
                m=M,
                p_pos=P_POS,
                p_neg=P_NEG,
                # Pin the production delta_t. Without this, Eq. 3 over the
                # subgraph's single timed node (ModCtrlLogic, TTC=50) gives
                # delta_t = 50 and therefore p_s = 50/50 = 1.0 -- the
                # precondition saturates in ONE slice, everything downstream
                # goes deterministic, and FF/EX agree trivially. That would
                # make these tests pass for the wrong reason.
                delta_t_override=DELTA_T,
            ),
        )
        return engine.run({}, n_steps)

    def test_all_three_nodes_persist(self, sub):
        assert sorted(n for n, d in sub.nodes(data=True) if d["self_loop"]) == [
            "ModCtrlLogic",
            "WrongLogicExec",
            "WrongLogicExec__Seen",
        ]

    def test_exact_reaction_tracks_fixed_fraction_of_precondition(self, sub):
        """Under EX, reaction / precondition -> 0.8, the fixed success prob.

        Tested as a RATIO rather than an absolute plateau: ModCtrlLogic has
        TTC=50, so at the calibrated delta_t it needs ~900 slices to saturate,
        and asserting the absolute 0.8 plateau would only be testing how long
        the test runs. The ratio is the actual structural invariant and holds
        throughout. It approaches 0.8 from slightly below because the one-shot
        fires the slice AFTER its precondition turns on, so it tracks
        0.8 x P(precondition by t-1) against a still-rising P(by t).
        """
        traj = self._run(sub, "EX", 400)
        final = traj.marginals[-1]

        ratio = final["WrongLogicExec"] / final["ModCtrlLogic"]
        assert ratio == pytest.approx(0.8, abs=0.01)

    def test_latch_tracks_precondition(self, sub):
        """The latch is a deterministic OR-accumulator over its precondition."""
        traj = self._run(sub, "EX", 400)
        final = traj.marginals[-1]
        assert final["WrongLogicExec__Seen"] == pytest.approx(
            final["ModCtrlLogic"], abs=0.01
        )

    def test_reaction_never_exceeds_fixed_success_probability(self, sub):
        """The one-shot cannot ratchet past 0.8, unlike a persistent p_s node."""
        traj = self._run(sub, "EX", 400)
        assert max(m["WrongLogicExec"] for m in traj.marginals) <= 0.8 + 1e-9

    def test_ff_and_ex_disagree(self, sub):
        """FF's independence assumption incurs a real error on a latched model.

        This is the property the memoryless reading cannot produce, and the
        reason Cerotti et al. Fig. 6c's divergence does not converge to 0.
        """
        ex = self._run(sub, "EX", 200)
        ff = self._run(sub, "FF", 200)
        gap = max(
            abs(e["WrongLogicExec"] - f["WrongLogicExec"])
            for e, f in zip(ex.marginals, ff.marginals)
        )
        assert gap > 1e-3, f"expected a real FF/EX gap, got {gap:.2e}"
