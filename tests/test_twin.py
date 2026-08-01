"""Digital-twin validation gate (Session 3).

Asserts the three gate conditions plus the discretization boundary:

  (a) attack steps complete in an order consistent with preconditions --
      never a child before its parent
  (b) analytics fire only after their triggering step, except false positives
  (c) grid state changes measurably when a compromised control action lands

Design rule throughout: assert INVARIANTS and RELATIONS, never measured
values. A test asserting vm_pu == 1.1311 would bake an experimental result
into the suite, which CLAUDE.md rule 1 forbids as surely as writing it into a
README.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.attack_graph.graph import build_attack_graph
from src.dbn.inference import (
    DBNInference,
    InferenceConfig,
    _interface_nodes,
    fully_factorized_clustering,
)
from src.twin.attacker import AttackerConfig, DelayLaw
from src.twin.grid import (
    ActionOrigin,
    ControlAction,
    GridConfig,
    GridModel,
    select_der_buses,
)
from src.twin.runner import (
    ContinuousTrace,
    TwinConfig,
    TwinRunner,
    discretize,
    validate_trace,
)

DELTA_T = 166.13 / 600  # Session-2 timebase (Table 5, m=1)
N_SEEDS = 25
SHORT_HORIZON = 120.0


@pytest.fixture(scope="module")
def ag():
    return build_attack_graph()


def _run_twin(ag, seed: int, horizon: float = SHORT_HORIZON, **cfg_kwargs) -> ContinuousTrace:
    config = TwinConfig(horizon_time_units=horizon, **cfg_kwargs)
    return TwinRunner(ag, config, np.random.SeedSequence(seed)).run()


# --- gate (a): precondition ordering ---------------------------------------


class TestPreconditionOrdering:
    @pytest.mark.parametrize("seed", range(N_SEEDS))
    def test_no_child_completes_before_its_parent(self, ag, seed):
        """The core ordering invariant, across many seeds.

        `validate_trace` derives the edge list from the attack graph at call
        time, so this cannot silently drift if the graph changes.
        """
        trace = _run_twin(ag, seed)
        assert validate_trace(trace, ag) == []

    @pytest.mark.parametrize("seed", range(5))
    def test_and_gate_equals_max_of_parents(self, ag, seed):
        """CredAccess is AND-gated: it completes exactly when its slower parent does."""
        trace = _run_twin(ag, seed)
        done = trace.step_completion_times
        if "CredAccess" not in done:
            pytest.skip("CredAccess did not complete within the horizon")
        assert done["CredAccess"] == pytest.approx(
            max(done["UnsecCred"], done["ModAuthProc"])
        )

    @pytest.mark.parametrize("seed", range(5))
    def test_or_gate_equals_min_of_completed_parents(self, ag, seed):
        """UnstablePS is OR-gated: it completes with its fastest completed parent."""
        trace = _run_twin(ag, seed)
        done = trace.step_completion_times
        if "UnstablePS" not in done:
            pytest.skip("UnstablePS did not complete within the horizon")
        parents = [
            done[p]
            for p in ("WrongLogicExec", "CorrReact", "UnauthCommand")
            if p in done
        ]
        assert done["UnstablePS"] == pytest.approx(min(parents))

    def test_deterministic_law_reproduces_precondition_arithmetic(self, ag):
        """With delay = TTC exactly, completion times are hand-checkable.

        UnsecCred = 3 x 1/3 = 1; ModAuthProc = 2 x 1/2 = 1; CredAccess =
        max(1,1) = 1; MITM = 1 + 2 = 3; SpoofRepMsg = 3 + 15 = 18;
        Masquerade = 2 + 2 = 4; UnauthCommand = min(MITM, Masquerade) + 40 = 43.
        """
        trace = _run_twin(
            ag,
            seed=0,
            horizon=200.0,
            attacker=AttackerConfig(delay_law=DelayLaw.DETERMINISTIC),
        )
        done = trace.step_completion_times
        assert done["UnsecCred"] == pytest.approx(1.0)
        assert done["ModAuthProc"] == pytest.approx(1.0)
        assert done["CredAccess"] == pytest.approx(1.0)
        assert done["MITM"] == pytest.approx(3.0)
        assert done["SpoofRepMsg"] == pytest.approx(18.0)
        assert done["Masquerade"] == pytest.approx(4.0)
        assert done["UnauthCommand"] == pytest.approx(43.0)

    def test_completion_is_monotone(self, ag):
        """A step never un-completes -- the self-loop persistence assumption."""
        trace = _run_twin(ag, seed=1)
        completions = [
            e for e in trace.events if e.kind.value == "step_completed"
        ]
        seen = set()
        for event in completions:
            assert event.node not in seen, f"{event.node} completed twice"
            seen.add(event.node)
        assert [e.t_units for e in completions] == sorted(e.t_units for e in completions)


# --- gate (b): analytics fire only after their trigger ----------------------


class TestAnalyticFiring:
    def _discretize(self, ag, trace, p_pos, p_neg, seed=0, n_slices=400):
        return discretize(
            trace, ag, DELTA_T, n_slices, np.random.default_rng(seed), p_pos, p_neg
        )

    @pytest.mark.parametrize("seed", range(8))
    def test_noiseless_analytic_equals_parent_activity(self, ag, seed):
        """With p_pos = p_neg = 0 the analytic bit EQUALS parent activity, slice for slice.

        This is gate (b) with the false-positive escape hatch removed by
        construction, so it is exact and completely seed-robust rather than
        statistical.
        """
        trace = _run_twin(ag, seed)
        discrete = self._discretize(ag, trace, p_pos=0.0, p_neg=0.0, seed=seed)
        triggers = {
            a: next(
                p
                for p in ag.predecessors(a)
                if ag.edges[p, a]["edge_type"] == "triggers_analytic"
            )
            for a in discrete.analytic_names
        }
        for record in discrete.records:
            for analytic, parent in triggers.items():
                assert record.analytics[analytic] == record.ground_truth[parent], (
                    f"slice {record.slice_index}: {analytic} != activity of {parent}"
                )

    @pytest.mark.parametrize("seed", range(5))
    def test_every_mismatch_is_attributed(self, ag, seed):
        """Under heavy noise, every deviation is recorded as an FP or an FN.

        p = 0.5 is a deliberate stress value, not a plausible one -- it makes
        mismatches common so the bookkeeping is actually exercised. The
        assertion is exact, with no statistical tolerance.
        """
        trace = _run_twin(ag, seed)
        discrete = self._discretize(ag, trace, p_pos=0.5, p_neg=0.5, seed=seed)
        triggers = {
            a: next(
                p
                for p in ag.predecessors(a)
                if ag.edges[p, a]["edge_type"] == "triggers_analytic"
            )
            for a in discrete.analytic_names
        }
        for record in discrete.records:
            for analytic, parent in triggers.items():
                emitted = record.analytics[analytic]
                active = record.ground_truth[parent]
                if emitted == active:
                    assert analytic not in record.false_positives
                    assert analytic not in record.false_negatives
                elif emitted == 1:
                    assert analytic in record.false_positives
                else:
                    assert analytic in record.false_negatives

    def test_false_positive_rate_is_calibrated(self, ag):
        """The only statistical test here. Band is a derived 6-sigma binomial
        bound, not a tuned tolerance."""
        trace = _run_twin(ag, seed=3)
        p_pos = 0.2
        discrete = self._discretize(ag, trace, p_pos=p_pos, p_neg=0.0, seed=7, n_slices=700)

        inactive = 0
        fired = 0
        triggers = {
            a: next(
                p
                for p in ag.predecessors(a)
                if ag.edges[p, a]["edge_type"] == "triggers_analytic"
            )
            for a in discrete.analytic_names
        }
        for record in discrete.records:
            for analytic, parent in triggers.items():
                if record.ground_truth[parent] == 0:
                    inactive += 1
                    fired += record.analytics[analytic]

        assert inactive > 100, "not enough inactive samples to calibrate against"
        expected = inactive * p_pos
        sigma = (inactive * p_pos * (1 - p_pos)) ** 0.5
        assert abs(fired - expected) <= 6 * sigma


# --- gate (c): grid changes measurably --------------------------------------


class TestGridPhysics:
    def test_der_placement_rule_yields_17_and_32(self):
        """The DER buses are derived from the topology, not hand-picked."""
        grid = GridModel()
        assert grid.der_buses == [17, 32]

    def test_placement_requires_in_service_filter(self):
        """case33bw's 5 tie lines ship out of service; including them leaves no leaves."""
        import pandapower.networks as pn

        net = pn.case33bw()
        assert (~net.line.in_service).sum() == 5
        net.line.loc[:, "in_service"] = True
        with pytest.raises(ValueError, match="leaf buses"):
            select_der_buses(net, 2)

    def test_limits_come_from_the_network(self):
        """Guards against 0.90/1.10 ever being typed as a literal."""
        import pandapower.networks as pn

        net = pn.case33bw()
        grid = GridModel()
        assert grid.limits.min_vm_pu == float(net.bus.min_vm_pu.min())
        assert grid.limits.max_vm_pu == float(net.bus.max_vm_pu.max())

    def test_compromised_setpoint_changes_grid_measurably(self):
        """Gate (c). Asserts RELATIONS, never measured voltages."""
        grid = GridModel()
        nominal = grid.solve(0.0)
        assert nominal.converged
        assert not nominal.unstable
        assert nominal.violated_buses == ()

        for der_id in grid.der_ids:
            grid.apply_control_action(
                ControlAction(der_id, grid.max_p_mw, ActionOrigin.ATTACKER, 1.0, ("UnauthCommand",))
            )
        compromised = grid.solve(1.0)

        assert compromised.converged
        assert compromised.vm_pu_max > nominal.vm_pu_max
        assert compromised.vm_pu_max > grid.limits.max_vm_pu
        assert compromised.violated_buses != ()
        assert compromised.unstable

    def test_origin_does_not_affect_physics(self):
        """The attacker label must never reach the solve, or C1 becomes circular."""
        results = []
        for origin in (ActionOrigin.OPERATOR, ActionOrigin.ATTACKER):
            grid = GridModel()
            for der_id in grid.der_ids:
                grid.apply_control_action(ControlAction(der_id, 3.0, origin, 0.0))
            state = grid.solve(0.0)
            results.append((state.vm_pu_min, state.vm_pu_max, state.violated_buses, state.unstable))
        assert results[0] == results[1]

    def test_is_unstable_requires_a_fresh_solve(self):
        grid = GridModel()
        grid.solve(0.0)
        grid.apply_control_action(ControlAction(grid.der_ids[0], 5.0, ActionOrigin.ATTACKER, 1.0))
        with pytest.raises(RuntimeError, match="stale"):
            grid.is_unstable()

    def test_setpoints_persist_between_solves(self):
        grid = GridModel()
        first = grid.solve(0.0)
        second = grid.solve(1.0)
        assert first.setpoints_mw == second.setpoints_mw
        assert first.vm_pu_max == second.vm_pu_max


# --- (d) the discretization boundary ----------------------------------------


class TestDiscretizationBoundary:
    def test_discretize_is_pure(self, ag):
        """Same trace + same seed -> identical result, run twice."""
        trace = _run_twin(ag, seed=2)
        a = discretize(trace, ag, DELTA_T, 200, np.random.default_rng(5), 1e-4, 1e-4)
        b = discretize(trace, ag, DELTA_T, 200, np.random.default_rng(5), 1e-4, 1e-4)
        assert [r.analytics for r in a.records] == [r.analytics for r in b.records]
        assert [r.ground_truth for r in a.records] == [r.ground_truth for r in b.records]

    def test_slice_alignment_uses_slice_end(self, ag):
        """A step completing at t is active from ceil(t/delta_t) onward, not before."""
        trace = _run_twin(ag, seed=4)
        discrete = discretize(
            trace, ag, DELTA_T, 400, np.random.default_rng(0), 0.0, 0.0
        )
        for node, completed_at in trace.step_completion_times.items():
            expected_first = int(np.ceil(completed_at / DELTA_T))
            if not 1 <= expected_first <= 400:
                continue
            before = [
                r.ground_truth[node] for r in discrete.records if r.slice_index < expected_first
            ]
            after = [
                r.ground_truth[node] for r in discrete.records if r.slice_index >= expected_first
            ]
            assert set(before) <= {0}, f"{node} active before slice {expected_first}"
            assert set(after) == {1}, f"{node} not persistently active from {expected_first}"

    def test_evidence_stream_is_dense(self, ag):
        """Every slice carries an explicit 0/1 -- never a missing key.

        `DBNInference.run` reads a MISSING slice as 'no evidence', not as zero,
        so a sparse stream would silently mean something different.
        """
        trace = _run_twin(ag, seed=6)
        discrete = discretize(trace, ag, DELTA_T, 300, np.random.default_rng(1), 1e-4, 1e-4)
        observed = ["FileAccess", "MeasureCoherence"]
        stream = discrete.evidence_stream(observed)
        assert set(stream) == set(range(1, 301))
        for slice_index, evidence in stream.items():
            assert set(evidence) == set(observed)
            assert all(v in (0, 1) for v in evidence.values())

    def test_evidence_stream_rejects_non_analytics(self, ag):
        """Session-4 scope guard: physical state must not reach the DBN yet."""
        trace = _run_twin(ag, seed=7)
        discrete = discretize(trace, ag, DELTA_T, 50, np.random.default_rng(2), 1e-4, 1e-4)
        with pytest.raises(ValueError, match="analytic"):
            discrete.evidence_stream(["FileAccess", "UnstablePS"])
        with pytest.raises(ValueError, match="analytic"):
            discrete.evidence_stream(["vm_pu_min"])

    def test_evidence_stream_drives_the_dbn(self, ag):
        """End-to-end through DBNInference.run, checking the 1-based key contract."""
        trace = _run_twin(ag, seed=8)
        discrete = discretize(trace, ag, DELTA_T, 5, np.random.default_rng(3), 1e-4, 1e-4)
        stream = discrete.evidence_stream(["FileAccess"])

        interface = _interface_nodes(ag)
        engine = DBNInference(
            ag,
            InferenceConfig(
                clustering=fully_factorized_clustering(interface),
                m=1.0,
                p_pos=1e-4,
                p_neg=1e-4,
                delta_t_override=DELTA_T,
            ),
        )
        trajectory = engine.run(stream, 5)
        assert len(trajectory.marginals) == 5
        assert all(0.0 <= m["UnstablePS"] <= 1.0 for m in trajectory.marginals)

    def test_exponential_delay_mean_matches_ttc(self, ag):
        """5-sigma CLT band on the sampled mean, fixed seed."""
        rng = np.random.default_rng(11)
        n = 5000
        for node in ("MITM", "SpoofRepMsg", "ModCtrlLogic"):
            ttc = float(ag.nodes[node]["ttc"])
            samples = np.array(
                [DelayLaw.EXPONENTIAL.sample(ttc, rng) for _ in range(n)]
            )
            sigma = ttc / np.sqrt(n)  # Exp(mean=ttc) has sd == mean
            assert abs(samples.mean() - ttc) <= 5 * sigma


class TestTimebasePin:
    """exp03 duplicates exp01's timebase constants rather than refactoring a
    validated, committed script. This pins them so the duplication cannot
    silently drift."""

    def _load(self, name: str):
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "experiments" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_exp03_timebase_matches_exp01(self):
        exp01 = self._load("exp01_reproduce_paper")
        exp03 = self._load("exp03_twin_open_loop")
        assert exp03.T_TIME_UNITS == exp01.T_TIME_UNITS
        assert exp03.TIME_UNIT_SECONDS == exp01.TIME_UNIT_SECONDS
        assert exp03.DELTA_T_OVERRIDE == exp01.DELTA_T_OVERRIDE
        assert exp03.SLICES_PER_TIME_UNIT == exp01.SLICES_PER_TIME_UNIT
        assert exp03.N_SLICES == exp01.N_SLICES
        assert exp03.RAISING_TIMES == exp01.RAISING_TIMES

    def test_exp03_scripted_stream_matches_exp01(self):
        exp01 = self._load("exp01_reproduce_paper")
        exp03 = self._load("exp03_twin_open_loop")
        assert exp03.scripted_scenario2_stream() == exp01.scenario2_evidence_stream()
