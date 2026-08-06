"""Tests for src/twin/rl_attacker.py (Session 9, claim C3)."""

from __future__ import annotations

import numpy as np
import pytest

from src.attack_graph.graph import build_attack_graph
from src.dbn.inference import DBNInference, InferenceConfig, _interface_nodes, fully_factorized_clustering
from src.twin.grid import GridConfig
from src.twin.rl_attacker import (
    CYBER_ANALYTICS,
    MITM_BRANCH_CHOICES,
    ROOTS,
    SPEED_CHOICES,
    FidelityConfig,
    RewardConfig,
    RLAttackerEnv,
    decode_action,
)

# Tiny, VALID fidelity for fast tests: delta_t must stay < min TTC (1/3,
# UnsecCred*) or a CPT probability p_s=delta_t/ttc would exceed 1.
TINY_FIDELITY = FidelityConfig(delta_t=0.3, n_slices=5, horizon_time_units=1.5)
TWIN_CFG = {"control_centre": {"dispatch_period_time_units": 1.0}, "comms": {"latency_time_units": 0.0}}
REWARD_CFG = RewardConfig(w_goal=1.0, w_detect=1.0, w_time=0.1)


@pytest.fixture(scope="module")
def ag():
    return build_attack_graph(reaction_mode="memoryless")


def _make_env(ag, knowledge_level, seed=0, fidelity=TINY_FIDELITY):
    return RLAttackerEnv(
        ag, GridConfig(), TWIN_CFG, knowledge_level, REWARD_CFG, fidelity,
        p_pos=1e-4, p_neg=1e-4, seed_sequence=np.random.SeedSequence(seed),
    )


class TestSpaces:
    def test_action_space_shape(self, ag):
        env = _make_env(ag, "blind")
        assert list(env.action_space.nvec) == [2, 2, 2, 2, 4, 3]

    def test_observation_space_shape(self, ag):
        env = _make_env(ag, "blind")
        assert env.observation_space.shape == (1,)

    def test_reset_returns_constant_observation(self, ag):
        env = _make_env(ag, "blind")
        obs, info = env.reset()
        assert obs.tolist() == [1.0]
        assert info == {}


class TestStepContract:
    def test_five_tuple_terminated_true_truncated_false(self, ag):
        env = _make_env(ag, "blind")
        env.reset()
        obs, reward, terminated, truncated, info = env.step(np.array([1, 0, 0, 0, 0, 1]))
        assert obs.shape == (1,)
        assert isinstance(reward, float)
        assert terminated is True
        assert truncated is False
        assert isinstance(info, dict)

    def test_step_before_reset_raises(self, ag):
        env = _make_env(ag, "blind")
        with pytest.raises(RuntimeError):
            env.step(np.array([1, 0, 0, 0, 0, 1]))


class TestActionDecoding:
    def test_all_roots_selected(self):
        decoded = decode_action(np.array([1, 1, 1, 1, 0, 1]))
        assert decoded.enabled_roots == ROOTS

    def test_no_roots_selected(self):
        decoded = decode_action(np.array([0, 0, 0, 0, 0, 1]))
        assert decoded.enabled_roots == ()

    def test_single_root(self):
        decoded = decode_action(np.array([0, 1, 0, 0, 0, 1]))
        assert decoded.enabled_roots == ("ModAuthProc1",)

    @pytest.mark.parametrize("idx", range(4))
    def test_every_mitm_branch_choice(self, idx):
        decoded = decode_action(np.array([0, 0, 0, 0, idx, 1]))
        assert decoded.excluded_nodes == MITM_BRANCH_CHOICES[idx]

    @pytest.mark.parametrize("idx", range(3))
    def test_every_speed_choice(self, idx):
        decoded = decode_action(np.array([0, 0, 0, 0, 0, idx]))
        assert decoded.speed_multiplier == SPEED_CHOICES[idx]

    def test_info_round_trips_decoded_action(self, ag):
        env = _make_env(ag, "blind")
        env.reset()
        action = np.array([1, 0, 1, 0, 2, 0])
        _, _, _, _, info = env.step(action)
        assert info["enabled_roots"] == ("UnsecCred1", "ModCtrlLogic")
        assert info["excluded_nodes"] == frozenset({"UnauthCommand"})
        assert info["speed_multiplier"] == 0.5


class TestKnowledgeLevels:
    def test_blind_never_constructs_dbn_engine(self, ag):
        env = _make_env(ag, "blind")
        assert env._dbn_engine is None

    def test_blind_detect_term_always_zero(self, ag):
        env = _make_env(ag, "blind")
        env.reset()
        _, _, _, _, info = env.step(np.array([1, 1, 1, 1, 0, 2]))
        assert info["detect_term"] == 0.0

    def test_analytics_constructs_no_dbn_engine(self, ag):
        env = _make_env(ag, "analytics")
        assert env._dbn_engine is None

    def test_analytics_detect_term_is_raw_evidence_rate(self, ag):
        env = _make_env(ag, "analytics")
        env.reset()
        _, _, _, _, info = env.step(np.array([1, 1, 1, 1, 0, 2]))
        assert 0.0 <= info["detect_term"] <= 1.0

    def test_full_dbn_constructs_exactly_one_engine(self, ag):
        env = _make_env(ag, "full_dbn")
        assert isinstance(env._dbn_engine, DBNInference)

    def test_full_dbn_reuses_same_engine_across_episodes(self, ag):
        env = _make_env(ag, "full_dbn")
        engine_before = env._dbn_engine
        env.reset()
        env.step(np.array([1, 1, 0, 0, 0, 1]))
        env.reset()
        env.step(np.array([0, 0, 1, 1, 1, 2]))
        assert env._dbn_engine is engine_before

    def test_full_dbn_matches_direct_dbninference_run(self, ag):
        """Each episode's detect_term must equal a fresh, independent
        DBNInference.run() call on the SAME evidence -- proving no belief
        carries over between episodes."""
        env = _make_env(ag, "full_dbn", seed=7)
        env.reset()
        action = np.array([1, 1, 1, 1, 0, 1])
        _, _, _, _, info = env.step(action)
        evid = env.last_discrete.evidence_stream(list(CYBER_ANALYTICS))

        clustering = fully_factorized_clustering(_interface_nodes(ag))
        independent_engine = DBNInference(
            ag, InferenceConfig(clustering=clustering, m=1.0, p_pos=1e-4, p_neg=1e-4, delta_t_override=TINY_FIDELITY.delta_t),
        )
        trajectory = independent_engine.run(evid, TINY_FIDELITY.n_slices)
        expected = float(np.mean([m["UnstablePS"] for m in trajectory.marginals]))
        assert info["detect_term"] == pytest.approx(expected)


class TestFidelityConsistency:
    def test_discretize_and_dbn_use_identical_delta_t_n_slices(self, ag):
        env = _make_env(ag, "full_dbn")
        env.reset()
        env.step(np.array([1, 1, 1, 1, 0, 1]))
        assert len(env.last_discrete.records) == TINY_FIDELITY.n_slices
        assert env.last_discrete.records[0].t_units == pytest.approx(TINY_FIDELITY.delta_t)


class TestRewardSanity:
    def test_no_roots_no_goal_reward_is_negative_or_zero(self, ag):
        """Selecting zero roots can never reach the goal -- reward should
        never contain a positive goal term."""
        env = _make_env(ag, "blind")
        env.reset()
        _, reward, _, _, info = env.step(np.array([0, 0, 0, 0, 0, 1]))
        assert info["goal_reached"] is False
        assert reward <= 0.0
