"""Gymnasium environment wrapping the twin for an adversarial RL attacker
(CLAUDE.md layer [0]/[3], Session 9, claim C3).

BANDIT, NOT A MULTI-STEP MDP (user-approved binding decision, LAB_NOTEBOOK.md
2026-08-06): one action per episode, chosen at `reset()`-then-`step()`; the
twin then plays out deterministically-but-stochastically (TTC sampling) to
either the goal or the horizon with NO further RL decisions.
`gymnasium.Env.step()` always returns `terminated=True`, `truncated=False`.
This was chosen specifically to avoid adding incremental pause/resume
capability to `TwinRunner` (verified: `TwinRunner.run()` is one monolithic
`env.run(until=horizon)` call, no stepping hook -- real, avoidable
engineering this session does not need).

OBSERVATION IS HONESTLY A CONSTANT. In a true bandit, nothing has happened
yet at decision time, so there is no legitimate per-episode context to
observe. `observation_space = Box(shape=(1,))`, always `[1.0]`. PPO solves
this as a stochastic policy-gradient bandit over the 384-arm structured
action space (`MultiDiscrete([2,2,2,2,4,3])`) -- a standard, legitimate RL
formulation. Fabricating a richer "context" vector would misrepresent what
the attacker actually knows before it acts.

ACTION SPACE (`MultiDiscrete([2,2,2,2,4,3])`, 384 total actions):
  dims 0-3: which of the 4 root nodes to activate (maps directly onto the
            EXISTING `AttackerConfig.enabled_roots`).
  dim 4 (4 values): the MITM-branch choice -- which of
            {SpoofRepMsg, UnauthCommand} to exclude (0=neither, 1=SpoofRepMsg,
            2=UnauthCommand, 3=both), via the NEW `AttackerConfig.excluded_nodes`
            (Session 9). Excluding UnauthCommand blocks BOTH its incoming
            edges (MITM's and Masquerade's) -- "avoid this graph node
            entirely" is the correct reading of a path-avoidance action.
  dim 5 (3 values): stealth level, mapped onto the EXISTING
            `AttackerConfig.speed_multiplier` (Session 8): {0.5, 1.0, 2.0}.

KNOWLEDGE LEVEL DETERMINES THE REWARD SIGNAL, NOT THE OBSERVATION -- there
is nothing to observe pre-decision in a bandit, so "knowledge" cannot
manifest as mid-episode observability the way it would in a multi-step MDP.
Each level's reward reflects what that class of attacker could plausibly
measure about detection over many training episodes:
  blind:      reward has NO detection term at all.
  analytics:  reward penalized by the mean RAW hard-evidence firing rate
              across the 8 cyber analytics (knows sensors exist, not how
              they causally fuse).
  full_dbn:   reward penalized by the mean TRUE fused P(UnstablePS)
              posterior from a real hard-evidence `DBNInference` run over
              the episode's trace (the most informed signal).
`goal_reached` always uses `grid_unstable` (the MEASURED consequence,
`GridState.unstable`), never `ground_truth["UnstablePS"]` -- this
project's own established rule (exp04/exp06's docstrings), which this
module inherits unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from src.dbn.inference import DBNInference, InferenceConfig, _interface_nodes, fully_factorized_clustering
from src.eval.lead_time import first_instability
from src.twin.attacker import AttackerConfig, DelayLaw
from src.twin.grid import GridConfig
from src.twin.runner import ContinuousTrace, DiscreteTrace, TwinConfig, TwinRunner, discretize

KnowledgeLevel = Literal["blind", "analytics", "full_dbn"]

ROOTS: tuple[str, ...] = ("UnsecCred1", "ModAuthProc1", "ModCtrlLogic", "ModifyProgram")
# index -> which of {SpoofRepMsg, UnauthCommand} to exclude this episode.
MITM_BRANCH_CHOICES: tuple[frozenset[str], ...] = (
    frozenset(),
    frozenset({"SpoofRepMsg"}),
    frozenset({"UnauthCommand"}),
    frozenset({"SpoofRepMsg", "UnauthCommand"}),
)
SPEED_CHOICES: tuple[float, ...] = (0.5, 1.0, 2.0)
# The 8 Table-2 cyber analytics, same set as exp01/exp04's hard-evidence arm.
CYBER_ANALYTICS: tuple[str, ...] = (
    "FileAccess", "FileIntegrity", "MeasureCoherence", "CommandCoherence",
    "SWIntegrityDER", "NewServiceStarted", "SWIntegritySCADA", "SuspArg",
)


@dataclass(frozen=True)
class RewardConfig:
    w_goal: float
    w_detect: float
    w_time: float


@dataclass(frozen=True)
class FidelityConfig:
    """Already-RESOLVED numbers, computed once by the caller (exp09) and
    threaded into BOTH `discretize()` and the DBN identically -- never
    re-derived inside the env, so the discretization boundary and the
    DBN's CPTs cannot silently disagree on `delta_t`."""

    delta_t: float
    n_slices: int
    horizon_time_units: float


@dataclass(frozen=True)
class DecodedAction:
    enabled_roots: tuple[str, ...]
    excluded_nodes: frozenset[str]
    speed_multiplier: float


def decode_action(action: np.ndarray) -> DecodedAction:
    roots = tuple(r for r, bit in zip(ROOTS, action[:4]) if int(bit) == 1)
    excluded = MITM_BRANCH_CHOICES[int(action[4])]
    speed = SPEED_CHOICES[int(action[5])]
    return DecodedAction(enabled_roots=roots, excluded_nodes=excluded, speed_multiplier=speed)


class RLAttackerEnv(gym.Env):
    """One bandit decision per episode: pick an attack-graph path (root
    subset + MITM-branch choice) and a stealth level, then observe the
    twin's real consequence. See module docstring for the full design."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        ag: nx.DiGraph,
        grid_cfg: GridConfig,
        twin_cfg: dict,
        knowledge_level: KnowledgeLevel,
        reward_cfg: RewardConfig,
        fidelity: FidelityConfig,
        p_pos: float,
        p_neg: float,
        seed_sequence: np.random.SeedSequence,
    ) -> None:
        super().__init__()
        if knowledge_level not in ("blind", "analytics", "full_dbn"):
            raise ValueError(f"unknown knowledge_level {knowledge_level!r}")
        self.ag = ag
        self.grid_cfg = grid_cfg
        self.twin_cfg = twin_cfg
        self.knowledge_level: KnowledgeLevel = knowledge_level
        self.reward_cfg = reward_cfg
        self.fidelity = fidelity
        self.p_pos = p_pos
        self.p_neg = p_neg

        self.action_space = spaces.MultiDiscrete([2, 2, 2, 2, 4, 3])
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self._episode_seed_root = seed_sequence
        self._pending_seed: np.random.SeedSequence | None = None

        # Built ONCE, reused across every episode -- DBNInference.run()
        # resets to initial_belief() internally on every call, so episodes
        # (independent scenarios, not a continuing timeline) never need
        # manual belief-carrying. Only full_dbn pays this cost at all.
        self._dbn_engine: DBNInference | None = None
        if knowledge_level == "full_dbn":
            clustering = fully_factorized_clustering(_interface_nodes(ag))
            self._dbn_engine = DBNInference(
                ag, InferenceConfig(clustering=clustering, m=1.0, p_pos=p_pos, p_neg=p_neg, delta_t_override=fidelity.delta_t),
            )

        self.last_trace: ContinuousTrace | None = None
        self.last_discrete: DiscreteTrace | None = None

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._pending_seed = self._episode_seed_root.spawn(1)[0]
        return np.array([1.0], dtype=np.float32), {}

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._pending_seed is None:
            raise RuntimeError("step() called before reset()")
        decoded = decode_action(np.asarray(action))

        attacker_cfg = AttackerConfig(
            delay_law=DelayLaw.EXPONENTIAL,
            enabled_roots=decoded.enabled_roots,
            excluded_nodes=decoded.excluded_nodes,
            speed_multiplier=decoded.speed_multiplier,
        )
        tw_cfg = TwinConfig(
            grid=self.grid_cfg,
            attacker=attacker_cfg,
            dispatch_period_time_units=float(self.twin_cfg["control_centre"]["dispatch_period_time_units"]),
            comms_latency_time_units=float(self.twin_cfg["comms"]["latency_time_units"]),
            horizon_time_units=self.fidelity.horizon_time_units,
        )
        trace = TwinRunner(self.ag, tw_cfg, self._pending_seed).run()
        rng = np.random.default_rng(self._pending_seed.spawn(1)[0])
        discrete = discretize(
            trace, self.ag, self.fidelity.delta_t, self.fidelity.n_slices, rng, self.p_pos, self.p_neg, zones=None,
        )
        self.last_trace = trace
        self.last_discrete = discrete

        unstable_flags = [bool(r.grid_unstable) for r in discrete.records]
        goal_reached = any(unstable_flags)
        t_first = first_instability(unstable_flags)
        slices_used = t_first if goal_reached else self.fidelity.n_slices
        time_term = slices_used / self.fidelity.n_slices

        detect_term = 0.0
        if self.knowledge_level == "analytics":
            evid = discrete.evidence_stream(list(CYBER_ANALYTICS))
            bits = [v for entry in evid.values() for v in entry.values()]
            detect_term = float(np.mean(bits)) if bits else 0.0
        elif self.knowledge_level == "full_dbn":
            assert self._dbn_engine is not None
            evid = discrete.evidence_stream(list(CYBER_ANALYTICS))
            trajectory = self._dbn_engine.run(evid, self.fidelity.n_slices)
            detect_term = float(np.mean([m["UnstablePS"] for m in trajectory.marginals]))

        reward = (
            self.reward_cfg.w_goal * float(goal_reached)
            - self.reward_cfg.w_detect * detect_term
            - self.reward_cfg.w_time * time_term
        )

        info = {
            "goal_reached": goal_reached,
            "detect_term": detect_term,
            "time_term": time_term,
            "enabled_roots": decoded.enabled_roots,
            "excluded_nodes": decoded.excluded_nodes,
            "speed_multiplier": decoded.speed_multiplier,
        }
        obs = np.array([1.0], dtype=np.float32)
        return obs, float(reward), True, False, info
