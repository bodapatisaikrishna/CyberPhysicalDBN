"""Scripted attacker executing the Figure-2 attack processes (CLAUDE.md layer [0]).

SCRIPTED, not adaptive. An RL attacker optimising against the detector is
claim C3 / Phase 5 and is explicitly later in CLAUDE.md's cut order; nothing
here reacts to the defender.

Every number comes from the attack graph. This module re-types no TTC, no
reaction probability, and no edge -- it reads `ag.nodes[n]["ttc"]`,
`ag.nodes[n]["fixed_success_prob"]` and `ag.edges` so it cannot drift from
`src/attack_graph/graph.py`.

Delay law (configs/twin.yaml `attacker.delay_law`):

  exponential   delay ~ Exp(mean = TTC). Not an aesthetic choice: Cerotti et
                al. Sec. III-E derives the CPTs by uniformization, which
                approximates an EXPONENTIAL continuous-time variable by a
                geometric discrete one. Exponential sampling is therefore the
                continuous law the DBN's CPTs are a discretization OF. It also
                makes UnsecCred's three 1/3 stages an Erlang(3) of mean 1 and
                ModAuthProc's two 1/2 stages an Erlang(2) of mean 1, matching
                the paper's own description of them as 3- and 2-stage tactics.

  deterministic delay = TTC exactly. The MISSPECIFICATION ARM: under the
                exponential law the twin draws from the DBN's own law and its
                analytics from the DBN's own likelihood, so agreement is close
                to guaranteed and the experiment would be a self-consistency
                check. This arm measures degradation when the twin's law is
                not the DBN's.

Delays are sampled when the precondition is satisfied, not at t=0. For the
exponential law this is irrelevant (memorylessness); for the deterministic arm
it is the whole difference between "takes TTC to do" and "happens at TTC".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import networkx as nx
import numpy as np
import simpy


class DelayLaw(Enum):
    EXPONENTIAL = "exponential"
    DETERMINISTIC = "deterministic"

    def sample(self, ttc: float, rng: np.random.Generator) -> float:
        if self is DelayLaw.EXPONENTIAL:
            return float(rng.exponential(ttc))
        return float(ttc)


class EventKind(Enum):
    PRECONDITION_SATISFIED = "precondition_satisfied"
    STEP_COMPLETED = "step_completed"
    REACTION_RESOLVED = "reaction_resolved"


@dataclass(frozen=True)
class AttackEvent:
    t_units: float
    kind: EventKind
    node: str
    detail: str = ""


@dataclass
class AttackerConfig:
    delay_law: DelayLaw = DelayLaw.EXPONENTIAL
    enabled_roots: tuple[str, ...] = (
        "UnsecCred1",
        "ModAuthProc1",
        "ModCtrlLogic",
        "ModifyProgram",
    )
    # Session 8 (claim C2, LAB_NOTEBOOK.md 2026-08-05): the two axes
    # exp08_transfer_c2.py sweeps to measure how realized attack-step
    # duration depends on attacker capability and defensive posture, so the
    # amortized TTC model has real (not synthetic-only) supervision for
    # both. Multiplicative on the node's own expert TTC -- never a new
    # distribution, only a rescaled mean of the SAME delay_law:
    # effective_ttc = ttc * defense_slowdown_multiplier / speed_multiplier.
    speed_multiplier: float = 1.0  # attacker capability: >1 faster/skilled
    defense_slowdown_multiplier: float = 1.0  # defensive posture: >1 hardened/monitored, slows the attacker

    def __post_init__(self) -> None:
        if self.speed_multiplier <= 0:
            raise ValueError(f"speed_multiplier must be positive, got {self.speed_multiplier}")
        if self.defense_slowdown_multiplier <= 0:
            raise ValueError(
                f"defense_slowdown_multiplier must be positive, got {self.defense_slowdown_multiplier}"
            )


class ScriptedAttacker:
    """One SimPy process per attack-graph node, gated on its preconditions."""

    def __init__(
        self,
        env: simpy.Environment,
        ag: nx.DiGraph,
        rng_delays: np.random.Generator,
        rng_reactions: np.random.Generator,
        config: AttackerConfig | None = None,
        on_complete: Callable[[str, float], None] | None = None,
    ):
        self.env = env
        self.ag = ag
        self.rng_delays = rng_delays
        self.rng_reactions = rng_reactions
        self.config = config or AttackerConfig()
        self._on_complete = on_complete

        self.events: list[AttackEvent] = []
        self.completion_times: dict[str, float] = {}
        self.eligible_times: dict[str, float] = {}
        self.reaction_outcomes: dict[str, bool] = {}

        # Every non-analytic node gets an event, including the AND/OR gates and
        # the goal, so the trace's node set matches the DBN's exactly.
        self._simulated = [
            n
            for n, d in ag.nodes(data=True)
            if d["node_type"] in ("attack_step", "reaction", "goal")
        ]
        self.done: dict[str, simpy.Event] = {n: env.event() for n in self._simulated}

        for node in self._simulated:
            env.process(self._node_process(node))

    # --- helpers ----------------------------------------------------------

    def _preconditions(self, node: str) -> list[str]:
        return sorted(
            p
            for p in self.ag.predecessors(node)
            if p != node
            and self.ag.edges[p, node]["edge_type"] == "precondition"
            and p in self.done
        )

    def _precondition_event(self, node: str):
        parents = self._preconditions(node)
        if not parents:
            return self.env.timeout(0)
        events = [self.done[p] for p in parents]
        # AND only at CredAccess (gate == "AND"); every other multi-parent node
        # is OR, matching Table 1's "precondition = OR over parents" rule.
        if self.ag.nodes[node].get("gate") == "AND":
            return self.env.all_of(events)
        return self.env.any_of(events)

    def _log(self, kind: EventKind, node: str, detail: str = "") -> None:
        self.events.append(AttackEvent(float(self.env.now), kind, node, detail))

    # --- processes --------------------------------------------------------

    def _node_process(self, node: str):
        data = self.ag.nodes[node]

        if node in self.ag and data["node_type"] == "attack_step" and data["ttc"] is None:
            yield from self._gate_process(node)
        elif data["node_type"] == "goal":
            yield from self._gate_process(node)
        elif data["node_type"] == "reaction":
            yield from self._reaction_process(node)
        else:
            yield from self._attack_step_process(node)

    def _attack_step_process(self, node: str):
        if not self._preconditions(node) and node not in self.config.enabled_roots:
            return  # a disabled root never launches

        yield self._precondition_event(node)
        self.eligible_times[node] = float(self.env.now)
        self._log(EventKind.PRECONDITION_SATISFIED, node)

        ttc = float(self.ag.nodes[node]["ttc"])
        effective_ttc = ttc * self.config.defense_slowdown_multiplier / self.config.speed_multiplier
        delay = self.config.delay_law.sample(effective_ttc, self.rng_delays)
        yield self.env.timeout(delay)

        self._complete(node, detail=f"delay={delay:.4f}")

    def _gate_process(self, node: str):
        """CredAccess (AND) and UnstablePS (OR): resolve within the slice, no TTC."""
        yield self._precondition_event(node)
        self.eligible_times[node] = float(self.env.now)
        self._log(EventKind.PRECONDITION_SATISFIED, node)
        self._complete(node, detail=f"gate={self.ag.nodes[node].get('gate')}")

    def _reaction_process(self, node: str):
        """CorrReact / WrongLogicExec: one Bernoulli draw at the fixed success prob.

        In continuous time there is no "redrawn every slice", so a twin
        reaction is inherently ONE-SHOT -- structurally the latched reading
        that LAB_NOTEBOOK.md 2026-07-31 investigated, arrived at from a
        different direction than Fig. 6c.
        """
        yield self._precondition_event(node)
        self.eligible_times[node] = float(self.env.now)
        self._log(EventKind.PRECONDITION_SATISFIED, node)

        p = float(self.ag.nodes[node]["fixed_success_prob"])
        succeeded = bool(self.rng_reactions.random() < p)
        self.reaction_outcomes[node] = succeeded
        self._log(EventKind.REACTION_RESOLVED, node, detail=f"p={p} success={succeeded}")

        if succeeded:
            self._complete(node, detail=f"p={p}")

    def _complete(self, node: str, detail: str = "") -> None:
        if node in self.completion_times:
            return
        self.completion_times[node] = float(self.env.now)
        self._log(EventKind.STEP_COMPLETED, node, detail)
        if not self.done[node].triggered:
            self.done[node].succeed()
        if self._on_complete is not None:
            self._on_complete(node, float(self.env.now))
