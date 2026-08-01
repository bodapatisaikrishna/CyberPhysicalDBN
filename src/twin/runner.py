"""Twin orchestration and the continuous -> discrete boundary (CLAUDE.md layer [0]).

Three layers, one boundary:

  A  continuous (SimPy)   attacker, comms, control centre, grid solves.
                          Layer A does not know delta_t exists.
  B  the boundary         `discretize(...)` -- a PURE function of
                          (ContinuousTrace, delta_t, n_slices, rng, p_pos, p_neg).
  C  discrete             analytic sampling, evidence stream, DBN inference.

SimPy is not stepped at delta_t. It runs event-driven to the horizon and the
trace is discretized afterwards, which buys three things: the same run can be
re-discretized at a different delta_t (i.e. a different m) without
re-simulating, the boundary is unit-testable against a synthetic trace with no
SimPy at all, and SimPy stays event-driven, which is the only reason to use it.

Slice convention: ground truth at slice s is evaluated at the slice END,
`active(step, s) == completed_at[step] <= s * delta_t`, and slice indices are
1-BASED to match `DBNInference.run`. The 2TBN's ulterior layer at slice s
represents state at the end of that interval. Using the slice start would
delay everything by exactly one slice -- the bug class Session 2 already hit
once (the slice-vs-time-unit fix, LAB_NOTEBOOK.md 2026-07-31).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import networkx as nx
import numpy as np
import simpy

from src.twin.attacker import AttackEvent, AttackerConfig, EventKind, ScriptedAttacker
from src.twin.comms import (
    CommsBus,
    DeliveryRecord,
    Message,
    MessageType,
    PayloadCategory,
    mitm_intercept,
    modified_control_logic,
    spoof_reporting_message,
    unauthorized_command,
)
from src.twin.grid import ActionOrigin, ControlAction, GridConfig, GridModel, GridState

CONTROL_CENTRE = "ControlCentre"


@dataclass
class TwinConfig:
    grid: GridConfig = field(default_factory=GridConfig)
    attacker: AttackerConfig = field(default_factory=AttackerConfig)
    dispatch_period_time_units: float = 1.0
    comms_latency_time_units: float = 0.0
    horizon_time_units: float = 200.0
    # Voltage the spoofed report claims, in pu. Chosen as "below the declared
    # lower limit" so the control centre's reaction is unambiguous; the actual
    # number is derived from the grid's own limits at runtime, not typed here.
    spoof_margin_pu: float = 0.05


@dataclass
class ContinuousTrace:
    """Everything layer A produced, in continuous time."""

    events: list[AttackEvent] = field(default_factory=list)
    step_completion_times: dict[str, float] = field(default_factory=dict)
    step_eligible_times: dict[str, float] = field(default_factory=dict)
    reaction_outcomes: dict[str, bool] = field(default_factory=dict)
    grid_solves: list[tuple[float, GridState]] = field(default_factory=list)
    messages: tuple[DeliveryRecord, ...] = ()
    horizon_time_units: float = 0.0

    def grid_state_at(self, t_units: float) -> GridState | None:
        """Zero-order hold of the last solve at or before t.

        ZOH is EXACT here, not an approximation: the model is steady-state and
        setpoints are piecewise-constant, so the grid genuinely does not change
        between control actions. Do not "improve" this into a solve per slice.
        """
        latest: GridState | None = None
        for solve_t, state in self.grid_solves:
            if solve_t <= t_units + 1e-12:
                latest = state
            else:
                break
        return latest


@dataclass(frozen=True)
class SliceRecord:
    slice_index: int  # 1-based
    t_units: float
    ground_truth: Mapping[str, int]
    analytics: Mapping[str, int]
    false_positives: frozenset[str]
    false_negatives: frozenset[str]
    grid: GridState | None
    grid_unstable: bool


@dataclass
class DiscreteTrace:
    records: list[SliceRecord]
    analytic_names: tuple[str, ...]

    def evidence_stream(self, observed: Sequence[str]) -> dict[int, dict[str, int]]:
        """Dense evidence stream for `DBNInference.run`.

        SCOPE GUARD: only analytic nodes may be observed. Feeding physical
        state into the DBN is Session 4 (claim C1); rejecting non-analytics
        here makes closing the loop early impossible rather than merely
        discouraged.

        Emits an explicit 0 at EVERY slice for every observed analytic --
        never a missing key. `DBNInference.run` treats a missing slice as NO
        evidence, not as zero, so a sparse stream would silently mean
        something different.
        """
        unknown = [n for n in observed if n not in self.analytic_names]
        if unknown:
            raise ValueError(
                f"evidence may only come from analytic nodes; got {unknown}. "
                "Physical feedback into the DBN is Session 4 (claim C1)."
            )
        return {
            record.slice_index: {name: record.analytics[name] for name in observed}
            for record in self.records
        }


# --- layer A: the simulation ------------------------------------------------


class TwinRunner:
    """Wires attacker -> comms -> grid -> trace in continuous SimPy time."""

    def __init__(self, ag: nx.DiGraph, config: TwinConfig, seed_sequence: np.random.SeedSequence):
        self.ag = ag
        self.config = config
        streams = seed_sequence.spawn(3)
        # Independent streams so that changing p_pos does not shift the
        # attacker's sampled delays -- sensitivity arms stay paired.
        self.rng_delays = np.random.default_rng(streams[0])
        self.rng_reactions = np.random.default_rng(streams[1])
        self.rng_analytics = np.random.default_rng(streams[2])

        self.env = simpy.Environment()
        self.grid = GridModel(config.grid)
        self.bus = CommsBus(self.env, config.comms_latency_time_units)
        self.trace = ContinuousTrace(horizon_time_units=config.horizon_time_units)

        self._der_ids = tuple(self.grid.der_ids)
        self._level_index = config.grid.nominal_level_index
        self._corr_react_done = False

        self.attacker = ScriptedAttacker(
            self.env,
            ag,
            self.rng_delays,
            self.rng_reactions,
            config.attacker,
            on_complete=self._on_step_complete,
        )

    # --- attack-step effects ---------------------------------------------

    def _on_step_complete(self, node: str, t_units: float) -> None:
        """Install the comms manipulation (if any) that this step enables."""
        top = self.grid.max_p_mw
        if node == "MITM":
            self.bus.register_manipulation("MITM", mitm_intercept())
        elif node == "SpoofRepMsg":
            spoofed = self.grid.limits.min_vm_pu - self.config.spoof_margin_pu
            self.bus.register_manipulation("SpoofRepMsg", spoof_reporting_message(spoofed))
        elif node == "UnauthCommand":
            self.bus.register_manipulation(
                "UnauthCommand", unauthorized_command(self._der_ids, top)
            )
        elif node == "WrongLogicExec":
            self.bus.register_manipulation(
                "WrongLogicExec", modified_control_logic(self._der_ids, top)
            )
        elif node == "CorrReact":
            # The control centre reacts CORRECTLY to data that is wrong. Its
            # reaction raises DER injection to fix an apparent undervoltage,
            # which on the real feeder drives overvoltage. Fig. 2 has
            # CorrReact feeding the OR gate into UnstablePS, so it CONTRIBUTES
            # to instability -- it is not a defensive correction.
            self._corr_react_done = True

    # --- processes --------------------------------------------------------

    def _der_report_process(self):
        """Each DER periodically reports the feeder voltage it observes."""
        while True:
            state = self.grid.solve(float(self.env.now))
            self.trace.grid_solves.append((float(self.env.now), state))
            for der_id in self._der_ids:
                self.bus.send(
                    Message(
                        msg_type=MessageType.MEASUREMENT,
                        sender=der_id,
                        receiver=CONTROL_CENTRE,
                        payload_category=PayloadCategory.VOLTAGE_REPORT,
                        payload={"vm_pu_min": state.vm_pu_min if state.converged else 0.0},
                        timestamp=float(self.env.now),
                    )
                )
            yield self.env.timeout(self.config.dispatch_period_time_units)

    def _control_centre_process(self):
        """Climb the dispatch ladder while the (possibly spoofed) view shows undervoltage."""
        inbox = self.bus.subscribe(CONTROL_CENTRE)
        levels = list(self.config.grid.p_mw_levels)
        while True:
            message = yield inbox.get()
            reported = float(message.payload.get("vm_pu_min", 1.0))
            if reported < self.grid.limits.min_vm_pu and self._corr_react_done:
                if self._level_index < len(levels) - 1:
                    self._level_index += 1
                    for der_id in self._der_ids:
                        self.bus.send(
                            Message(
                                msg_type=MessageType.COMMAND,
                                sender=CONTROL_CENTRE,
                                receiver=der_id,
                                payload_category=PayloadCategory.SETPOINT,
                                payload={"p_mw": levels[self._level_index]},
                                timestamp=float(self.env.now),
                            )
                        )

    def _der_actuation_process(self, der_id: str):
        """A DER executes whatever setpoint command reaches it."""
        inbox = self.bus.subscribe(der_id)
        while True:
            message = yield inbox.get()
            if message.payload_category is not PayloadCategory.SETPOINT:
                continue
            action = ControlAction(
                der_id=der_id,
                p_mw=float(message.payload["p_mw"]),
                origin=message.origin,
                issued_at=float(self.env.now),
                provenance=message.tampered_by,
            )
            self.grid.apply_control_action(action)
            state = self.grid.solve(float(self.env.now))
            self.trace.grid_solves.append((float(self.env.now), state))

    def run(self) -> ContinuousTrace:
        initial = self.grid.solve(0.0)
        self.trace.grid_solves.append((0.0, initial))

        self.env.process(self._der_report_process())
        self.env.process(self._control_centre_process())
        for der_id in self._der_ids:
            self.env.process(self._der_actuation_process(der_id))

        self.env.run(until=self.config.horizon_time_units)

        self.trace.events = list(self.attacker.events)
        self.trace.step_completion_times = dict(self.attacker.completion_times)
        self.trace.step_eligible_times = dict(self.attacker.eligible_times)
        self.trace.reaction_outcomes = dict(self.attacker.reaction_outcomes)
        self.trace.messages = self.bus.log
        self.trace.grid_solves.sort(key=lambda pair: pair[0])
        return self.trace


# --- layer B: the boundary (pure) -------------------------------------------


def discretize(
    trace: ContinuousTrace,
    ag: nx.DiGraph,
    delta_t: float,
    n_slices: int,
    rng: np.random.Generator,
    p_pos: float,
    p_neg: float,
) -> DiscreteTrace:
    """Map a continuous trace onto the DBN's slice grid. Pure w.r.t. `trace`.

    Analytics are sampled HERE, not in continuous time. A false-positive RATE
    is meaningless without a clock, and the DBN's clock is the slice; sampling
    per slice is literally forward-sampling Cerotti et al. Table 2. It also
    reproduces exp01's resolved evidence semantics (dense, 0 before / 1 at and
    after the trigger) as an emergent consequence rather than a rule.
    """
    simulated = [
        n
        for n, d in ag.nodes(data=True)
        if d["node_type"] in ("attack_step", "reaction", "goal")
    ]
    analytics = tuple(
        sorted(n for n, d in ag.nodes(data=True) if d["node_type"] == "analytic")
    )
    trigger_parent = {
        analytic: next(
            p
            for p in ag.predecessors(analytic)
            if ag.edges[p, analytic]["edge_type"] == "triggers_analytic"
        )
        for analytic in analytics
    }

    records: list[SliceRecord] = []
    for slice_index in range(1, n_slices + 1):
        t_units = slice_index * delta_t

        ground_truth = {
            node: int(trace.step_completion_times.get(node, float("inf")) <= t_units + 1e-12)
            for node in simulated
        }

        emitted: dict[str, int] = {}
        false_positives: set[str] = set()
        false_negatives: set[str] = set()
        for analytic in analytics:
            parent_active = ground_truth.get(trigger_parent[analytic], 0)
            if parent_active:
                detected = rng.random() >= p_neg
                emitted[analytic] = int(detected)
                if not detected:
                    false_negatives.add(analytic)
            else:
                spurious = rng.random() < p_pos
                emitted[analytic] = int(spurious)
                if spurious:
                    false_positives.add(analytic)

        grid_state = trace.grid_state_at(t_units)
        records.append(
            SliceRecord(
                slice_index=slice_index,
                t_units=t_units,
                ground_truth=ground_truth,
                analytics=emitted,
                false_positives=frozenset(false_positives),
                false_negatives=frozenset(false_negatives),
                grid=grid_state,
                grid_unstable=bool(grid_state.unstable) if grid_state else False,
            )
        )

    return DiscreteTrace(records=records, analytic_names=analytics)


# --- invariant validation (shared by tests and the experiment gate) ---------


@dataclass(frozen=True)
class Violation:
    kind: str
    detail: str


def validate_trace(trace: ContinuousTrace, ag: nx.DiGraph) -> list[Violation]:
    """Structural invariants of a continuous trace.

    Edges are derived from `ag` at call time so this cannot drift from the
    attack graph. Used both by tests/test_twin.py and by exp03's gate, so they
    can never disagree about what "valid" means.
    """
    violations: list[Violation] = []
    completed = trace.step_completion_times

    for source, target, data in ag.edges(data=True):
        if data["edge_type"] != "precondition" or source == target:
            continue
        if target not in completed:
            continue
        gate = ag.nodes[target].get("gate")
        if gate == "AND":
            if source not in completed:
                violations.append(
                    Violation("and_gate_missing_parent", f"{target} without {source}")
                )
            elif completed[source] > completed[target] + 1e-9:
                violations.append(
                    Violation(
                        "child_before_parent",
                        f"{target}@{completed[target]:.6f} < {source}@{completed[source]:.6f}",
                    )
                )

    # OR/default nodes: at least one precondition parent must have completed
    # no later than the child.
    for node in completed:
        parents = [
            p
            for p in ag.predecessors(node)
            if p != node
            and ag.edges[p, node]["edge_type"] == "precondition"
            and ag.nodes[p]["node_type"] in ("attack_step", "reaction", "goal")
        ]
        if not parents or ag.nodes[node].get("gate") == "AND":
            continue
        satisfied = [
            p for p in parents if p in completed and completed[p] <= completed[node] + 1e-9
        ]
        if not satisfied:
            violations.append(
                Violation(
                    "no_satisfied_precondition",
                    f"{node}@{completed[node]:.6f} parents={parents}",
                )
            )

    return violations
