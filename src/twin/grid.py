"""Steady-state distribution-grid model (CLAUDE.md layer [0]/[4]).

A pandapower model of the Baran & Wu 33-bus radial distribution feeder with
DERs attached at its electrically weakest points, exposing the control-centre
setpoint that the attack graph's UnauthCommand / ModCtrlLogic steps compromise.

Steady-state only: `pandapower.runpp`. CLAUDE.md explicitly rejects dynamic or
transient simulation ("Steady-state runpp() is sufficient for the
instability-detection claim"), so nothing here integrates dynamics.

Instability is MEASURED, not asserted: a state is unstable when a bus voltage
leaves the band the network itself declares in `net.bus.min_vm_pu` /
`max_vm_pu` (0.90/1.10 pu for case33bw, coinciding with EN 50160's +/-10% band
for European distribution networks -- the source paper's setting is Italian,
under CEI 0-16). Those limits are read from the network at construction and
never written as literals in this module.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

import networkx as nx
import pandapower as pp
import pandapower.networks as pn
from pandapower.toolbox.result_info import violated_buses


class ActionOrigin(Enum):
    """Who issued a control action.

    GROUND-TRUTH LABEL ONLY. `GridModel` must never branch on this: the physics
    is a pure function of (setpoints, network). If origin leaked into the
    solve, a compromised action would be "unstable" because it was labelled
    compromised rather than because of what it did to the grid, which would
    make claim C1 circular. Enforced by
    tests/test_twin.py::test_origin_does_not_affect_physics.
    """

    OPERATOR = "operator"
    ATTACKER = "attacker"


@dataclass(frozen=True)
class ControlAction:
    """A setpoint command addressed to one DER."""

    der_id: str
    p_mw: float
    origin: ActionOrigin
    issued_at: float  # twin time units (1 unit = 10 min, the paper's Table 3 unit)
    provenance: tuple[str, ...] = ()  # attack-step node names responsible; () if legitimate


@dataclass(frozen=True)
class VoltageLimits:
    """Per-unit voltage band defining instability, with its provenance."""

    min_vm_pu: float
    max_vm_pu: float
    source: str


@dataclass(frozen=True)
class GridState:
    """Result of one steady-state solve. Immutable snapshot."""

    t_units: float
    converged: bool
    vm_pu: Mapping[int, float]  # empty when not converged
    vm_pu_min: float | None
    vm_pu_max: float | None
    argmin_bus: int | None
    argmax_bus: int | None
    violated_buses: tuple[int, ...]
    setpoints_mw: Mapping[str, float]
    limits: VoltageLimits
    unstable: bool


@dataclass
class GridConfig:
    network: str = "case33bw"
    der_placement_rule: str = "top_n_leaf_buses_by_z_ohm_from_slack"
    n_der: int = 2
    p_mw_levels: Sequence[float] = field(default_factory=lambda: [0.0, 0.8, 2.0, 3.0, 5.0])
    nominal_level_index: int = 1
    nonconvergence_is_unstable: bool = True


def _build_base_network(name: str):
    if name != "case33bw":
        raise ValueError(
            f"unsupported network {name!r}; only case33bw is wired up this session"
        )
    return pn.case33bw()


def select_der_buses(net, n_der: int) -> list[int]:
    """Top-`n_der` in-service leaf buses by impedance distance from the slack.

    DERs are sited at the electrically weakest points of the feeder, which is
    where injection most affects voltage and therefore where a compromised
    setpoint has a measurable physical consequence at all.

    The in-service filter is load-bearing, not incidental: case33bw ships 37
    lines of which the 5 Baran-Wu tie switches are out of service (normal
    radial operation). Including them makes the graph meshed, leaving no leaf
    buses whatsoever and silently returning an empty placement.
    """
    graph = nx.Graph()
    in_service = net.line[net.line.in_service]
    for _, line in in_service.iterrows():
        z_ohm = (line.r_ohm_per_km**2 + line.x_ohm_per_km**2) ** 0.5 * line.length_km
        graph.add_edge(int(line.from_bus), int(line.to_bus), z_ohm=z_ohm)

    slack = int(net.ext_grid.bus.iloc[0])
    distance = nx.single_source_dijkstra_path_length(graph, slack, weight="z_ohm")
    leaves = [n for n in graph.nodes if graph.degree(n) == 1 and n != slack]
    if len(leaves) < n_der:
        raise ValueError(f"need {n_der} leaf buses, found {len(leaves)}: {leaves}")

    ranked = sorted(leaves, key=lambda b: (-distance[b], b))
    return ranked[:n_der]


class GridModel:
    """Steady-state grid with control-centre-commanded DER setpoints."""

    def __init__(self, config: GridConfig | None = None):
        self.config = config or GridConfig()
        self._net = _build_base_network(self.config.network)

        self.der_buses = select_der_buses(self._net, self.config.n_der)
        self.der_ids: list[str] = []
        self._der_index: dict[str, int] = {}
        nominal = self.config.p_mw_levels[self.config.nominal_level_index]
        for bus in self.der_buses:
            der_id = f"DER_{bus}"
            idx = pp.create_sgen(self._net, bus=bus, p_mw=nominal, q_mvar=0.0, name=der_id)
            self.der_ids.append(der_id)
            self._der_index[der_id] = int(idx)

        # Read the band from the network rather than declaring it here, so the
        # threshold cannot drift from the data it claims to come from.
        self._limits = VoltageLimits(
            min_vm_pu=float(self._net.bus.min_vm_pu.min()),
            max_vm_pu=float(self._net.bus.max_vm_pu.max()),
            source=f"net.bus.min_vm_pu/max_vm_pu ({self.config.network}); EN 50160 +/-10%",
        )

        self._dirty = True
        self._state: GridState | None = None

    @property
    def limits(self) -> VoltageLimits:
        return self._limits

    @property
    def nominal_p_mw(self) -> float:
        return float(self.config.p_mw_levels[self.config.nominal_level_index])

    @property
    def max_p_mw(self) -> float:
        return float(max(self.config.p_mw_levels))

    def reset(self) -> None:
        for der_id in self.der_ids:
            self._net.sgen.at[self._der_index[der_id], "p_mw"] = self.nominal_p_mw
        self._dirty = True
        self._state = None

    def current_setpoints(self) -> dict[str, float]:
        return {
            der_id: float(self._net.sgen.at[idx, "p_mw"])
            for der_id, idx in self._der_index.items()
        }

    def apply_control_action(self, action: ControlAction) -> None:
        """Apply a setpoint. Mutates the network; does NOT solve.

        `action.origin` is deliberately not consulted -- see ActionOrigin.
        """
        if action.der_id not in self._der_index:
            raise KeyError(f"unknown DER {action.der_id!r}; have {self.der_ids}")
        self._net.sgen.at[self._der_index[action.der_id], "p_mw"] = float(action.p_mw)
        self._dirty = True

    def solve(self, t_units: float) -> GridState:
        """Run steady-state power flow and snapshot the result."""
        converged = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # numba is not installed in this venv; passing numba=False
                # suppresses a per-call warning rather than changing results.
                # enforce_p_lims/enforce_q_lims stay False (pandapower's own
                # defaults) so a malicious setpoint actually propagates instead
                # of being silently clamped back into a feasible range.
                pp.runpp(self._net, numba=False)
            except Exception:
                converged = False

        converged = converged and bool(self._net["converged"])
        setpoints = self.current_setpoints()

        if not converged:
            state = GridState(
                t_units=t_units,
                converged=False,
                vm_pu={},
                vm_pu_min=None,
                vm_pu_max=None,
                argmin_bus=None,
                argmax_bus=None,
                violated_buses=(),
                setpoints_mw=setpoints,
                limits=self._limits,
                unstable=self.config.nonconvergence_is_unstable,
            )
        else:
            vm = self._net.res_bus.vm_pu
            # violated_buses() takes explicit scalar limits; it does not read
            # net.bus.min_vm_pu/max_vm_pu itself.
            violated = tuple(
                int(b)
                for b in violated_buses(
                    self._net, self._limits.min_vm_pu, self._limits.max_vm_pu
                )
            )
            state = GridState(
                t_units=t_units,
                converged=True,
                vm_pu={int(b): float(v) for b, v in vm.items()},
                vm_pu_min=float(vm.min()),
                vm_pu_max=float(vm.max()),
                argmin_bus=int(vm.idxmin()),
                argmax_bus=int(vm.idxmax()),
                violated_buses=violated,
                setpoints_mw=setpoints,
                limits=self._limits,
                unstable=len(violated) > 0,
            )

        self._state = state
        self._dirty = False
        return state

    def is_unstable(self) -> bool:
        """Whether the last solved state violates the declared voltage band.

        Raises if the network was mutated since the last solve, so a stale
        answer is impossible rather than silently wrong.
        """
        if self._dirty or self._state is None:
            raise RuntimeError(
                "grid state is stale; call solve() after apply_control_action()"
            )
        return self._state.unstable


def voltage_sensitivity(
    model: GridModel, delta_p_mw: float
) -> dict[str, dict[int, float]]:
    """Linearized dV_b/dP_d (pu/MW) about the nominal dispatch, per DER.

    Answers "whose injection moves this bus" from the network topology alone,
    with no reference to any attack trajectory -- this is what lets
    `consequence.build_zone_map` derive DER-to-bus zones rather than having
    them hardcoded.

    Does not mutate `model`: solves are performed on fresh `GridModel(
    model.config)` instances, one per DER, each with that DER's setpoint
    perturbed by `delta_p_mw` from nominal.
    """
    if delta_p_mw <= 0:
        raise ValueError(f"delta_p_mw must be positive, got {delta_p_mw}")

    baseline = GridModel(model.config)
    v0 = baseline.solve(0.0)
    if not v0.converged:
        raise RuntimeError("baseline (nominal dispatch) solve did not converge")

    sensitivity: dict[str, dict[int, float]] = {}
    for der_id in model.der_ids:
        perturbed = GridModel(model.config)
        nominal_p = perturbed.current_setpoints()[der_id]
        perturbed.apply_control_action(
            ControlAction(der_id, nominal_p + delta_p_mw, ActionOrigin.OPERATOR, 0.0)
        )
        v_d = perturbed.solve(0.0)
        if not v_d.converged:
            raise RuntimeError(f"perturbed solve for {der_id!r} did not converge")

        sensitivity[der_id] = {
            bus: (v_d.vm_pu[bus] - v0.vm_pu[bus]) / delta_p_mw for bus in v0.vm_pu
        }
    return sensitivity
