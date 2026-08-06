"""A family of synthetic attack-graph variants for testing claim C2's
transfer requirement (CLAUDE.md layer [3], Session 8): "transfer is the
entire claim -- same-graph fitting is not a contribution."

`src/attack_graph/graph.py::build_attack_graph()` builds exactly ONE fixed
20-node graph (the source paper's Figure 2) -- it has no depth/branching
parameter. This module generates a FAMILY of differently-shaped graphs
(varying depth, branching factor, technique mix, analytic coverage),
reusing the SAME 8-technique vocabulary `graph.py::technique_table3_ttc()`
declares (never inventing a 9th technique -- that's what makes the
Table-3-lookup baseline in `experiments/exp08_transfer_c2.py` meaningful),
and the SAME node-attribute shape / self-loop / `inter_slice` bookkeeping
`build_attack_graph`'s own tail uses, so every generated graph compiles
through the EXISTING, unmodified `src/dbn/compiler.py::compile_to_2tbn`
and `src/dbn/parameterization.py::attach_cpds`.

NEVER TWIN-EXECUTED (LAB_NOTEBOOK.md 2026-08-05): `src/twin/runner.py`'s
physical/comms side-effect dispatch is hardcoded by literal node name,
specific to the one paper graph. Each generated `attack_step` node
therefore carries its own SYNTHETIC `asset_context`/`defensive_posture`/
`attacker_capability` (sampled, documented as such) and a ground-truth
`ttc` computed via the SAME closed-form multiplicative mechanism newly
instrumented into `src/twin/attacker.py`
(`table3_ttc[technique] * defensive_posture / attacker_capability`) --
deliberately consistent with how the amortized model's real twin-measured
training rows are generated, not a different, unexplained formula.

NO `reaction`-TYPE NODES ARE EVER CREATED. `build_reaction_cpt` requires
exactly one precondition parent; sidestepping that constraint entirely
keeps this generator simple and is not needed for the structural axes the
task asks to vary (depth, branching, technique mix, analytic coverage).
Only `attack_step` (timed), `analytic` (untimed evidence), and
`attack_step`/`goal`-shaped `gate`-typed merge/goal nodes are created --
all three CPT builders this touches (`build_attack_step_cpt`,
`build_analytic_cpt`, `build_gate_cpt`) are confirmed shape-agnostic.

VERIFIED PERFORMANCE CONSTRAINT (measured directly, not assumed): `pgmpy`'s
`VariableElimination` (the exact inference engine `DBNInference` uses,
`src/dbn/inference.py`) has no bound on elimination-order quality here, and
a graph with MULTIPLE root processes (`n_root_processes > 1`) that EACH
also branch internally (`branching_factor > 1`) creates a wide enough
moral-graph clique at the final goal node to blow up combinatorially --
measured: `depth=3, branching_factor=2, n_root_processes=2` (26 nodes) took
4.8s for `T=2`; `depth=4` at the same branching/process counts OOM-killed
the process outright. Isolated by sweeping each axis independently:
`n_root_processes` alone (branching_factor=1) and `branching_factor` alone
(n_root_processes=1) both stay under 0.03s even at larger sizes -- it is
specifically the COMBINATION that is pathological. Rather than tune
`pgmpy`'s elimination order (out of scope -- CLAUDE.md's own "do not sink
weeks into... clustering optimization" applies equally to the inference
engine's internals, which this project treats as a fixed dependency), this
generator caps `branching_factor` to 1 whenever the sampled
`n_root_processes > 1` (and vice versa is not needed since
`branching_factor` is sampled first) -- see `generate_family`. This is a
real, measured constraint stated here, not a silently-chosen small number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from src.attack_graph.graph import GateType, technique_table3_ttc
from src.parameterization.amortized import known_techniques

Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class FamilyGraphSpec:
    graph_id: str
    depth: int
    branching_factor: int
    n_root_processes: int
    analytic_coverage: float
    gate_type: GateType
    seed: int


@dataclass(frozen=True)
class FamilyGeneratorConfig:
    n_graphs: int
    n_train: int
    n_val: int
    n_test: int
    depth_range: tuple[int, int]
    branching_factor_range: tuple[int, int]
    n_root_processes_range: tuple[int, int]
    analytic_coverage_range: tuple[float, float]
    asset_context_range: tuple[float, float]
    defensive_posture_range: tuple[float, float]
    attacker_capability_range: tuple[float, float]

    def __post_init__(self) -> None:
        if self.n_train + self.n_val + self.n_test != self.n_graphs:
            raise ValueError(
                f"n_train({self.n_train}) + n_val({self.n_val}) + n_test({self.n_test}) "
                f"!= n_graphs({self.n_graphs})"
            )


@dataclass(frozen=True)
class FamilyGraph:
    graph_id: str
    split: Split
    ag: nx.DiGraph
    spec: FamilyGraphSpec


def _add_attack_step(
    g: nx.DiGraph,
    name: str,
    *,
    techniques: tuple[str, ...],
    table3: dict[str, float],
    config: FamilyGeneratorConfig,
    analytic_coverage: float,
    rng: np.random.Generator,
) -> None:
    technique = techniques[int(rng.integers(0, len(techniques)))]
    asset_context = float(rng.uniform(*config.asset_context_range))
    defensive_posture = float(rng.uniform(*config.defensive_posture_range))
    attacker_capability = float(rng.uniform(*config.attacker_capability_range))
    # Ground-truth ttc: the SAME closed form src/twin/attacker.py now
    # applies (module docstring) -- never a different, unexplained formula.
    ttc = table3[technique] * defensive_posture / attacker_capability
    g.add_node(
        name, name=name, node_type="attack_step", ttc=ttc, self_loop=True, gate=None,
        mitre_matrix=None, mitre_tactic=None, mitre_technique=technique, mitre_technique_id=None,
        asset_context=asset_context, defensive_posture=defensive_posture,
        attacker_capability=attacker_capability,
    )
    if rng.random() < analytic_coverage:
        analytic_name = f"{name}__Evidence"
        g.add_node(
            analytic_name, name=analytic_name, node_type="analytic", ttc=None, self_loop=False, gate=None,
            mitre_matrix=None, mitre_tactic=None, mitre_technique=None, mitre_technique_id=None,
            sensor_model=None, observable_kind="cyber",
        )
        g.add_edge(name, analytic_name, edge_type="triggers_analytic")


def _add_gate_node(g: nx.DiGraph, name: str, gate: GateType, node_type: str) -> None:
    g.add_node(
        name, name=name, node_type=node_type, ttc=None, self_loop=False, gate=gate,
        mitre_matrix=None, mitre_tactic=None, mitre_technique=None, mitre_technique_id=None,
    )


def _build_graph(
    spec: FamilyGraphSpec, config: FamilyGeneratorConfig, rng: np.random.Generator,
) -> nx.DiGraph:
    techniques = known_techniques()
    table3 = technique_table3_ttc()
    g = nx.DiGraph()

    process_outputs: list[str] = []
    for p in range(spec.n_root_processes):
        subchain_tails: list[str] = []
        for b in range(spec.branching_factor):
            prev: str | None = None
            for d in range(spec.depth):
                node_name = f"{spec.graph_id}_p{p}_b{b}_s{d}"
                _add_attack_step(
                    g, node_name, techniques=techniques, table3=table3, config=config,
                    analytic_coverage=spec.analytic_coverage, rng=rng,
                )
                if prev is not None:
                    g.add_edge(prev, node_name, edge_type="precondition")
                prev = node_name
            subchain_tails.append(prev)  # type: ignore[arg-type]

        if spec.branching_factor > 1:
            # Mirrors CredAccess exactly: node_type="attack_step", ttc=None,
            # self_loop=False, gate="AND" -- an AND-merge over this
            # process's parallel sub-chains.
            merge_name = f"{spec.graph_id}_p{p}_merge"
            _add_gate_node(g, merge_name, "AND", "attack_step")
            for tail in subchain_tails:
                g.add_edge(tail, merge_name, edge_type="precondition")
            process_outputs.append(merge_name)
        else:
            process_outputs.append(subchain_tails[0])

    # Mirrors UnstablePS exactly: node_type="goal", gate=spec.gate_type.
    goal_name = f"{spec.graph_id}_goal"
    _add_gate_node(g, goal_name, spec.gate_type, "goal")
    for out in process_outputs:
        g.add_edge(out, goal_name, edge_type="precondition")

    # Tail copied verbatim from build_attack_graph: self-loop edges for
    # persistent nodes, then derived inter_slice on every edge.
    for name, attrs in g.nodes(data=True):
        if attrs["self_loop"]:
            g.add_edge(name, name, edge_type="precondition")
    for u, v, data in g.edges(data=True):
        data["inter_slice"] = g.nodes[u]["self_loop"] and g.nodes[v]["self_loop"]

    return g


def generate_family(
    config: FamilyGeneratorConfig, seed_sequence: np.random.SeedSequence,
) -> list[FamilyGraph]:
    """`config.n_graphs` graphs, split `n_train`/`n_val`/`n_test` in that
    fixed order (first `n_train` generated graphs are "train", etc.) --
    deterministic given `seed_sequence` (each graph gets its own
    independently-spawned child seed, never ad hoc reseeding)."""
    child_seeds = seed_sequence.spawn(config.n_graphs)
    splits: list[Split] = (
        ["train"] * config.n_train + ["val"] * config.n_val + ["test"] * config.n_test
    )
    out: list[FamilyGraph] = []
    for i, (seed, split) in enumerate(zip(child_seeds, splits)):
        rng = np.random.default_rng(seed)
        graph_id = f"family_{i:03d}"
        depth = int(rng.integers(config.depth_range[0], config.depth_range[1] + 1))
        branching_factor = int(rng.integers(config.branching_factor_range[0], config.branching_factor_range[1] + 1))
        n_root_processes = int(rng.integers(config.n_root_processes_range[0], config.n_root_processes_range[1] + 1))
        if n_root_processes > 1 and branching_factor > 1:
            # Verified performance constraint (module docstring): capping
            # here, not silently choosing smaller ranges overall, so the
            # single-axis-varied cases still reach the full configured range.
            branching_factor = 1
        analytic_coverage = float(rng.uniform(*config.analytic_coverage_range))
        gate_type: GateType = str(rng.choice(["AND", "OR"]))  # type: ignore[assignment]
        spec = FamilyGraphSpec(
            graph_id=graph_id, depth=depth, branching_factor=branching_factor,
            n_root_processes=n_root_processes, analytic_coverage=analytic_coverage,
            gate_type=gate_type, seed=int(seed.generate_state(1)[0]),
        )
        ag = _build_graph(spec, config, rng)
        out.append(FamilyGraph(graph_id=graph_id, split=split, ag=ag, spec=spec))
    return out


def family_graph_rows(graphs: Sequence[FamilyGraph]) -> pd.DataFrame:
    """Every timed node across `graphs`, flattened to one row per node --
    columns `technique, asset_context, defensive_posture,
    attacker_capability, true_ttc` match
    `src.parameterization.amortized.fit_ttc_amortized_model`'s expected
    schema directly (plus `graph_id, split, node` provenance, which that
    function ignores)."""
    rows = []
    for fg in graphs:
        for node, data in fg.ag.nodes(data=True):
            if data.get("ttc") is None:
                continue
            rows.append({
                "graph_id": fg.graph_id, "split": fg.split, "node": node,
                "technique": data["mitre_technique"],
                "asset_context": data["asset_context"],
                "defensive_posture": data["defensive_posture"],
                "attacker_capability": data["attacker_capability"],
                "true_ttc": data["ttc"],
            })
    return pd.DataFrame(rows)
