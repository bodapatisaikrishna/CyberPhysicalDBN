"""Forward (generative) sampling of a 2TBN attack-graph trajectory (CLAUDE.md
layer [3], Session 8, claim C2).

`DBNInference` (src/dbn/inference.py) is inference-ONLY -- given evidence, it
computes a posterior. Nothing anywhere in this project generates a stochastic
GROUND-TRUTH trajectory from an attack graph's CPTs. This module fills that
gap so `experiments/exp08_transfer_c2.py` can build a "true" attack
progression for a synthetic (never twin-executed) held-out graph, then score
three candidate TTC-parameterization arms against it via DBN inference --
without ever reimplementing Eq. 3: `forward_sample_trajectory` computes p_s
via `src.dbn.parameterization.collect_uniformization_ttcs`/`compute_delta_t`/
`compute_ps` (the same public functions `build_attack_step_cpt` calls), and
resolves gates/reactions with the SAME precedence rules
`build_attack_step_cpt`/`build_gate_cpt`/`build_reaction_cpt` encode in
tabular form -- this module only samples from those rules, it does not
restate them differently.

SCOPE: `latch`-type nodes (reaction_mode="latched") are not supported --
raises `NotImplementedError`. Neither the fixed paper graph under its
default `"memoryless"` reaction mode nor any Session-8 family graph
(which never creates `reaction`/`latch` nodes at all, per LAB_NOTEBOOK.md
2026-08-05) needs it.

INITIAL CONDITION: Eq. 3's CPTs have no anterior-layer prior (deliberately --
see `attach_cpds`'s docstring, CLAUDE.md rule 1). Forward sampling needs one
to start from; this module's own choice, stated here rather than buried in
code, is that every self-loop (persistent) node starts INACTIVE at slice 0 --
"nothing has happened yet before the simulation begins", mirroring the
twin's own `ScriptedAttacker` starting with an empty `completion_times`.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from src.dbn.parameterization import (
    analytic_error_rates,
    collect_uniformization_ttcs,
    compute_delta_t,
    compute_ps,
    precondition_parents,
)


def _topo_order(ag: nx.DiGraph) -> list[str]:
    """Topological order over precondition edges, self-loops excluded (a
    self-loop is a length-1 cycle and would otherwise make the graph
    infeasible to sort) -- mirrors `precondition_parents`'s own exclusion."""
    g = nx.DiGraph()
    g.add_nodes_from(ag.nodes())
    for u, v, data in ag.edges(data=True):
        if u != v and data.get("edge_type") == "precondition":
            g.add_edge(u, v)
    return list(nx.topological_sort(g))


def _parent_value(
    ag: nx.DiGraph, parent: str, node: str, current: dict[str, int], previous: dict[str, int],
) -> int:
    """The value a parent contributes to `node`'s activation rule this
    slice: ANTERIOR (previous slice) if the edge is `inter_slice`, else
    ULTERIOR (this slice, already resolved by topological order) -- exactly
    `_parent_slice`'s convention in `src/dbn/parameterization.py`."""
    if ag.edges[parent, node]["inter_slice"]:
        return previous.get(parent, 0)
    return current[parent]


def forward_sample_trajectory(
    ag: nx.DiGraph, m: float, n_slices: int, rng: np.random.Generator,
) -> tuple[dict[str, list[int]], float]:
    """Per-slice 0/1 ground truth for every non-analytic node (`attack_step`,
    `reaction`, `goal`/gate), replaying `build_attack_step_cpt`'s exact
    precedence (precondition unmet -> 0; else already-active -> persists;
    else `Bernoulli(p_s)`), `build_gate_cpt`'s `combine=all/any`, and
    `build_reaction_cpt`'s "no persistence, resolves within the slice" rule.
    `p_s` is computed via `collect_uniformization_ttcs(ag)`/
    `compute_delta_t(ttcs, m)`/`compute_ps` -- called on `ag` AS GIVEN, so
    passing the oracle (ground-truth `ttc`) graph samples the TRUE
    trajectory; this function never mutates or re-derives `ttc` itself.

    Returns `(trajectory, delta_t)`. `trajectory[node]` is a length-`n_slices`
    list of `int` (0/1), index `i` = slice `i+1` (1-based elsewhere in this
    project, e.g. `features.slice_of`).
    """
    ttcs = collect_uniformization_ttcs(ag)
    delta_t = compute_delta_t(ttcs, m)

    order = _topo_order(ag)
    trajectory: dict[str, list[int]] = {n: [] for n in order if ag.nodes[n]["node_type"] != "analytic"}

    previous: dict[str, int] = {n: 0 for n in trajectory}
    for _slice in range(n_slices):
        current: dict[str, int] = {}
        for node in order:
            data = ag.nodes[node]
            if data["node_type"] == "analytic":
                continue
            if data["node_type"] == "latch":
                raise NotImplementedError(
                    f"forward_sample_trajectory does not support latch nodes (node={node!r}); "
                    "reaction_mode='latched' graphs are out of scope for Session 8 (see module docstring)"
                )

            parents = precondition_parents(ag, node)
            parent_vals = [_parent_value(ag, p, node, current, previous) for p in parents]
            precondition_met = any(parent_vals) if parents else True

            if data["gate"] is not None:
                combine = all if data["gate"] == "AND" else any
                active = int(combine(parent_vals) if parents else False)
            elif data["node_type"] == "reaction":
                if "latch" in data:
                    raise NotImplementedError(
                        f"forward_sample_trajectory does not support latched reactions (node={node!r})"
                    )
                active = int(precondition_met and rng.random() < float(data["fixed_success_prob"]))
            else:  # attack_step, self_loop=True, ttc is not None
                prev_self = previous.get(node, 0)
                if not precondition_met:
                    active = 0
                elif prev_self:
                    active = 1
                else:
                    p_s = compute_ps(float(data["ttc"]), delta_t)
                    active = int(rng.random() < p_s)

            current[node] = active
            trajectory[node].append(active)

        previous = current

    return trajectory, delta_t


def forward_sample_evidence_stream(
    ag: nx.DiGraph,
    trajectory: dict[str, list[int]],
    p_pos: float,
    p_neg: float,
    rng: np.random.Generator,
) -> dict[int, dict[str, int]]:
    """Analytic 0/1 emissions from ground truth, the discrete analog of
    `runner.discretize()`'s cyber-analytic sampling block for a graph with
    no continuous trace to discretize. Same rule `build_analytic_cpt`
    encodes: `P(observed=1 | parent=1) = 1-p_neg`, `P(observed=1 | parent=0)
    = p_pos`, using `analytic_error_rates` for any per-node `SensorModel`
    override exactly as the CPT builder does. `evidence[slice][node]` for
    every analytic node whose `triggers_analytic` parent is timed."""
    n_slices = len(next(iter(trajectory.values())))
    analytic_parent: dict[str, str] = {}
    for node, data in ag.nodes(data=True):
        if data["node_type"] != "analytic":
            continue
        parents = [
            u for u, v, d in ag.in_edges(node, data=True) if d.get("edge_type") == "triggers_analytic"
        ]
        if len(parents) == 1:
            analytic_parent[node] = parents[0]

    evidence: dict[int, dict[str, int]] = {}
    for i in range(n_slices):
        slice_index = i + 1
        row: dict[str, int] = {}
        for node, parent in analytic_parent.items():
            node_p_pos, node_p_neg = analytic_error_rates(ag, node, p_pos, p_neg)
            parent_active = trajectory[parent][i]
            p_observed_1 = (1.0 - node_p_neg) if parent_active else node_p_pos
            row[node] = int(rng.random() < p_observed_1)
        evidence[slice_index] = row
    return evidence


def validate_slice_trajectory(trajectory: dict[str, list[int]], ag: nx.DiGraph) -> list[str]:
    """Discrete analog of `src.twin.runner.validate_trace`: every violation
    found, as a human-readable string (empty list = valid). Checks (1)
    self-loop (persistent) nodes never revert 1->0, (2) a node is never
    active in a slice where its precondition (respecting the same
    ANTERIOR/ULTERIOR layer rule the sampler itself used) was not met."""
    violations: list[str] = []
    n_slices = len(next(iter(trajectory.values())))

    for node, series in trajectory.items():
        if ag.nodes[node]["self_loop"]:
            for i in range(1, n_slices):
                if series[i - 1] == 1 and series[i] == 0:
                    violations.append(f"{node} reverted from active to inactive at slice {i + 1}")

    for i in range(n_slices):
        previous = {n: (trajectory[n][i - 1] if i > 0 else 0) for n in trajectory}
        current = {n: trajectory[n][i] for n in trajectory}
        for node in trajectory:
            data = ag.nodes[node]
            if data["node_type"] == "latch":
                continue
            parents = precondition_parents(ag, node)
            if not parents:
                continue
            parent_vals = [_parent_value(ag, p, node, current, previous) for p in parents]
            precondition_met = any(parent_vals)
            was_active_before = previous.get(node, 0) == 1
            if current[node] == 1 and not precondition_met and not was_active_before:
                violations.append(f"{node} active at slice {i + 1} with no precondition met and no prior persistence")

    return violations
