"""The cyber-physical asset graph (CLAUDE.md layer [1]).

Static structure + static features only. Per-slice dynamic telemetry
(voltage, messages, ...) is layered on separately by `src/perception/features.py`
and concatenated with this module's static per-node vectors at model time --
this module never touches a `SliceRecord` or a `ContinuousTrace`.

Why the GNN lives HERE and not on the ~20-node attack graph (CLAUDE.md lists
"putting the GNN on the attack graph" among rejected directions -- these are
the actual reasons, not a restatement of the ban):

1. DIVISION OF LABOR. The attack graph's structure is already consumed: it is
   compiled to the 2TBN by `src/dbn/compiler.py`, and its conditional
   structure IS the DBN's job (causal fusion). A GNN over it would learn the
   fusion the DBN already performs exactly and explainably, violating
   CLAUDE.md's "neither is asked to do the other's job."
2. NO VARYING INPUT. The attack graph is 23-25 nodes carrying one bit each,
   with topology identical across every run and every slice. A GNN over it
   is a 23-row lookup table.
3. THE PERCEPTION PROBLEM IS ACTUALLY RELATIONAL HERE. Mapping a 33-dim
   voltage vector to LOCALIZED vs WIDESPREAD (PhysLocalDER/PhysWideArea) is a
   question about which buses are electrically adjacent to which -- that
   adjacency exists only on THIS graph. On the in-service case33bw tree
   (verified: `networkx.is_tree` True, diameter 20) the two DER buses sit
   exactly the tree's diameter apart, so this is a genuinely non-local
   relational problem, not a lookup.
4. TRANSFER (claim C2). This graph is derived from a pandapower net and an
   observed comms log; it rebuilds unchanged for a different feeder. The
   attack graph does not transfer to a different attack graph by
   construction (it names specific MITRE techniques).

The diameter-20 fact above has one more consequence, stated here so it is not
discovered later as an excuse: `PhysWideArea` needs bus-17 and bus-32 state
simultaneously, 20 hops apart electrically, but only 3 hops apart via the
CYBER overlay (`bus -> DER -> IED -> host`, verified by
`test_hops_from_der_bus_to_host_equals_n_gnn_layers`). This is WHY
`src/perception/encoder.py` uses exactly 3 HGT layers, and why a clean pass on
the physical positive-control targets validates the readout and this cyber
shortcut, not the electrical convolutions in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import networkx as nx
import pandapower.networks as pn
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

from src.twin.comms import DeliveryRecord
from src.twin.grid import GridModel

DECLARED_NODE_TYPES: tuple[str, ...] = (
    "bus", "line", "transformer", "IED", "RTU", "DER", "relay", "host",
)
DECLARED_EDGE_TYPES: tuple[str, ...] = (
    "electrical_coupling", "network_reachability", "control_authority",
)

# Default static feature widths, used only for node types that are EMPTY on
# the current network (so the [0, F] tensor still has a defined width and a
# feeder that later adds e.g. a transformer works without a schema change).
_DEFAULT_FEATURE_DIM: Mapping[str, int] = {
    "bus": 6, "line": 5, "transformer": 4, "IED": 1, "RTU": 1,
    "DER": 3, "relay": 1, "host": 1,
}


@dataclass(frozen=True)
class CyberAsset:
    """One node of the cyber overlay, declared in config, never invented.

    `endpoint` must be a string that actually appears as a `Message.sender`
    or `.receiver` in a real twin run's `CommsBus.log` -- enforced by
    `build_asset_graph`, not merely documented (the fabrication guard).
    """

    asset_id: str
    node_type: str  # must be in DECLARED_NODE_TYPES
    endpoint: str
    controls: tuple[str, ...] = ()  # DER ids (asset_ids) this asset commands


@dataclass(frozen=True)
class CyberOverlayConfig:
    assets: tuple[CyberAsset, ...]
    source: str  # provenance; required non-empty, mirrors SensorModel

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("CyberOverlayConfig.source must be non-empty provenance")
        unknown_types = {a.node_type for a in self.assets} - set(DECLARED_NODE_TYPES)
        if unknown_types:
            raise ValueError(f"undeclared node types in overlay: {unknown_types}")


@dataclass(frozen=True)
class AssetGraph:
    data: HeteroData
    node_index: Mapping[str, Mapping[str, int]]  # node_type -> asset_id -> row
    empty_node_types: tuple[str, ...]
    counts: Mapping[str, int]
    edge_counts: Mapping[tuple[str, str, str], int]
    provenance: Mapping[str, str]
    graph: nx.Graph  # bus-level in-service electrical adjacency (for topology tests)


def observed_endpoints(log: Sequence[DeliveryRecord]) -> frozenset[str]:
    """Every endpoint that actually sent or received a message in a real
    twin run -- the observational half of the fabrication guard (the other
    half is "is a pandapower row")."""
    endpoints: set[str] = set()
    for record in log:
        endpoints.add(record.sent.sender)
        endpoints.add(record.sent.receiver)
        if record.delivered is not None:
            endpoints.add(record.delivered.sender)
            endpoints.add(record.delivered.receiver)
    return frozenset(endpoints)


def _line_graph(net) -> nx.Graph:
    """In-service electrical adjacency, bus-indexed. Mirrors
    `src.twin.grid.select_der_buses`'s own construction exactly (same
    in-service filter, same z_ohm weight) so the two never disagree about
    what "electrically adjacent" means."""
    graph = nx.Graph()
    graph.add_nodes_from(int(b) for b in net.bus.index)
    in_service = net.line[net.line.in_service]
    for _, line in in_service.iterrows():
        z_ohm = (line.r_ohm_per_km**2 + line.x_ohm_per_km**2) ** 0.5 * line.length_km
        graph.add_edge(int(line.from_bus), int(line.to_bus), z_ohm=z_ohm)
    return graph


def _bus_static_features(net, grid: GridModel, dijkstra_dist: Mapping[int, float]) -> torch.Tensor:
    """[degree, z_dist_from_slack (normalized), is_slack, has_load,
    load_p_mw, load_q_mvar] per bus, in bus-index order."""
    slack = int(net.ext_grid.bus.iloc[0])
    der_buses = set(grid.der_buses)
    load_by_bus: dict[int, tuple[float, float]] = {}
    for _, row in net.load.iterrows():
        b = int(row.bus)
        p, q = load_by_bus.get(b, (0.0, 0.0))
        load_by_bus[b] = (p + float(row.p_mw), q + float(row.q_mvar))

    in_service = net.line[net.line.in_service]
    degree: dict[int, int] = {int(b): 0 for b in net.bus.index}
    for _, line in in_service.iterrows():
        degree[int(line.from_bus)] += 1
        degree[int(line.to_bus)] += 1

    max_dist = max(dijkstra_dist.values()) if dijkstra_dist else 1.0
    rows = []
    for b in net.bus.index:
        b = int(b)
        p_mw, q_mvar = load_by_bus.get(b, (0.0, 0.0))
        rows.append([
            float(degree.get(b, 0)),
            float(dijkstra_dist.get(b, 0.0)) / max(max_dist, 1e-9),
            1.0 if b == slack else 0.0,
            1.0 if b in load_by_bus else 0.0,
            p_mw,
            q_mvar,
        ])
    del der_buses  # kept for symmetry with DER features; bus features stay
    # feeder-topology-only -- "is this bus a DER bus" is redundant with the
    # explicit bus<->DER control_authority/electrical_coupling edges below,
    # not duplicated as a feature bit.
    return torch.tensor(rows, dtype=torch.float32)


def _line_static_features(net) -> torch.Tensor:
    """[r_ohm_per_km, x_ohm_per_km, length_km, z_ohm, in_service] per line,
    in line-index order (both in- and out-of-service lines included -- an
    open tie switch is a real asset, not deleted)."""
    rows = []
    for _, line in net.line.iterrows():
        z_ohm = (line.r_ohm_per_km**2 + line.x_ohm_per_km**2) ** 0.5 * line.length_km
        rows.append([
            float(line.r_ohm_per_km), float(line.x_ohm_per_km),
            float(line.length_km), float(z_ohm), float(bool(line.in_service)),
        ])
    return torch.tensor(rows, dtype=torch.float32)


def build_asset_graph(
    grid: GridModel,
    overlay: CyberOverlayConfig,
    *,
    observed_endpoints_set: frozenset[str],
    feature_dims: Mapping[str, int] | None = None,
) -> AssetGraph:
    """Build the static asset graph.

    Governing rule (mechanically enforced, not just documented): an asset
    node exists iff it is (a) a row derived from the pandapower net, or (b) a
    `CyberAsset` whose `endpoint` actually appears in `observed_endpoints_set`
    (built from a real twin run's `CommsBus.log` via `observed_endpoints`).
    Raises if any declared cyber asset's endpoint was never observed --
    nothing is created because a config or a type list merely mentions it.

    Only `case33bw` is wired up (matches `src/twin/grid.py`'s own
    `_build_base_network` restriction) -- raises otherwise rather than
    silently building the wrong topology.
    """
    if grid.config.network != "case33bw":
        raise ValueError(
            f"only case33bw is wired up for the asset graph, got {grid.config.network!r}"
        )
    feature_dims = {**_DEFAULT_FEATURE_DIM, **(feature_dims or {})}

    unknown_endpoints = {a.endpoint for a in overlay.assets} - observed_endpoints_set
    if unknown_endpoints:
        raise ValueError(
            f"cyber asset endpoint(s) {sorted(unknown_endpoints)} never appeared "
            "in the observed twin comms log; an asset node must not be created "
            "for an endpoint that was never seen (CLAUDE.md rule 1)"
        )

    net = pn.case33bw()
    if len(net.bus) != 33 or int((~net.line.in_service).sum()) != 5:
        raise AssertionError(
            "pandapower.networks.case33bw() no longer matches the counts this "
            "module was verified against; re-derive rather than trust stale "
            "constants"
        )

    slack = int(net.ext_grid.bus.iloc[0])
    line_graph = _line_graph(net)
    dijkstra_dist = nx.single_source_dijkstra_path_length(line_graph, slack, weight="z_ohm")

    node_index: dict[str, dict[str, int]] = {t: {} for t in DECLARED_NODE_TYPES}
    counts: dict[str, int] = {}
    data = HeteroData()

    # --- bus, line: derived from the pandapower net -------------------------
    data["bus"].x = _bus_static_features(net, grid, dijkstra_dist)
    node_index["bus"] = {str(int(b)): i for i, b in enumerate(net.bus.index)}
    counts["bus"] = len(net.bus)

    data["line"].x = _line_static_features(net)
    node_index["line"] = {str(i): i for i in range(len(net.line))}
    counts["line"] = len(net.line)

    data["transformer"].x = torch.zeros((0, feature_dims["transformer"]), dtype=torch.float32)
    counts["transformer"] = 0

    # --- DER: derived from GridModel, never a literal bus number -----------
    node_index["DER"] = {der_id: i for i, der_id in enumerate(grid.der_ids)}
    counts["DER"] = len(grid.der_ids)
    der_rows = []
    for der_id, bus in zip(grid.der_ids, grid.der_buses):
        der_rows.append([1.0, float(dijkstra_dist.get(bus, 0.0)) / max(max(dijkstra_dist.values(), default=1.0), 1e-9), float(bus)])
    data["DER"].x = torch.tensor(der_rows, dtype=torch.float32) if der_rows else torch.zeros(
        (0, feature_dims["DER"]), dtype=torch.float32
    )

    # --- cyber overlay: IED / RTU / relay / host, from CyberOverlayConfig --
    for node_type in ("IED", "RTU", "relay", "host"):
        assets = [a for a in overlay.assets if a.node_type == node_type]
        node_index[node_type] = {a.asset_id: i for i, a in enumerate(assets)}
        counts[node_type] = len(assets)
        width = feature_dims[node_type]
        data[node_type].x = (
            torch.ones((len(assets), width), dtype=torch.float32)
            if assets else torch.zeros((0, width), dtype=torch.float32)
        )

    empty_node_types = tuple(t for t in DECLARED_NODE_TYPES if counts[t] == 0)

    # --- edges ---------------------------------------------------------------
    edge_counts: dict[tuple[str, str, str], int] = {}

    def _add_edges(src_type: str, rel: str, dst_type: str, pairs: list[tuple[int, int]]) -> None:
        key = (src_type, rel, dst_type)
        if pairs:
            data[key].edge_index = torch.tensor(pairs, dtype=torch.long).T.contiguous()
        else:
            data[key].edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_counts[key] = len(pairs)

    # electrical_coupling: bus<->bus (in-service lines), bus<->line (all lines,
    # including open ties -- a tie switch electrically touches both its
    # endpoint buses whether or not it is currently closed), bus<->DER.
    bus_bus = [(u, v) for u, v in line_graph.edges()]
    _add_edges("bus", "electrical_coupling", "bus", bus_bus)

    bus_line = []
    for i, (_, line) in enumerate(net.line.iterrows()):
        bus_line.append((int(line.from_bus), i))
        bus_line.append((int(line.to_bus), i))
    _add_edges("bus", "electrical_coupling", "line", bus_line)

    bus_der = [
        (node_index["bus"][str(bus)], node_index["DER"][der_id])
        for der_id, bus in zip(grid.der_ids, grid.der_buses)
    ]
    _add_edges("bus", "electrical_coupling", "DER", bus_der)

    # network_reachability: host <-> IED (every host/IED pair declared with
    # a plausible reachability relation -- in this twin's 3-endpoint universe
    # that is every (host, IED) pair, since the control centre polls and
    # commands every DER's IED).
    hosts = list(node_index["host"])
    ieds = list(node_index["IED"])
    host_ied = [
        (node_index["host"][h], node_index["IED"][i]) for h in hosts for i in ieds
    ]
    _add_edges("host", "network_reachability", "IED", host_ied)

    # control_authority: IED->DER only, from CyberAsset.controls.
    #
    # Deliberately NOT host->DER, despite the twin's own comms model routing
    # SETPOINT messages directly from "ControlCentre" to a DER's endpoint
    # string (src/twin/comms.py) with no separate IED name in transit. That
    # is a simulation-plumbing simplification, not the real control
    # architecture: a real ADMS commands a field device through the site RTU
    # /IED, never with a direct wire from the control centre to the breaker.
    # `host_reaches_der_only_through_ied` (below) enforces this design
    # choice matters structurally, not just semantically -- a host->DER edge
    # would create a 2-hop bus->DER->host shortcut (verified) that silently
    # defeats the 3-hop cyber-shortcut claim this module's docstring makes
    # and that `n_layers=3` in encoder.py depends on.
    ied_der = []
    for asset in overlay.assets:
        for der_id in asset.controls:
            if der_id not in node_index["DER"]:
                raise ValueError(
                    f"asset {asset.asset_id!r} declares control over unknown DER {der_id!r}"
                )
            if asset.node_type != "IED":
                raise ValueError(
                    f"control_authority is only defined for IED assets (a host "
                    f"reaches a DER via network_reachability -> IED -> "
                    f"control_authority -> DER, never directly); got "
                    f"node_type={asset.node_type!r} for {asset.asset_id!r}"
                )
            ied_der.append((node_index["IED"][asset.asset_id], node_index["DER"][der_id]))
    _add_edges("IED", "control_authority", "DER", ied_der)

    data = ToUndirected()(data)

    # nx graph over bus indices only, for topology tests (hops etc. use the
    # richer full-graph helper `hops_between` below instead).
    return AssetGraph(
        data=data,
        node_index=node_index,
        empty_node_types=empty_node_types,
        counts=counts,
        edge_counts=edge_counts,
        provenance={"cyber_overlay_source": overlay.source, "network": "case33bw"},
        graph=line_graph,
    )


def nonempty_metadata(data: HeteroData) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Metadata restricted to node types with >= 1 row.

    HGTConv raises `IndexError` on a node type with 0 rows (verified,
    torch_geometric 2.8.0.post1) and silently drops a node type with only
    OUTGOING edges from its output dict (also verified) -- callers must
    construct their convolution layer with THIS, not `data.metadata()`.

    This alone is not sufficient at forward-call time: HGTConv also KeyErrors
    if the `x_dict` / `edge_index_dict` handed to it at call time contain
    keys outside its constructor metadata (verified). Use `filtered_input`
    below to build the matching x_dict/edge_index_dict for the SAME call.

    `data` itself is unchanged by either function: the empty types and their
    zero counts stay visible for reporting (CLAUDE.md rule 1 -- a zero must
    be printed, never silently dropped).
    """
    node_types = [t for t in data.node_types if data[t].num_nodes > 0]
    node_set = set(node_types)
    edge_types = [
        et for et in data.edge_types if et[0] in node_set and et[2] in node_set
    ]
    return node_types, edge_types


def filtered_input(
    data: HeteroData, node_types: Sequence[str], edge_types: Sequence[tuple[str, str, str]]
) -> tuple[dict[str, torch.Tensor], dict[tuple[str, str, str], torch.Tensor]]:
    """`(x_dict, edge_index_dict)` restricted to `node_types`/`edge_types` --
    the companion to `nonempty_metadata`, required at every HGTConv forward
    call (see that function's docstring for why passing `data.x_dict`
    unfiltered raises `KeyError`)."""
    x_dict = {t: data[t].x for t in node_types}
    edge_index_dict = {et: data[et].edge_index for et in edge_types}
    return x_dict, edge_index_dict


def hops_between(data: HeteroData, source: tuple[str, str], target: tuple[str, str]) -> int:
    """Shortest path length (in relation hops) between two typed nodes,
    treating every relation triple in `data` (already `ToUndirected`-expanded)
    as an unweighted directed edge. `source`/`target` are (node_type, asset_key).

    Used to pin the load-bearing fact that a DER's bus is exactly 3 hops from
    `host` via the cyber overlay, independent of the 20-hop electrical
    distance -- see this module's docstring.
    """
    g = nx.DiGraph()
    for et in data.edge_types:
        src_t, _, dst_t = et
        ei = data[et].edge_index
        for u, v in ei.T.tolist():
            g.add_edge((src_t, u), (dst_t, v))
    return nx.shortest_path_length(g, source, target)
