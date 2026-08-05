"""GAT/GraphSAGE end-to-end binary classifier over the asset graph
(CLAUDE.md's `src/baselines/`: "GAT classifier"). Predicts `grid_unstable`
directly per slice, supervised, unlike `src/perception/encoder.py`'s
`PerceptionEncoder` (which predicts 4 intermediate analytic targets, not
instability itself).

Reuses `src/perception/encoder.py`'s conv-agnostic plumbing directly rather
than reimplementing it: `flatten_for_spatial`/`replicate_static_graph`/
`unflatten_from_spatial` (the exact-block-diagonal-replication trick) and
`Readout` (host + mean(IED) + mean(DER) + mean(bus) + max(bus) + globals).
Only the spatial convolution (GAT/SAGE instead of HGT) and the temporal head
(one short causal conv instead of a 5-block dilated TCN) are new.

VERIFIED before writing this module: `GATConv`'s default `add_self_loops=True`
RAISES `ValueError` on a heterogeneous (bipartite) edge type inside
`HeteroConv` ("This will lead to incorrect message passing results");
`add_self_loops=False` is required and was confirmed to produce correct
shapes by direct repro. Also verified: `GATConv` needs the TUPLE form
`(-1, -1)` for lazy in_channels on a bipartite edge (a bare `-1` silently
produces a shape mismatch at the first forward call); `SAGEConv` accepts
`(-1, -1)` natively. `GATConv(..., concat=False)` is used so its output
width is exactly `hidden` regardless of `heads`, matching `SAGEConv`'s
output width -- so the rest of the architecture (LayerNorm, residual,
`Readout`) never needs to know which conv type produced its input.

TEMPORAL-HEAD DECISION, explicit: neither the full 5-block dilated TCN
(would make this baseline architecturally a near-clone of the proposed
system's perception stage) nor zero temporal memory (would unfairly
penalize a baseline compared against a system whose claim partly rests on
temporal fusion). Middle ground: ONE short causal 1-D conv, receptive field
= `temporal_kernel_size` (5-17 slices, itself searched), a small fraction of
the DBN perception TCN's 63-slice receptive field. This asymmetry is stated
in `configs/baselines.yaml`'s comments and the exp06 report's interpretation
notes -- not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv

from src.perception.encoder import (
    CausalConv1d,
    Readout,
    flatten_for_spatial,
    replicate_static_graph,
    unflatten_from_spatial,
)


@dataclass(frozen=True)
class GNNBaselineConfig:
    conv_type: Literal["gat", "sage"]
    n_layers: int
    hidden: int
    heads: int = 4  # GAT-only
    dropout: float = 0.1
    sage_aggr: str = "mean"  # SAGE-only
    temporal_kernel_size: int = 9

    def __post_init__(self) -> None:
        if self.conv_type not in ("gat", "sage"):
            raise ValueError(f"conv_type must be 'gat' or 'sage', got {self.conv_type!r}")
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers}")
        if self.temporal_kernel_size < 1:
            raise ValueError(f"temporal_kernel_size must be >= 1, got {self.temporal_kernel_size}")


class HeteroSpatialLayer(nn.Module):
    """One `HeteroConv`-wrapped layer (GAT or SAGE per edge type) + GELU +
    per-type LayerNorm. Residual composition (only valid from the second
    layer onward, since layer 0's input width varies per node type) is
    handled by the caller (`GNNClassifier`), mirroring
    `src/perception/encoder.py::SpatialEncoder`'s exact pattern."""

    def __init__(
        self,
        node_types: tuple[str, ...],
        edge_types: tuple[tuple[str, str, str], ...],
        config: GNNBaselineConfig,
    ):
        super().__init__()
        self.node_types = list(node_types)
        conv_dict: dict[tuple[str, str, str], nn.Module] = {}
        for edge_type in edge_types:
            if config.conv_type == "gat":
                conv_dict[edge_type] = GATConv(
                    (-1, -1), config.hidden, heads=config.heads,
                    dropout=config.dropout, add_self_loops=False, concat=False,
                )
            else:
                conv_dict[edge_type] = SAGEConv((-1, -1), config.hidden, aggr=config.sage_aggr)
        self.conv = HeteroConv(conv_dict, aggr="sum")
        self.norms = nn.ModuleDict({t: nn.LayerNorm(config.hidden) for t in node_types})

    def forward(
        self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict[tuple[str, str, str], torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        out = self.conv(x_dict, edge_index_dict)
        result = {}
        for t in self.node_types:
            o = out.get(t)
            if o is None:
                # A type with no incoming edges in this layer's relation set
                # (shouldn't happen for bus/IED/host/DER given ToUndirected,
                # but handled rather than KeyError-ing) keeps its input.
                o = x_dict[t]
            result[t] = self.norms[t](F.gelu(o))
        return result


class GNNClassifier(nn.Module):
    """spatial (`HeteroSpatialLayer` x n_layers, with residual from layer 1
    on) -> `Readout` -> one short causal conv -> 1x1 conv head -> logits
    `[B, S]`. Trained supervised on `grid_unstable` directly (BCE +
    `pos_weight`), unlike `PerceptionEncoder`'s 4 masked intermediate
    targets."""

    def __init__(
        self,
        node_types: tuple[str, ...],
        edge_types: tuple[tuple[str, str, str], ...],
        config: GNNBaselineConfig,
    ):
        super().__init__()
        self.node_types = list(node_types)
        self.config = config
        self.spatial_layers = nn.ModuleList(
            [HeteroSpatialLayer(node_types, edge_types, config) for _ in range(config.n_layers)]
        )
        self.readout = Readout(config.hidden)
        self.temporal = CausalConv1d(config.hidden, config.hidden, config.temporal_kernel_size, dilation=1)
        self.head = nn.Conv1d(config.hidden, 1, kernel_size=1)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        globals_: torch.Tensor,
    ) -> torch.Tensor:
        flat, num_nodes, b, s = flatten_for_spatial(x_dict)
        rep_edges = replicate_static_graph(edge_index_dict, num_nodes, b * s)

        h = flat
        for i, layer in enumerate(self.spatial_layers):
            out = layer(h, rep_edges)
            if i > 0:
                out = {t: out[t] + h[t] for t in self.node_types}
            h = out

        h = unflatten_from_spatial(h, num_nodes, b, s)
        u = self.readout(h, globals_)  # [B, S, hidden]
        trunk = F.gelu(self.temporal(u.transpose(1, 2)))  # [B, hidden, S]
        return self.head(trunk).squeeze(1)  # [B, S]
