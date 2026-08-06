"""Heterogeneous GNN + causal temporal encoder (CLAUDE.md layer [1]).

Pipeline (see `src/perception/asset_graph.py` and `.../features.py` for the
inputs this consumes):

    x_dict[t]: [B, S, N_t, F_t]     static (asset_graph) + dynamic (features)
                                     features already concatenated per type
     |
     v  STAGE 1 -- spatial (SpatialEncoder)
    Topology is STATIC across slices and batch elements (only node FEATURES
    vary), so the slice/batch axes are folded into the node axis and the
    graph is block-diagonally replicated B*S times -- an EXACT operation
    (not an approximation of running the conv B*S times independently),
    verified by `test_static_graph_replication_equals_per_slice_loop`.
    N_GNN_LAYERS=3 heterogeneous graph transformer layers.
     |
     v  h_dict[t]: [B, S, N_t, HIDDEN]
     v  STAGE 2 -- typed readout (Readout)
    host + mean(IED) + mean(DER) + mean(bus) + max(bus) + globals -> u
     |
     v  u: [B, S, HIDDEN]
     v  STAGE 3 -- causal TCN (CausalTCN)
    5 residual blocks, kernel=3, dilations (1,2,4,8,16), LEFT-padded causal
    convolutions only. Receptive field = 63 slices (`receptive_field()`,
    pinned by test) ~= 17 dispatch periods -- comfortable margin over the
    ~4 rungs CommandCoherence's ladder-climb signature needs.
     |
     v  STAGE 4 -- 4 per-target 1x1 heads -> logits[target]: [B, S]

WHY N_GNN_LAYERS=3 IS LOAD-BEARING: `PhysWideArea` needs bus-17 and bus-32
state simultaneously, but they sit 20 hops apart on the in-service case33bw
tree (verified in asset_graph.py). The cyber overlay makes each DER's bus
exactly 3 hops from `host` (`bus -> DER -> IED -> host`,
`test_hops_from_der_bus_to_host_equals_n_gnn_layers`). Changing this constant
without re-verifying that hop count breaks the architecture's only route to
a wide-area signal.

WHY NO BatchNorm ANYWHERE: BatchNorm computes statistics across the batch
dimension at each call, which is fine architecturally but is also exactly
the kind of layer a future edit could misuse to leak information across the
slice axis if it were ever applied per-slice instead of per-batch -- a whole
class of causality bug this module avoids by construction, not by discipline.
`test_no_batchnorm_anywhere` is a mechanical guard, not a style preference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv

N_GNN_LAYERS = 3  # load-bearing; see module docstring. Must match
                   # tests/test_asset_graph.py::N_GNN_LAYERS.
HIDDEN_DIM = 64
GNN_HEADS = 4
TCN_KERNEL_SIZE = 3
TCN_DILATIONS: tuple[int, ...] = (1, 2, 4, 8, 16)
TCN_DROPOUT = 0.1
TARGETS: tuple[str, ...] = (
    "MeasureCoherence", "CommandCoherence", "PhysLocalDER", "PhysWideArea",
)
# Node types the readout always pools over -- fixed for this feeder (every
# one is guaranteed non-empty on case33bw: host=1, IED=2, DER=2, bus=33).
READOUT_TYPES: tuple[str, ...] = ("host", "IED", "DER", "bus")


def receptive_field(kernel_size: int = TCN_KERNEL_SIZE, dilations: Sequence[int] = TCN_DILATIONS) -> int:
    """1 + sum((kernel_size-1) * dilation) over one causal conv per dilation
    level. Hand-computed for the defaults: 1 + 2*(1+2+4+8+16) = 63."""
    return 1 + sum((kernel_size - 1) * d for d in dilations)


# --- static-topology replication (Stage 1 plumbing) -------------------------


def combine_static_dynamic(
    static_x: Mapping[str, torch.Tensor], dynamic_x: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """`static_x[t]: [N_t, F_static_t]` (from `AssetGraph.data[t].x`) +
    `dynamic_x[t]: [S, N_t, F_dynamic_t]` (from `build_dynamic_features`) ->
    `[S, N_t, F_static_t + F_dynamic_t]`, static features broadcast across
    every slice."""
    out = {}
    for t, dyn in dynamic_x.items():
        S = dyn.shape[0]
        stat = static_x[t].unsqueeze(0).expand(S, -1, -1)
        out[t] = torch.cat([stat, dyn], dim=-1)
    return out


def stack_scenarios(per_scenario: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """`[{"bus": [S,N,F], ...}, ...]` (one dict per scenario, all sharing the
    SAME topology/N since the asset graph is one feeder) -> `{"bus": [B,S,N,F],
    ...}`. Raises if scenarios disagree on N or F for any type (a topology
    mismatch would silently corrupt the block-diagonal graph replication)."""
    if not per_scenario:
        raise ValueError("per_scenario must not be empty")
    types = set(per_scenario[0])
    for i, scenario in enumerate(per_scenario[1:], start=1):
        if set(scenario) != types:
            raise ValueError(f"scenario {i} has a different node-type set than scenario 0")
    out = {}
    for t in types:
        shapes = {tuple(s[t].shape) for s in per_scenario}
        if len(shapes) != 1:
            raise ValueError(f"scenarios disagree on shape for node type {t!r}: {shapes}")
        out[t] = torch.stack([s[t] for s in per_scenario], dim=0)
    return out


def flatten_for_spatial(
    x_dict: Mapping[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, int], int, int]:
    """`[B,S,N_t,F_t]` -> `[B*S*N_t, F_t]`, folding batch+slice into the node
    axis so ONE call to each HGTConv layer processes every (batch, slice)
    simultaneously. Row order within each type is `(b, s, n)`, matching
    `replicate_static_graph`'s block ordering exactly -- the two must agree
    or node features and edges silently misalign."""
    any_t = next(iter(x_dict))
    B, S, _, _ = x_dict[any_t].shape
    flat, num_nodes = {}, {}
    for t, x in x_dict.items():
        b, s, n, f = x.shape
        if (b, s) != (B, S):
            raise ValueError(f"node type {t!r} has (B,S)=({b},{s}), expected ({B},{S})")
        num_nodes[t] = n
        flat[t] = x.reshape(b * s * n, f)
    return flat, num_nodes, B, S


def unflatten_from_spatial(
    h_dict: Mapping[str, torch.Tensor], num_nodes: Mapping[str, int], B: int, S: int
) -> dict[str, torch.Tensor]:
    out = {}
    for t, h in h_dict.items():
        n = num_nodes[t]
        d = h.shape[-1]
        out[t] = h.reshape(B, S, n, d)
    return out


def replicate_static_graph(
    edge_index_dict: Mapping[tuple[str, str, str], torch.Tensor],
    num_nodes: Mapping[str, int],
    n_repeats: int,
) -> dict[tuple[str, str, str], torch.Tensor]:
    """Block-diagonal replication of a topology identical across every
    (batch, slice) pair -- EXACT, not an approximation of running the conv
    n_repeats times independently: block r's node indices are offset by
    `r * num_nodes[type]`, so message passing within block r can never reach
    block r' != r (verified by `test_static_graph_replication_equals_per_slice_loop`,
    which compares this against literally looping the conv per-slice)."""
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")
    out = {}
    for (src_t, rel, dst_t), ei in edge_index_dict.items():
        if ei.numel() == 0:
            out[(src_t, rel, dst_t)] = torch.zeros((2, 0), dtype=torch.long)
            continue
        n_src, n_dst = num_nodes[src_t], num_nodes[dst_t]
        offsets = torch.arange(n_repeats, dtype=torch.long)
        src_off = (offsets * n_src).repeat_interleave(ei.shape[1])
        dst_off = (offsets * n_dst).repeat_interleave(ei.shape[1])
        src = ei[0].repeat(n_repeats) + src_off
        dst = ei[1].repeat(n_repeats) + dst_off
        out[(src_t, rel, dst_t)] = torch.stack([src, dst], dim=0)
    return out


# --- Stage 1: spatial encoder ------------------------------------------------


class SpatialEncoder(nn.Module):
    def __init__(
        self,
        node_types: Sequence[str],
        edge_types: Sequence[tuple[str, str, str]],
        *,
        hidden: int = HIDDEN_DIM,
        heads: int = GNN_HEADS,
        n_layers: int = N_GNN_LAYERS,
    ):
        super().__init__()
        self.node_types = list(node_types)
        metadata = (list(node_types), list(edge_types))
        self.convs = nn.ModuleList(
            [HGTConv(-1, hidden, metadata, heads=heads) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.ModuleDict({t: nn.LayerNorm(hidden) for t in node_types}) for _ in range(n_layers)]
        )

    def forward(
        self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict[tuple[str, str, str], torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        h = x_dict
        for i, conv in enumerate(self.convs):
            out = conv(h, edge_index_dict)
            new_h = {}
            for t in self.node_types:
                o = out[t]
                if i > 0:  # dims only match hidden->hidden from layer 1 on
                    o = o + h[t]
                o = self.norms[i][t](F.gelu(o))
                new_h[t] = o
            h = new_h
        return h


class SpatialEncoderMLP(nn.Module):
    """Ablation encoder (Session 10, GNN-vs-MLP axis): per-node-type MLP with
    NO message passing, satisfying the identical `forward(x_dict,
    edge_index_dict) -> h_dict` contract as `SpatialEncoder`. `edge_types`,
    `heads`, and `n_layers` are accepted only to keep the constructor
    drop-in compatible with `SpatialEncoder`'s call sites -- they are
    otherwise unused, since there is no graph convolution to configure.
    `edge_index_dict` is accepted but never read: this is the point of the
    ablation, not an oversight."""

    def __init__(
        self,
        node_types: Sequence[str],
        edge_types: Sequence[tuple[str, str, str]],
        *,
        hidden: int = HIDDEN_DIM,
        heads: int = GNN_HEADS,
        n_layers: int = N_GNN_LAYERS,
    ):
        super().__init__()
        self.node_types = list(node_types)
        self.mlps = nn.ModuleDict(
            {t: nn.Sequential(nn.LazyLinear(hidden), nn.GELU(), nn.LayerNorm(hidden)) for t in node_types}
        )

    def forward(
        self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict[tuple[str, str, str], torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {t: self.mlps[t](x_dict[t]) for t in self.node_types}


# --- Stage 2: typed readout --------------------------------------------------


class Readout(nn.Module):
    """host + mean(IED) + mean(DER) + mean(bus) + max(bus) + globals -> one
    shared trunk `u`. All four targets read the SAME telemetry stream (this
    is why the trunk -- and the TCN below -- are shared rather than
    per-target: four separate TCNs would quadruple parameters for no added
    capacity); the four heads at the end are what differentiate the targets.

    Session 7 generalization: on case33bw, `h_dict` always has all of
    `host`/`IED`/`DER`/`bus` (READOUT_TYPES), so this degrades to exactly
    the original fixed part list -- verified byte-identical by
    `TestReadout::test_matches_original_fixed_part_list_when_all_types_present`.
    A caller whose asset graph lacks `DER`/`bus` (any topology-free or
    partial overlay) still runs: `nn.LazyLinear` accommodates the narrower
    concatenation on its first call, so no per-topology model class is
    needed (mirrors `nonempty_metadata`/`filtered_input`'s "restrict to
    what's actually present" discipline in `asset_graph.py`). NOTE: Sherlock
    grounding (LAB_NOTEBOOK.md 2026-08-05) ended up NOT using this pipeline
    at all -- the real Sherlock export has no verified electrical topology,
    so `experiments/exp07_sherlock.py` uses a separate topology-free
    `CausalTCN` classifier directly. This generalization remains a real,
    tested improvement for any future asset graph with a partial node-type
    set, independent of that outcome.
    """

    def __init__(self, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.proj = nn.LazyLinear(hidden)

    def forward(self, h_dict: dict[str, torch.Tensor], globals_: torch.Tensor) -> torch.Tensor:
        parts = []
        if "host" in h_dict:
            parts.append(h_dict["host"][:, :, 0, :])
        for t in ("IED", "DER", "bus"):
            if t in h_dict:
                parts.append(h_dict[t].mean(dim=2))
        if "bus" in h_dict:
            parts.append(h_dict["bus"].max(dim=2).values)
        parts.append(globals_)
        return self.proj(torch.cat(parts, dim=-1))


# --- Stage 3: causal TCN -----------------------------------------------------


class CausalConv1d(nn.Module):
    """Left-pad-only causal 1D convolution: output[..., t] depends only on
    input[..., <=t]. No BatchNorm; see module docstring."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, S]
        return self.conv(F.pad(x, (self.left_pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = TCN_DROPOUT):
        super().__init__()
        self.conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(F.gelu(self.conv(x)))


class CausalTCN(nn.Module):
    def __init__(
        self,
        channels: int = HIDDEN_DIM,
        kernel_size: int = TCN_KERNEL_SIZE,
        dilations: Sequence[int] = TCN_DILATIONS,
        dropout: float = TCN_DROPOUT,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TCNBlock(channels, kernel_size, d, dropout) for d in dilations]
        )
        self.receptive_field = receptive_field(kernel_size, dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, S]
        for block in self.blocks:
            x = block(x)
        return x


# --- Stage 4: per-target heads + the full pipeline ---------------------------


@dataclass(frozen=True)
class EncoderConfig:
    node_types: tuple[str, ...]
    edge_types: tuple[tuple[str, str, str], ...]
    targets: tuple[str, ...] = TARGETS
    hidden: int = HIDDEN_DIM
    heads: int = GNN_HEADS
    n_gnn_layers: int = N_GNN_LAYERS
    tcn_kernel_size: int = TCN_KERNEL_SIZE
    tcn_dilations: tuple[int, ...] = TCN_DILATIONS
    dropout: float = TCN_DROPOUT
    encoder_type: str = "gnn"  # "gnn" (SpatialEncoder) or "mlp" (SpatialEncoderMLP,
                                # Session 10 GNN-vs-MLP ablation, no message passing)


class PerceptionEncoder(nn.Module):
    """The full pipeline: `x_dict [B,S,N_t,F_t]` + static edges + globals ->
    `{target: logits [B,S]}`."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        if config.encoder_type not in ("gnn", "mlp"):
            raise ValueError(f"encoder_type must be 'gnn' or 'mlp', got {config.encoder_type!r}")
        self.config = config
        spatial_cls = SpatialEncoderMLP if config.encoder_type == "mlp" else SpatialEncoder
        self.spatial = spatial_cls(
            config.node_types, config.edge_types,
            hidden=config.hidden, heads=config.heads, n_layers=config.n_gnn_layers,
        )
        self.readout = Readout(config.hidden)
        self.tcn = CausalTCN(
            channels=config.hidden, kernel_size=config.tcn_kernel_size,
            dilations=config.tcn_dilations, dropout=config.dropout,
        )
        self.heads = nn.ModuleDict(
            {t: nn.Conv1d(config.hidden, 1, kernel_size=1) for t in config.targets}
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        globals_: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        flat, num_nodes, B, S = flatten_for_spatial(x_dict)
        rep_edges = replicate_static_graph(edge_index_dict, num_nodes, B * S)
        h_flat = self.spatial(flat, rep_edges)
        h = unflatten_from_spatial(h_flat, num_nodes, B, S)

        u = self.readout(h, globals_)  # [B, S, hidden]
        trunk = self.tcn(u.transpose(1, 2))  # [B, hidden, S]
        return {t: self.heads[t](trunk).squeeze(1) for t in self.config.targets}  # [B, S] each
