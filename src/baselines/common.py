"""Shared scaffolding for external ML baselines (CLAUDE.md minimum-viable-
publication bar: "Phase 1-3 + external baselines + lead-time evaluation").

Scenario generation is NOT reimplemented here. `experiments/exp06_baselines.py`
imports `ScenarioData`/`run_scenario`/`generate_split`/`build_overlay`/
`build_zones`/`extract_labels`/`compute_base_rates` directly from
`experiments/exp05_perception.py` (via `importlib.util`, the pattern already
established by `tests/test_twin.py::TestTimebasePin._load`) and reuses the
identical `SeedSequence(42).spawn(5)` split -- this is the only way "identical
twin scenarios and seeds" is literally true rather than merely "generated the
same way." This module holds only what is genuinely NEW for baselines: the
engineered-feature flattening shared by the GBM and LSTM-AE baselines, and the
hyperparameter-search harness shared by all four.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

# Node types with genuine dynamic (per-slice) signal -- excludes line/
# transformer/RTU/relay, which are zero-width in src/perception/features.py's
# output on this feeder (nothing to aggregate).
ENGINEERED_TYPES: tuple[str, ...] = ("bus", "IED", "host", "DER")
ENGINEERED_AGGS: tuple[str, ...] = ("mean", "max")


@dataclass(frozen=True)
class BaselineResult:
    """One baseline's per-slice score trajectory for one scenario. `score`
    is aligned 1:1 with `discrete.records` (index i == slice_index i+1,
    matching every other per-slice array convention in this repo)."""

    name: str
    run_id: int
    score: np.ndarray  # [N_SLICES]


def aggregate_node_type(
    x: torch.Tensor, aggs: tuple[str, ...] = ENGINEERED_AGGS
) -> torch.Tensor:
    """`[S, N, F] -> [S, F * len(effective_aggs)]`.

    When `N == 1` (host), `mean` and `max` over the node axis are
    IDENTICAL by construction (there is only one value to reduce), so only
    ONE copy of the F columns is returned rather than two duplicate copies
    -- duplicating a constant column wastes downstream model capacity and
    would silently double-weight `host` relative to every other type in a
    flat concatenation.
    """
    if x.dim() != 3:
        raise ValueError(f"expected [S,N,F], got shape {tuple(x.shape)}")
    n = x.shape[1]
    parts = []
    seen: set[str] = set()
    for agg in aggs:
        if n == 1 and agg in seen:
            continue
        if n == 1:
            parts.append(x[:, 0, :])
            seen.add(agg)
            continue
        if agg == "mean":
            parts.append(x.mean(dim=1))
        elif agg == "max":
            parts.append(x.max(dim=1).values)
        else:
            raise ValueError(f"unknown aggregation {agg!r}")
    if n == 1:
        # mean and max coincide for a single node: emit exactly one copy
        # regardless of how many entries `aggs` listed.
        return x[:, 0, :]
    return torch.cat(parts, dim=-1)


def flatten_engineered_features(
    dynamic_x: Mapping[str, torch.Tensor],
    node_types: tuple[str, ...] = ENGINEERED_TYPES,
    aggs: tuple[str, ...] = ENGINEERED_AGGS,
) -> tuple[torch.Tensor, list[str]]:
    """`build_dynamic_features`'s DYNAMIC output (never
    `combine_static_dynamic`'s static+dynamic concat) -> `[S, F_total]` plus
    the column names that produced each entry, so `F_total` (40 on this
    feeder: bus 4*2=8 + IED 10*2=20 + host 6*1=6 + DER 3*2=6) is derived and
    printed, never a magic number in a downstream CSV.

    Static asset-graph features are excluded deliberately: on this one
    feeder/overlay they are identical, zero-variance constants across every
    scenario (the same bus degrees, the same DER bus indices, every time) --
    including them would only add noise columns for a tree or an
    autoencoder to waste capacity on, never a source of discriminative
    signal between scenarios.
    """
    from src.perception.features import (
        BUS_DYNAMIC_COLUMNS,
        DER_DYNAMIC_COLUMNS,
        HOST_DYNAMIC_COLUMNS,
        IED_DYNAMIC_COLUMNS,
    )

    column_source = {
        "bus": BUS_DYNAMIC_COLUMNS,
        "IED": IED_DYNAMIC_COLUMNS,
        "host": HOST_DYNAMIC_COLUMNS,
        "DER": DER_DYNAMIC_COLUMNS,
    }

    parts: list[torch.Tensor] = []
    names: list[str] = []
    for node_type in node_types:
        x = dynamic_x[node_type]
        agg_out = aggregate_node_type(x, aggs)
        parts.append(agg_out)
        cols = column_source[node_type]
        n = x.shape[1]
        effective_aggs = ("value",) if n == 1 else aggs
        for agg in effective_aggs:
            for col in cols:
                names.append(f"{node_type}.{col}.{agg}")

    flat = torch.cat(parts, dim=-1)
    if flat.shape[-1] != len(names):
        raise AssertionError(
            f"column-name bookkeeping drifted from the tensor: {flat.shape[-1]} "
            f"columns vs {len(names)} names -- fix aggregate_node_type or this "
            "loop, do not silently truncate/pad"
        )
    return flat, names


@dataclass(frozen=True)
class TrialResult:
    trial_id: int
    config: Mapping[str, object]
    val_auc_pr: float


def hyperparameter_search(
    configs: Sequence[Mapping[str, object]],
    train_and_score_fn: Callable[[Mapping[str, object], np.random.Generator], float],
    *,
    rng: np.random.Generator,
) -> tuple[list[TrialResult], Mapping[str, object]]:
    """Runs `train_and_score_fn(config, rng)` for EVERY entry in `configs`
    and returns `(all_trials, best_config)`, best selected by max
    `val_auc_pr`. Callers must write `all_trials` to a CSV in full -- never
    just the winner (CLAUDE.md rule 2: a result that only records the
    winner is written from a foregone conclusion, not a documented search).

    `rng` is a single generator threaded through every trial in order (not
    re-seeded per trial), so the full search is reproducible from one seed
    while still giving each trial independent draws.
    """
    if not configs:
        raise ValueError("configs must not be empty")
    trials: list[TrialResult] = []
    for trial_id, config in enumerate(configs):
        val_auc_pr = train_and_score_fn(config, rng)
        trials.append(TrialResult(trial_id=trial_id, config=config, val_auc_pr=val_auc_pr))
    best = max(trials, key=lambda t: t.val_auc_pr)
    return trials, best.config


TUNING_BUDGET_NOTE_BASELINE = (
    "genuine grid/random search over this baseline's own hyperparameter "
    "space (configs/baselines.yaml), every trial logged (not just the "
    "winner), selected by val-split AUC-PR against grid_unstable"
)
TUNING_BUDGET_NOTE_DBN = (
    "the proposed system's ARCHITECTURE (n_gnn_layers=3, hidden=64, "
    "tcn_dilations=(1,2,4,8,16)) was NOT grid-searched -- it is fixed by a "
    "structural/theoretical argument (src/perception/encoder.py's 3-hop "
    "cyber-shortcut proof: bus->DER->IED->host is exactly 3 hops, "
    "independent of the 20-hop electrical distance, verified by "
    "test_hops_from_der_bus_to_host_equals_n_gnn_layers), not tuned by "
    "validation performance. Only TRAINING hyperparameters (lr, epochs, "
    "early-stopping patience -- configs/perception.yaml training:) were "
    "used AS GIVEN, and those were likewise set once a priori, not "
    "grid-searched either. Every baseline in this session got a genuine "
    "search; the proposed system's architecture did not, by design -- no "
    "equivalent search was invented post hoc to close this gap."
)
