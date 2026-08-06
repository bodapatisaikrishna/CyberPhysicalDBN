"""Technique -> TTC amortized model (CLAUDE.md layer [2], Session 8, claim
C2).

Maps `(MITRE technique, asset_context, defensive_posture,
attacker_capability)` -> `T_bar_s` (mean time-to-compromise), trained on
rows pooling real twin-measured completion times with the 30 TRAIN
`src/attack_graph/family.py` graphs' synthetic-ground-truth rows
(LAB_NOTEBOOK.md 2026-08-05, binding decision 5). Feeds `src/dbn`'s
uniformization step UNCHANGED: this module predicts `T_bar_s` only, then
`apply_ttc_predictions` relabels an attack graph's `ttc` attribute with the
prediction and hands the mutated graph straight to the EXISTING
`src.dbn.parameterization.attach_cpds` (which itself calls the real
`compute_delta_t`/`compute_ps`, Eq. 3) -- this module never reimplements
that math.

Only 8 MITRE techniques are known (`src.attack_graph.graph.
technique_table3_ttc`'s keys) -- the "embedding" is deliberately a small
lookup table over that fixed, tiny vocabulary, not a general text/graph
embedding. The real generalization work is the continuous context vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attack_graph.graph import technique_table3_ttc

CONTEXT_COLUMNS: tuple[str, ...] = ("asset_context", "defensive_posture", "attacker_capability")


def known_techniques() -> tuple[str, ...]:
    """8-technique vocabulary, derived from `technique_table3_ttc()` --
    never hand-duplicated, so this and `family.py`'s vocabulary cannot
    drift apart."""
    return tuple(sorted(technique_table3_ttc()))


@dataclass(frozen=True)
class ContextNormalizer:
    """Mean/std of `CONTEXT_COLUMNS`, fit ONCE on TRAIN rows only (mirrors
    `src.perception.features.FeatureScaler`'s train-only-fit discipline).
    `transform` never refits -- val/test rows are always scaled with the
    train split's statistics."""

    mean: np.ndarray  # [3]
    std: np.ndarray  # [3]

    def transform(self, context: np.ndarray) -> np.ndarray:
        if context.shape[-1] != len(CONTEXT_COLUMNS):
            raise ValueError(f"expected {len(CONTEXT_COLUMNS)} context columns, got {context.shape[-1]}")
        return (context - self.mean) / self.std


def fit_context_normalizer(train_rows: pd.DataFrame, *, eps: float = 1e-6) -> ContextNormalizer:
    values = train_rows[list(CONTEXT_COLUMNS)].to_numpy(dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < eps, eps, std)
    return ContextNormalizer(mean=mean, std=std)


class TTCAmortizedModel(nn.Module):
    """`nn.Embedding(n_techniques, embedding_dim)` for technique, concatenated
    with normalized `[asset_context, defensive_posture, attacker_capability]`,
    through a 2-layer MLP to one raw scalar `r`; predicted `T_bar_s =
    exp(r)`. Log-space output because TTCs span ~1/3 to 50 (~150x) -- a
    plain linear head regressing raw `T_bar_s` would be dominated by the
    largest values."""

    def __init__(self, n_techniques: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(n_techniques, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim + len(CONTEXT_COLUMNS), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, technique_idx: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(technique_idx)
        raw = self.mlp(torch.cat([emb, context], dim=-1)).squeeze(-1)
        return torch.exp(raw)

    def forward_log(self, technique_idx: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Raw log-space output (pre-exp) -- used for the MSE training loss
        directly, avoiding an exp-then-log round trip."""
        emb = self.embedding(technique_idx)
        return self.mlp(torch.cat([emb, context], dim=-1)).squeeze(-1)


@dataclass(frozen=True)
class AmortizedTrainConfig:
    embedding_dim: int
    hidden_dim: int
    learning_rate: float
    weight_decay: float
    n_epochs: int
    early_stopping_patience_epochs: int
    grad_clip_norm: float
    seed: int


def _rows_to_tensors(
    rows: pd.DataFrame, technique_to_idx: dict[str, int], normalizer: ContextNormalizer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    unknown = set(rows["technique"]) - set(technique_to_idx)
    if unknown:
        raise ValueError(f"row(s) reference unknown technique(s) {sorted(unknown)}, not in known_techniques()")
    technique_idx = torch.tensor([technique_to_idx[t] for t in rows["technique"]], dtype=torch.long)
    context = torch.tensor(
        normalizer.transform(rows[list(CONTEXT_COLUMNS)].to_numpy(dtype=np.float64)), dtype=torch.float32,
    )
    log_ttc = torch.tensor(np.log(rows["true_ttc"].to_numpy(dtype=np.float64)), dtype=torch.float32)
    return technique_idx, context, log_ttc


def fit_ttc_amortized_model(
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    techniques: tuple[str, ...],
    config: AmortizedTrainConfig,
) -> tuple[TTCAmortizedModel, ContextNormalizer, list[dict]]:
    """`train_rows`/`val_rows` columns: `technique, asset_context,
    defensive_posture, attacker_capability, true_ttc` (plus any extra
    provenance columns, ignored). MSE on log `T_bar_s`; early-stopping
    checkpoint selection by `val_rows` MAE(log T_bar_s) -- mirrors
    `experiments/exp07_sherlock.py::train_sherlock_model`'s per-epoch
    train/val-loss / best-state-restore pattern."""
    torch.manual_seed(config.seed)
    normalizer = fit_context_normalizer(train_rows)
    technique_to_idx = {t: i for i, t in enumerate(techniques)}

    train_technique, train_context, train_log_ttc = _rows_to_tensors(train_rows, technique_to_idx, normalizer)
    val_technique, val_context, val_log_ttc = _rows_to_tensors(val_rows, technique_to_idx, normalizer)

    model = TTCAmortizedModel(len(techniques), config.embedding_dim, config.hidden_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    rows, best_val, bad_epochs, best_state = [], float("inf"), 0, None
    for epoch in range(config.n_epochs):
        model.train()
        pred_log = model.forward_log(train_technique, train_context)
        train_loss = F.mse_loss(pred_log, train_log_ttc)
        opt.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        opt.step()

        model.eval()
        with torch.no_grad():
            val_pred_log = model.forward_log(val_technique, val_context)
            val_mae = F.l1_loss(val_pred_log, val_log_ttc).item()

        rows.append({"epoch": epoch, "train_loss": float(train_loss.item()), "val_mae_log_ttc": val_mae})

        if val_mae < best_val - 1e-6:
            best_val = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.early_stopping_patience_epochs:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, normalizer, rows


def predict_ttc_for_graph(
    ag, model: TTCAmortizedModel, normalizer: ContextNormalizer, techniques: tuple[str, ...],
) -> dict[str, float]:
    """Zero-expert-input predictions for every timed node (`data["ttc"] is
    not None`). Reads ONLY `mitre_technique`/`asset_context`/
    `defensive_posture`/`attacker_capability` -- NEVER `ag.nodes[n]["ttc"]`
    (which on a family/test graph IS the ground truth). Leak-barrier
    enforced by `tests/test_amortized.py`'s ttc-perturbation invariance
    test."""
    technique_to_idx = {t: i for i, t in enumerate(techniques)}
    timed_nodes = [n for n, d in ag.nodes(data=True) if d.get("ttc") is not None]
    if not timed_nodes:
        return {}

    unknown = {ag.nodes[n]["mitre_technique"] for n in timed_nodes} - set(technique_to_idx)
    if unknown:
        raise ValueError(f"graph node(s) reference unknown technique(s) {sorted(unknown)}")

    technique_idx = torch.tensor([technique_to_idx[ag.nodes[n]["mitre_technique"]] for n in timed_nodes], dtype=torch.long)
    context_raw = np.array(
        [[ag.nodes[n][c] for c in CONTEXT_COLUMNS] for n in timed_nodes], dtype=np.float64,
    )
    context = torch.tensor(normalizer.transform(context_raw), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        predictions = model(technique_idx, context)
    return {n: float(p) for n, p in zip(timed_nodes, predictions)}


def apply_ttc_predictions(ag, predictions):
    """Returns `ag.copy()` with every timed node's `"ttc"` replaced by
    `predictions[node]`. Raises on missing or extra node keys -- the
    prediction set must cover exactly the graph's own timed-node set, never
    silently partial. `delta_t` is NEVER touched here: `attach_cpds`
    recomputes it fresh from the mutated graph's own
    `collect_uniformization_ttcs`, exactly as LAB_NOTEBOOK.md's binding
    decision 4 specifies -- this function relabels Eq. 3's INPUT, never its
    math."""
    timed_nodes = {n for n, d in ag.nodes(data=True) if d.get("ttc") is not None}
    given = set(predictions)
    if given != timed_nodes:
        missing = timed_nodes - given
        extra = given - timed_nodes
        raise ValueError(
            f"predictions must cover exactly the graph's timed nodes; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    mutated = ag.copy()
    for node, ttc in predictions.items():
        mutated.nodes[node]["ttc"] = float(ttc)
    return mutated
