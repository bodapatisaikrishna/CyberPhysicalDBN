"""LSTM autoencoder, reconstruction-error anomaly detection (CLAUDE.md's
`src/baselines/`: "LSTM-AE"). Hand-implemented in `torch` (already pinned) --
no PyOD dependency, per this session's user-approved decision.

THE LEAK QUESTION, answered directly (LAB_NOTEBOOK.md 2026-08-03), not
smoothed over: every twin scenario is under attack from `t=0`
(`configs/twin.yaml`'s `enabled_roots` are all active at the start), so there
is no attack-free scenario to train an anomaly detector on. This module's
"presumed-nominal" training corpus is the EARLY PREFIX of each scenario,
before the fastest of the four attack branches completes
(`training_window_cutoff`, below).

Is reading `ground_truth` to find that cutoff a leak? It IS mild
privileged-information use in TRAINING-SET CURATION (a fielded system has no
`ground_truth` to consult; a real deployment would substitute an
operator-declared quiet period -- stated here as a limitation, not hidden).
It is NOT the class of leak `src/perception/features.py`'s `SliceObservation`
barrier exists to prevent: that barrier stops a label being read as a
FEATURE VALUE at scoring time, so the model literally sees the answer. Here,
`ground_truth` is read exactly ONCE, to pick a slice-index boundary, before
any feature is computed; the model never receives it as input or target, and
the identical feature-extraction/scoring code
(`src.baselines.common.flatten_engineered_features`) runs on every split
regardless of this boundary. Enforced structurally, mirroring
`SliceObservation`'s own barrier: `training_window_cutoff`'s signature
accepts ONLY ground_truth-shaped sequences and `flatten_engineered_features`
never accepts anything ground-truth-shaped at all -- the two are
type-disjoint, not merely disciplined by convention (see
`tests/test_lstm_ae.py::test_cutoff_selection_structurally_disjoint_from_feature_computation`).
Scope is confined to training-corpus curation; VAL/CALIB/TEST scoring covers
the FULL trajectory of every scenario, identically to every other baseline
and the DBN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import expit

from src.perception.encoder import TCN_DILATIONS, TCN_KERNEL_SIZE, receptive_field

# Reused directly from encoder.py, not re-derived, for cross-baseline
# comparability (the same 63-slice horizon the DBN's perception TCN sees).
WINDOW_SLICES: int = receptive_field(TCN_KERNEL_SIZE, TCN_DILATIONS)


def training_window_cutoff(ground_truth_stream: Sequence[Mapping[str, int]]) -> int:
    """0-based index of the FIRST slice where ANY attack-graph node's
    `ground_truth` bit is 1 -- the MINIMUM across all branches, not an
    average or per-branch value, so the resulting "presumed-nominal" prefix
    `[0, cutoff)` is conservative: it costs training data rather than risk
    including an already-compromised slice.

    Reads ONLY `ground_truth`-shaped dicts. Returns `len(ground_truth_stream)`
    (the whole scenario) if no node ever activates -- a legitimate, if
    unusual, fully-nominal scenario.
    """
    for i, gt in enumerate(ground_truth_stream):
        if any(v == 1 for v in gt.values()):
            return i
    return len(ground_truth_stream)


def build_causal_windows(flat: torch.Tensor, window: int = WINDOW_SLICES) -> torch.Tensor:
    """`[S, F] -> [S, window, F]`: windows[i] is the `window` slices ending
    at (and including) slice `i`, LEFT-ZERO-PADDED for `i < window - 1` --
    mirrors `CausalConv1d`'s left-pad convention (`src/perception/encoder.py`)
    so the AE and the DBN's perception TCN treat scenario-start identically.
    """
    if flat.dim() != 2:
        raise ValueError(f"expected [S,F], got shape {tuple(flat.shape)}")
    s, f = flat.shape
    padded = torch.cat([torch.zeros(window - 1, f, dtype=flat.dtype), flat], dim=0)
    windows = padded.unfold(0, window, 1)  # [S, F, window]
    return windows.permute(0, 2, 1).contiguous()  # [S, window, F]


@dataclass(frozen=True)
class LSTMAETrialConfig:
    hidden_dim: int
    latent_dim: int
    n_layers: int
    dropout: float
    learning_rate: float


class LSTMAutoencoder(nn.Module):
    """Encoder-LSTM -> bottleneck -> decoder-LSTM reconstructing the
    `window x input_dim` sequence. The bottleneck is broadcast to every
    decoder timestep (a fixed-length reconstruction target, not a
    variable-length seq2seq problem, so there is no separate decoder input
    sequence to feed token-by-token)."""

    def __init__(self, input_dim: int, config: LSTMAETrialConfig):
        super().__init__()
        lstm_dropout = config.dropout if config.n_layers > 1 else 0.0
        self.encoder_lstm = nn.LSTM(
            input_dim, config.hidden_dim, config.n_layers,
            batch_first=True, dropout=lstm_dropout,
        )
        self.to_latent = nn.Linear(config.hidden_dim, config.latent_dim)
        self.from_latent = nn.Linear(config.latent_dim, config.hidden_dim)
        self.decoder_lstm = nn.LSTM(
            config.hidden_dim, config.hidden_dim, config.n_layers,
            batch_first=True, dropout=lstm_dropout,
        )
        self.output_proj = nn.Linear(config.hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x: [B, W, F] -> recon: [B, W, F]`."""
        b, w, _ = x.shape
        _, (h_n, _) = self.encoder_lstm(x)
        z = self.to_latent(h_n[-1])  # [B, latent]
        seed = self.from_latent(z).unsqueeze(1).expand(b, w, -1)  # [B, W, hidden]
        dec_out, _ = self.decoder_lstm(seed)
        return self.output_proj(dec_out)


def reconstruction_error_last_step(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-window MSE of the LAST timestep only -- keeps the per-slice score
    trajectory causal and 1:1 with slices (score at slice s depends only on
    slices <= s), even though the model was trained to reconstruct the
    WHOLE window."""
    return ((recon[:, -1, :] - target[:, -1, :]) ** 2).mean(dim=-1)


def train_autoencoder(
    config: LSTMAETrialConfig,
    train_windows: torch.Tensor,
    val_windows: torch.Tensor,
    *,
    n_epochs: int,
    batch_size: int,
    grad_clip_norm: float,
    patience: int,
    torch_seed: int,
) -> tuple[LSTMAutoencoder, list[dict]]:
    """Unsupervised: trains on whole-window reconstruction MSE (never
    touches a label). Early stopping on VAL whole-window reconstruction
    loss -- both `train_windows`/`val_windows` must already be restricted
    to the presumed-nominal prefix by the caller (this function has no way
    to enforce that from tensor data alone; the restriction happens strictly
    outside, via `training_window_cutoff`, per this module's leak-barrier
    design)."""
    torch.manual_seed(torch_seed)
    input_dim = train_windows.shape[-1]
    model = LSTMAutoencoder(input_dim, config)
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    rows, best_val, bad_epochs, best_state = [], float("inf"), 0, None
    n = train_windows.shape[0]
    rng = np.random.default_rng(torch_seed)

    for epoch in range(n_epochs):
        model.train()
        order = rng.permutation(n)
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            batch = train_windows[idx]
            recon = model(batch)
            loss = F.mse_loss(recon, batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_recon = model(val_windows)
            val_loss = float(F.mse_loss(val_recon, val_windows).item())

        rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, rows


def score_trajectory(model: LSTMAutoencoder, windows: torch.Tensor) -> np.ndarray:
    """`windows: [S, W, F] -> [S]` raw reconstruction-error scores (NOT yet
    a probability -- see `error_to_probability`)."""
    model.eval()
    with torch.no_grad():
        recon = model(windows)
        errors = reconstruction_error_last_step(recon, windows)
    return errors.numpy()


@dataclass(frozen=True)
class ReconErrorScaler:
    mean_err: float
    std_err: float
    fit_split: str  # provenance, mirrors TemperatureScaler's fit_split field


def fit_recon_error_scaler(
    errors_nominal_val: np.ndarray, *, eps: float = 1e-6, fit_split: str = "val_nominal_prefix"
) -> ReconErrorScaler:
    """Fit ONLY on the presumed-nominal prefix of the VAL split -- never
    test, never the full (attack-included) trajectory of any split."""
    if len(errors_nominal_val) == 0:
        raise ValueError("no nominal validation errors to fit a scaler on")
    return ReconErrorScaler(
        mean_err=float(errors_nominal_val.mean()),
        std_err=float(max(float(errors_nominal_val.std()), eps)),
        fit_split=fit_split,
    )


def error_to_probability(mse: np.ndarray, scaler: ReconErrorScaler) -> np.ndarray:
    """`sigmoid((mse - mean_err) / std_err)`. Monotonic in `mse`, so AUC-PR
    (rank-only) is unaffected by this choice of link function -- but ECE and
    Brier ARE transform-dependent, and this must be read alongside the
    comparison table: the AE's "probability" is a CHOSEN link function over
    an unsupervised error signal, not the output of a fitted probabilistic
    inference procedure like the DBN's posterior. Its calibration numbers
    answer a narrower question ("how well does this sigmoid map error to
    frequency") than the DBN's calibration claim.
    """
    # float64 + scipy's stable expit, deliberately, NOT `1/(1+exp(-z))` on the
    # incoming dtype. `score_trajectory` returns `errors.numpy()` off a torch
    # tensor, i.e. float32, and a hand-rolled sigmoid overflows float32 for
    # |z| > ~88 (exp(88)~=1.7e38 against a float32 max of 3.4e38) even though
    # the old +-500 clip is perfectly safe in float64. The overflow was not
    # cosmetic: it drove `np.exp(-z)` to inf and saturated the returned
    # probability to EXACTLY 0.0 / 1.0 rather than 7e-218 / 1-eps, which makes
    # any log-based metric infinite and creates large blocks of exactly-tied
    # scores. Observed as `RuntimeWarning: overflow encountered in exp` in
    # exp09's Session-10 run log.
    z = np.clip(
        (np.asarray(mse, dtype=np.float64) - scaler.mean_err) / scaler.std_err, -500.0, 500.0
    )
    return expit(z)
