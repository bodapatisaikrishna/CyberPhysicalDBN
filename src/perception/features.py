"""Per-slice, per-node dynamic features for the perception layer (CLAUDE.md
layer [1]). Static structure/features live in `src/perception/asset_graph.py`;
this module produces the time-varying tensors concatenated with those at
model time.

THE LEAK BARRIER is a type boundary, not a blacklist: every function here
takes only a `SliceObservation` (a `(grid, sanitized messages)` pair) or a
`ContinuousTrace` reduced to exactly that. `SliceRecord` -- which carries
`ground_truth`, `analytics`, `false_positives`/`false_negatives` (the
training LABELS) -- never appears in any signature in this file, so those
fields cannot leak into a feature even by accident. `tests/
test_perception_features.py::test_features_bitwise_invariant_to_label_field_perturbation`
enforces this with `torch.equal`, not `allclose`.

THE VIEWPOINT RULE: `sanitize_bus_log` takes exactly one end of each message
(the end the modeled viewpoint could plausibly observe), never both. Taking
both ends and differencing them IS the tamper label (`DeliveryRecord.sent`
vs. `.delivered`, or `Message.origin`/`tampered_by`) -- the observability
model that makes CommandCoherence a genuine inference problem rather than a
lookup: the control centre knows the rung it COMMANDED (its own `sent` log),
never what actually arrived at the wire.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np
import torch

from src.perception.asset_graph import AssetGraph, CyberAsset, CyberOverlayConfig
from src.twin.comms import DeliveryRecord
from src.twin.grid import GridConfig, GridState
from src.twin.runner import ContinuousTrace

NUMERIC_PAYLOAD_KEYS: frozenset[str] = frozenset({"vm_pu_min", "p_mw"})
Observability = Literal["voltage_only", "full_telemetry"]

# Fixed column layouts, exported so tests and exp05's CSV writer can name
# columns rather than index by position.
BUS_DYNAMIC_COLUMNS: tuple[str, ...] = ("vm_pu", "converged", "is_violated", "delta_vm_pu")
IED_DYNAMIC_COLUMNS: tuple[str, ...] = (
    "has_report", "n_report", "zoh_report_vm_pu_min", "report_staleness",
    "has_cmd", "n_cmd", "zoh_cmd_p_mw", "cmd_staleness",
    "zoh_cmd_ladder_index", "n_cmds_so_far",
)
HOST_DYNAMIC_COLUMNS: tuple[str, ...] = (
    "n_reports_this_slice", "mean_reported_vm_pu_min", "n_cmds_this_slice",
    "dispatch_phase", "n_ieds_reporting", "slice_frac",
)
DER_DYNAMIC_COLUMNS: tuple[str, ...] = ("telemetry_available", "zoh_cmd_p_mw", "zoh_cmd_ladder_index")


@dataclass(frozen=True)
class SanitizedMessage:
    """One message, from exactly one viewpoint's side. NO `origin`, NO
    `tampered_by` -- both are attack labels. `numeric_payload` is filtered to
    `NUMERIC_PAYLOAD_KEYS`, which structurally excludes `payload["spoofed"]`
    even though it never carried a numeric value in the first place (belt
    and suspenders: the type filter below also rejects it)."""

    msg_type: str
    sender: str
    receiver: str
    payload_category: str
    numeric_payload: Mapping[str, float]
    timestamp: float


@dataclass(frozen=True)
class SliceObservation:
    """The ONLY input a feature builder in this module may take. This field
    list IS the audit surface -- `tests/test_perception_features.py::
    test_slice_observation_fields_are_whitelisted` pins it via dataclass
    introspection, so a future edit that adds a leaky field fails CI."""

    slice_index: int
    t_units: float
    grid: GridState | None
    messages: tuple[SanitizedMessage, ...]


def sanitize_bus_log(
    log: Sequence[DeliveryRecord], *, viewpoint: str = "ControlCentre"
) -> tuple[SanitizedMessage, ...]:
    """One end of each message, from `viewpoint`'s perspective, never both.

    - Message ORIGINATED by `viewpoint` -> `DeliveryRecord.sent` (the
      viewpoint's own outbound log; what it actually transmitted).
    - Message ADDRESSED to `viewpoint` -> `DeliveryRecord.delivered` (what
      arrived -- possibly tampered; `None` if a manipulation dropped it, in
      which case the message is observable only as an ABSENCE, never
      reconstructed from `sent`).
    - Anything else (a message neither from nor to `viewpoint`) is not
      observable from here and is dropped.
    """
    out: list[SanitizedMessage] = []
    for record in log:
        sent = record.sent
        if sent.sender == viewpoint:
            msg = sent
        elif sent.receiver == viewpoint:
            if record.delivered is None:
                continue
            msg = record.delivered
        else:
            continue
        numeric_payload = {
            k: float(v)
            for k, v in msg.payload.items()
            if k in NUMERIC_PAYLOAD_KEYS and isinstance(v, (int, float))
        }
        out.append(
            SanitizedMessage(
                msg_type=msg.msg_type.value,
                sender=msg.sender,
                receiver=msg.receiver,
                payload_category=msg.payload_category.value,
                numeric_payload=numeric_payload,
                timestamp=msg.timestamp,
            )
        )
    return tuple(out)


def slice_of(t_units: float, delta_t: float, n_slices: int) -> int:
    """1-based slice a continuous-time instant belongs to, matching
    `src.twin.runner.discretize`'s own convention exactly: ground truth (and
    therefore observability) at slice s is evaluated at the slice END,
    `active <=> t <= s * delta_t`. A message at exactly `t = s * delta_t`
    belongs to slice s, not s+1 (pinned by
    `test_message_slice_assignment_matches_discretize_convention`)."""
    if delta_t <= 0:
        raise ValueError(f"delta_t must be positive, got {delta_t}")
    idx = int(math.ceil(t_units / delta_t - 1e-12))
    return int(min(max(1, idx), n_slices))


def build_slice_observations(
    trace: ContinuousTrace,
    delta_t: float,
    n_slices: int,
    *,
    viewpoint: str = "ControlCentre",
) -> list[SliceObservation]:
    """The bridge from a `ContinuousTrace` to this module's whitelisted
    input. Reads ONLY `trace.messages` (via `sanitize_bus_log`) and
    `trace.grid_state_at` -- never `step_completion_times`, `events`,
    `reaction_outcomes`, or any other trace field, all of which are labels
    or label-adjacent."""
    sanitized = sanitize_bus_log(trace.messages, viewpoint=viewpoint)
    by_slice: dict[int, list[SanitizedMessage]] = {}
    for msg in sanitized:
        by_slice.setdefault(slice_of(msg.timestamp, delta_t, n_slices), []).append(msg)

    observations = []
    for s in range(1, n_slices + 1):
        t_units = s * delta_t
        observations.append(
            SliceObservation(
                slice_index=s,
                t_units=t_units,
                grid=trace.grid_state_at(t_units),
                messages=tuple(by_slice.get(s, ())),
            )
        )
    return observations


def _ordered_keys(node_index: Mapping[str, int]) -> list[str]:
    """`{asset_id: row}` -> `[asset_id, ...]` ordered by row index."""
    return [k for k, _ in sorted(node_index.items(), key=lambda kv: kv[1])]


def bus_dynamic_features(
    observations: Sequence[SliceObservation],
    bus_ids: Sequence[int],
    *,
    se_noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """`[S, n_buses, 4]`: vm_pu, converged, is_violated, delta_vm_pu.

    A non-converged or not-yet-solved slice contributes `vm_pu=1.0`
    (nominal), `converged=0`, `is_violated=0` -- never NaN, and distinguished
    from a genuine nominal reading only by the `converged` bit, which a
    consumer must check (mirrors `consequence.classify`'s own
    `NONCONVERGED`-shaped-unobserved discipline).

    `se_noise_sigma` (Session 5 sensitivity arm, LAB_NOTEBOOK.md 2026-08-02
    P2): independent Gaussian state-estimation noise added per bus per
    slice. At `sigma=0` (the default/control), `MeasureCoherence`'s residual
    against a genuinely converged solve is near-noiseless.
    """
    if se_noise_sigma > 0 and rng is None:
        raise ValueError("se_noise_sigma > 0 requires an rng")
    n_buses = len(bus_ids)
    out = torch.zeros((len(observations), n_buses, 4), dtype=torch.float32)
    prev_vm = {b: 1.0 for b in bus_ids}
    for si, obs in enumerate(observations):
        g = obs.grid
        for bi, b in enumerate(bus_ids):
            if g is not None and g.converged and b in g.vm_pu:
                vm = float(g.vm_pu[b])
                if se_noise_sigma > 0:
                    vm = vm + float(rng.normal(0.0, se_noise_sigma))
                converged, violated = 1.0, float(b in g.violated_buses)
            else:
                vm, converged, violated = 1.0, 0.0, 0.0
            out[si, bi, 0] = vm
            out[si, bi, 1] = converged
            out[si, bi, 2] = violated
            out[si, bi, 3] = vm - prev_vm[b]
            prev_vm[b] = vm
    return out


def _nearest_ladder_index(p_mw: float, levels: Sequence[float]) -> float:
    if not levels:
        return -1.0
    return float(min(range(len(levels)), key=lambda i: abs(levels[i] - p_mw)))


def ied_dynamic_features(
    observations: Sequence[SliceObservation],
    ied_assets: Sequence[CyberAsset],
    *,
    p_mw_levels: Sequence[float],
    max_staleness_slices: float,
) -> torch.Tensor:
    """`[S, n_ied, 10]`, columns = `IED_DYNAMIC_COLUMNS`.

    Every message-derived quantity is a 4-channel group (has / n / zoh /
    staleness): the DER report/command cadence leaves ~72% of slices with no
    message at all (dispatch_period=1.0 time unit vs. delta_t~=0.277, at the
    repo's Table-5 m=1 timebase), so a dense value alone cannot distinguish
    "reported 0" from "never reported." The ZOH (zero-order-hold, last
    observed value carried forward) channel is what keeps the TCN's input
    sequence dense despite that sparsity.
    """
    n = len(ied_assets)
    out = torch.zeros((len(observations), n, len(IED_DYNAMIC_COLUMNS)), dtype=torch.float32)
    zoh_report = [1.0] * n
    zoh_cmd = [p_mw_levels[0] if p_mw_levels else 0.0] * n
    last_report_slice: list[int | None] = [None] * n
    last_cmd_slice: list[int | None] = [None] * n
    n_cmds_so_far = [0] * n
    ladder_seen = [False] * n

    for si, obs in enumerate(observations):
        for ii, asset in enumerate(ied_assets):
            der_id = asset.endpoint
            reports = [
                m for m in obs.messages if m.msg_type == "measurement" and m.sender == der_id
            ]
            cmds = [
                m for m in obs.messages if m.msg_type == "command" and m.receiver == der_id
            ]
            if reports and "vm_pu_min" in reports[-1].numeric_payload:
                zoh_report[ii] = reports[-1].numeric_payload["vm_pu_min"]
                last_report_slice[ii] = si
            if cmds and "p_mw" in cmds[-1].numeric_payload:
                zoh_cmd[ii] = cmds[-1].numeric_payload["p_mw"]
                last_cmd_slice[ii] = si
                n_cmds_so_far[ii] += len(cmds)
                ladder_seen[ii] = True

            report_stale = (
                0.0 if last_report_slice[ii] is None
                else min(float(si - last_report_slice[ii]), max_staleness_slices)
            )
            cmd_stale = (
                0.0 if last_cmd_slice[ii] is None
                else min(float(si - last_cmd_slice[ii]), max_staleness_slices)
            )
            ladder_idx = (
                _nearest_ladder_index(zoh_cmd[ii], p_mw_levels) if ladder_seen[ii] else -1.0
            )
            out[si, ii] = torch.tensor([
                float(len(reports) > 0), float(len(reports)), zoh_report[ii], report_stale,
                float(len(cmds) > 0), float(len(cmds)), zoh_cmd[ii], cmd_stale,
                ladder_idx, float(n_cmds_so_far[ii]),
            ])
    return out


def host_dynamic_features(
    observations: Sequence[SliceObservation],
    ied_dynamic: torch.Tensor,
    *,
    dispatch_period_time_units: float,
) -> torch.Tensor:
    """`[S, 1, 6]`, columns = `HOST_DYNAMIC_COLUMNS`. Aggregates ACROSS the
    already-computed per-IED tensor rather than re-deriving from raw
    messages, so host and IED features cannot silently disagree."""
    S = ied_dynamic.shape[0]
    out = torch.zeros((S, 1, len(HOST_DYNAMIC_COLUMNS)), dtype=torch.float32)
    for si, obs in enumerate(observations):
        has_report = ied_dynamic[si, :, 0]
        report_vals = ied_dynamic[si, :, 2]
        n_reports = float(ied_dynamic[si, :, 1].sum().item())
        n_cmds = float(ied_dynamic[si, :, 5].sum().item())
        mask = has_report > 0
        mean_report = float(
            (report_vals[mask].mean() if mask.any() else report_vals.mean()).item()
        )
        phase = (
            (obs.t_units % dispatch_period_time_units) / dispatch_period_time_units
            if dispatch_period_time_units > 0 else 0.0
        )
        out[si, 0] = torch.tensor([
            n_reports, mean_report, n_cmds, float(phase),
            float(mask.sum().item()), float(si + 1) / max(S, 1),
        ])
    return out


def der_dynamic_features(
    der_ids: Sequence[str],
    ied_by_der: Mapping[str, int],
    ied_dynamic: torch.Tensor,
    *,
    observability: Observability,
) -> torch.Tensor:
    """`[S, n_der, 3]`, columns = `DER_DYNAMIC_COLUMNS`.

    Under `"voltage_only"` (the default / headline arm), DER setpoint
    telemetry is NOT exposed -- every column is zero regardless of what the
    IED tensor actually observed, verified by
    `test_voltage_only_excludes_setpoint_telemetry` perturbing
    `state.setpoints_mw` and asserting bit-identical output. Under
    `"full_telemetry"` (the upper-bound control arm), the DER's own IED
    command channel is copied through. The tensor WIDTH is identical in
    both arms so downstream shapes never depend on the observability
    setting.
    """
    if observability not in ("voltage_only", "full_telemetry"):
        raise ValueError(f"unknown observability {observability!r}")
    S = ied_dynamic.shape[0]
    n = len(der_ids)
    out = torch.zeros((S, n, len(DER_DYNAMIC_COLUMNS)), dtype=torch.float32)
    if observability == "full_telemetry":
        for di, der_id in enumerate(der_ids):
            if der_id not in ied_by_der:
                continue
            ii = ied_by_der[der_id]
            out[:, di, 0] = 1.0
            out[:, di, 1] = ied_dynamic[:, ii, 6]
            out[:, di, 2] = ied_dynamic[:, ii, 8]
    return out


@dataclass(frozen=True)
class DynamicFeatureConfig:
    se_noise_sigma: float = 0.0
    observability: Observability = "voltage_only"
    max_staleness_slices: float = 50.0
    viewpoint: str = "ControlCentre"


def build_dynamic_features(
    trace: ContinuousTrace,
    delta_t: float,
    n_slices: int,
    asset_graph: AssetGraph,
    overlay: CyberOverlayConfig,
    grid_config: GridConfig,
    *,
    dispatch_period_time_units: float,
    config: DynamicFeatureConfig = DynamicFeatureConfig(),
    rng: np.random.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """The single entry point exp05 calls: `ContinuousTrace` + the static
    `AssetGraph` -> per-type dynamic tensors, `[S, N_type, F_type]`, row
    order matching `asset_graph.node_index[type]` exactly. `line`,
    `transformer`, `RTU`, `relay` have no dynamic signal in this twin and get
    zero-width tensors (still shaped, so a caller's concatenation code needs
    no per-type special case).
    """
    observations = build_slice_observations(trace, delta_t, n_slices, viewpoint=config.viewpoint)

    bus_order = _ordered_keys(asset_graph.node_index["bus"])
    bus_x = bus_dynamic_features(
        observations, [int(b) for b in bus_order],
        se_noise_sigma=config.se_noise_sigma, rng=rng,
    )

    ied_order = _ordered_keys(asset_graph.node_index["IED"])
    ied_by_id = {a.asset_id: a for a in overlay.assets if a.node_type == "IED"}
    ied_assets = [ied_by_id[aid] for aid in ied_order]
    ied_x = ied_dynamic_features(
        observations, ied_assets,
        p_mw_levels=list(grid_config.p_mw_levels),
        max_staleness_slices=config.max_staleness_slices,
    )

    host_x = host_dynamic_features(
        observations, ied_x, dispatch_period_time_units=dispatch_period_time_units
    )

    der_order = _ordered_keys(asset_graph.node_index["DER"])
    ied_by_der = {a.endpoint: i for i, a in enumerate(ied_assets)}
    der_x = der_dynamic_features(
        der_order, ied_by_der, ied_x, observability=config.observability
    )

    zero_width = {
        t: torch.zeros((n_slices, asset_graph.counts[t], 0), dtype=torch.float32)
        for t in ("line", "transformer", "RTU", "relay")
    }
    return {"bus": bus_x, "IED": ied_x, "host": host_x, "DER": der_x, **zero_width}


@dataclass(frozen=True)
class FeatureScaler:
    """Per-feature z-score, fit ONCE on the train split and frozen. `transform`
    never recomputes statistics -- val/calib/test slices are scaled with the
    train split's mean/std, never their own, or the split boundary a
    calibration report depends on (`src/perception/calibration.py`) would be
    silently violated."""

    mean: torch.Tensor  # [F]
    std: torch.Tensor  # [F]

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.mean.shape[0]:
            raise ValueError(
                f"feature width mismatch: scaler fit on {self.mean.shape[0]}, "
                f"got {x.shape[-1]}"
            )
        return (x - self.mean) / self.std


def fit_feature_scaler(x: torch.Tensor, *, eps: float = 1e-6) -> FeatureScaler:
    """Fit mean/std over every leading dimension of `x` (e.g. `[n_runs, S, N,
    F]` or `[S, N, F]`), keeping only the last (feature) axis. `eps` floors
    std so a constant feature (e.g. a permanently-zero DER column under
    `voltage_only`) does not divide by zero."""
    flat = x.reshape(-1, x.shape[-1])
    mean = flat.mean(dim=0)
    std = flat.std(dim=0).clamp_min(eps)
    return FeatureScaler(mean=mean, std=std)
