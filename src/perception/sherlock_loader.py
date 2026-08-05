"""Sherlock (Wagner et al., ACM CODASPY'25) real-data loader for the
perception layer (CLAUDE.md layer [1], Session 7 grounding).

VERIFIED REAL FORMAT (this module was rewritten once the download completed
and the real files were inspected -- LAB_NOTEBOOK.md 2026-08-05 records the
full discrepancy against this project's original task description, which
assumed message-level IEC-104 traffic). What `data/sherlock/01-Basic/`
actually ships, at the scenario root:

    train.n302.state.gz   -- one JSON object per line, one line per SECOND
    test.n302.state.gz       (verified: exact 1.0s cadence, 43204 lines each):
        {"timestamp": <unix float>,
         "state": {"bus.0:voltage": <pu>, "bus.0:voltage_angle": <deg>,
                    "line.0:active_power_from": <W>, "switch.8:closed": <bool>,
                    "load.0:active_power": <W>, "trafo.0:tap_position": <int>,
                    "sgen.5:active_power": <W>, ...},   # 470 keys/line
         "malicious": false | "<event_id> (benign event)" | "<event_id>"}

This is PHYSICAL POWER-GRID TELEMETRY (PowerOwl/pandapower state, keyed by
semantic component name -- confirmed against `raw/train/data-point-map.json`,
which maps each real IEC-104 point address to exactly this `element:attribute`
naming), not IEC-104 message traffic with `src`/`dest`/`activity` fields.
`malicious` is per-record ground truth encoded as a STRING event id, already
resolved: `false` = nothing active, `"<n> (benign event)"` = a non-attack
event, a bare numeric string = a REAL ATTACK. This module's original
message-level design (`SherlockMessage`, host/IED classification, a comms-
only asset graph) does not apply to what is actually shipped and has been
removed rather than left as unused/speculative code.

NO VERIFIED ELECTRICAL TOPOLOGY. Unlike the twin's `case33bw` (a known
pandapower net with `from_bus`/`to_bus` per line), Sherlock's state export
names components (`bus.N`, `line.N`, `trafo.N`, `sgen.N`, `load.N`,
`switch.N`) but never their connectivity -- `data-point-map.json` maps
point addresses to `element:attribute`, not to a line's endpoint buses.
Reconstructing real connectivity would require parsing `raw/train/docs/
network.svg` or the raw pcaps, out of scope for this session (stated here,
not silently attempted). Consequently `src/perception/asset_graph.py` /
`encoder.PerceptionEncoder` (HGTConv over a real graph) are NOT used for
Sherlock: this module builds a topology-FREE, per-slice AGGREGATE feature
vector instead, consumed by a plain `CausalTCN` classifier
(`experiments/exp07_sherlock.py`). This is a real architectural divergence
from the twin pipeline, stated directly, not a silent downgrade.

THE LEAK BARRIER (mirrors `features.py`): `parse_state_line` splits the SAME
raw dict into a `SherlockStateRecord` (no `malicious` field, structurally)
and a `SherlockLabel` (ONLY `malicious`) at the earliest possible point --
no function past that split can see both.

SHARED SUBSPACE FOR BIDIRECTIONAL TRANSFER (user-approved design, revised
after this rewrite): the twin and Sherlock both genuinely measure BUS
VOLTAGE IN PER-UNIT (confirmed: Sherlock's `bus.N:voltage` carries
`"unit": "PER_UNIT"` in `data-point-map.json`, directly comparable to the
twin's own `vm_pu`) -- so the shared subspace is `(mean bus voltage pu,
its slice-to-slice delta)`, not the comms-report columns this module
originally proposed before the real format was known.
"""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import torch

STATE_KEY_RE = re.compile(r"^(?P<component>[a-z_]+)\.(?P<index>\d+):(?P<attribute>.+)$")

# Aggregate global feature vector: (component, attribute, agg) -> one column.
# Every entry is a plain mean/std/min/max/fraction over REAL values present in
# the record -- no invented quantity, no assumed topology. Components/
# attributes verified present in the real downloaded 01-Basic state export.
_AGG_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("bus_voltage_pu_mean", "bus", "voltage", "mean"),
    ("bus_voltage_pu_std", "bus", "voltage", "std"),
    ("bus_voltage_pu_min", "bus", "voltage", "min"),
    ("bus_voltage_pu_max", "bus", "voltage", "max"),
    ("line_current_a_mean", "line", "current_from", "mean"),
    ("line_active_power_w_mean", "line", "active_power_from", "mean"),
    ("load_active_power_w_mean", "load", "active_power", "mean"),
    ("load_reactive_power_var_mean", "load", "reactive_power", "mean"),
    ("sgen_active_power_w_mean", "sgen", "active_power", "mean"),
    ("switch_open_fraction", "switch", "closed", "open_fraction"),
    ("trafo_tap_position_mean", "trafo", "tap_position", "mean"),
)
SHERLOCK_GLOBAL_COLUMNS: tuple[str, ...] = tuple(name for name, *_ in _AGG_SPECS)

SHARED_TRANSFER_COLUMNS: tuple[str, ...] = ("mean_bus_voltage_pu", "delta_mean_bus_voltage_pu")


@dataclass(frozen=True)
class SherlockStateRecord:
    """One state snapshot, ground truth stripped. NO `malicious` field --
    structurally, not by convention (mirrors `features.SliceObservation`)."""

    timestamp: float
    state: Mapping[str, float]


@dataclass(frozen=True)
class SherlockLabel:
    """Kept type-disjoint from `SherlockStateRecord`. `malicious=True` ONLY
    for a real attack event (a bare numeric id string) -- a benign event
    (`"<n> (benign event)"`) is a NEGATIVE example, same as `false`."""

    slice_index: int
    malicious: bool
    event_id: str | None


def read_ipal(path: Path) -> Iterator[dict]:
    """`.gz` or plain -> one `dict` per JSON line. Transparent gzip by
    extension, matching Sherlock's own `*.state.gz` naming."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_state_line(raw: Mapping, slice_index: int) -> tuple[SherlockStateRecord, SherlockLabel]:
    """Splits one real state-export JSON object into
    `(SherlockStateRecord, SherlockLabel)` -- the leak-barrier boundary. Only
    numeric/bool state values are kept (bool -> 1.0/0.0); everything else in
    `raw` (i.e. `malicious`) never reaches the record."""
    state = {
        k: (1.0 if v is True else 0.0 if v is False else float(v))
        for k, v in raw["state"].items()
        if isinstance(v, (int, float, bool))
    }
    record = SherlockStateRecord(timestamp=float(raw["timestamp"]), state=state)

    malicious_raw = raw.get("malicious", False)
    if malicious_raw is False:
        label = SherlockLabel(slice_index=slice_index, malicious=False, event_id=None)
    elif isinstance(malicious_raw, str) and "benign" in malicious_raw:
        # Verified against the REAL downloaded 01-Basic scenario: the train
        # file's benign marker is the bare string "benign-event" (no event
        # id, hyphenated), while the test file's is "<n> (benign event)"
        # (spaced, with id) -- both forms confirmed present, so this checks
        # the substring "benign" alone rather than either exact phrase.
        # Caught by TestParseStateLine::test_train_style_benign_event_hyphenated
        # after this exact bug (space-only match silently mis-scored 39
        # real train slices as attacks) was found running exp07 on real
        # data and traced to this line.
        label = SherlockLabel(slice_index=slice_index, malicious=False, event_id=None)
    else:
        label = SherlockLabel(slice_index=slice_index, malicious=True, event_id=str(malicious_raw))
    return record, label


def read_state_file(path: Path) -> tuple[tuple[SherlockStateRecord, ...], tuple[SherlockLabel, ...]]:
    """One real `*.state.gz` file -> `(records, labels)`, `slice_index`
    assigned by file position (1-based) -- valid because the real export's
    cadence is exactly uniform (verified: 1.0s, no gaps, in the downloaded
    01-Basic scenario); a future scenario with irregular cadence would need
    real-timestamp-based slicing instead, not assumed here silently."""
    records, labels = [], []
    for i, raw in enumerate(read_ipal(path), start=1):
        record, label = parse_state_line(raw, i)
        records.append(record)
        labels.append(label)
    return tuple(records), tuple(labels)


def component_attribute_values(record: SherlockStateRecord, component: str, attribute: str) -> list[float]:
    """Every real value in `record.state` matching `f"{component}.<N>:{attribute}"`,
    in whatever index order they appear -- never a hardcoded index list."""
    prefix_suffix = f".{attribute}"
    out = []
    for k, v in record.state.items():
        m = STATE_KEY_RE.match(k)
        if m and m.group("component") == component and m.group("attribute") == attribute:
            out.append(v)
    return out


def component_indices(records: Sequence[SherlockStateRecord], component: str) -> tuple[int, ...]:
    """Real, derived component indices (e.g. which `bus.N` exist) -- scans
    every record's keys (not just the first) so an index present only in
    later records is not silently missed."""
    idxs: set[int] = set()
    for record in records:
        for k in record.state:
            m = STATE_KEY_RE.match(k)
            if m and m.group("component") == component:
                idxs.add(int(m.group("index")))
    return tuple(sorted(idxs))


def per_bus_voltage(records: Sequence[SherlockStateRecord], bus_indices: Sequence[int]) -> torch.Tensor:
    """`[S, n_bus]`, real per-unit voltage per bus per slice, column order
    matching `bus_indices`. Raises if a bus's voltage is missing from a
    record (real data is dense at 1Hz; a genuine gap would mean this
    module's uniform-cadence assumption in `read_state_file` is wrong for
    that record, which must surface loudly, not silently zero-fill)."""
    out = torch.zeros((len(records), len(bus_indices)), dtype=torch.float32)
    for si, record in enumerate(records):
        for bi, b in enumerate(bus_indices):
            key = f"bus.{b}:voltage"
            if key not in record.state:
                raise ValueError(f"record {si} missing {key!r} -- uniform-cadence assumption violated")
            out[si, bi] = record.state[key]
    return out


def _aggregate(values: list[float], agg: str) -> float:
    if not values:
        return 0.0
    t = torch.tensor(values, dtype=torch.float32)
    if agg == "mean":
        return float(t.mean())
    if agg == "std":
        return float(t.std()) if len(values) > 1 else 0.0
    if agg == "min":
        return float(t.min())
    if agg == "max":
        return float(t.max())
    if agg == "open_fraction":
        return float(1.0 - t.mean())  # `closed` bool -> 1.0 - mean(closed) = fraction open
    raise ValueError(f"unknown aggregation {agg!r}")


def build_global_features(records: Sequence[SherlockStateRecord]) -> torch.Tensor:
    """`[S, len(SHERLOCK_GLOBAL_COLUMNS)]`. Every column is a plain
    aggregate (`_AGG_SPECS`) over REAL per-component values present in that
    slice -- a component/attribute pair absent from a given record
    contributes 0.0 for that slice (e.g. no `sgen` in a scenario without
    distributed generation), never fabricated."""
    out = torch.zeros((len(records), len(SHERLOCK_GLOBAL_COLUMNS)), dtype=torch.float32)
    for si, record in enumerate(records):
        for ci, (_, component, attribute, agg) in enumerate(_AGG_SPECS):
            values = component_attribute_values(record, component, attribute)
            out[si, ci] = _aggregate(values, agg)
    return out


def build_labels(labels: Sequence[SherlockLabel]) -> torch.Tensor:
    """`[S]` binary, directly from each record's own resolved `malicious`
    field -- no interval-overlap computation needed (module docstring: the
    real export already resolves ground truth per slice)."""
    return torch.tensor([float(l.malicious) for l in labels], dtype=torch.float32)


def build_shared_subspace(bus_voltage: torch.Tensor) -> torch.Tensor:
    """`[S, n_bus]` per-bus voltage -> `[S, 2]` (`SHARED_TRANSFER_COLUMNS`):
    mean bus voltage (pu) and its slice-to-slice delta (0.0 for slice 0).
    Mirrors `features.py`'s twin-side counterpart
    (`twin_bus_voltage_shared_subspace`) exactly -- both reduce their own
    domain's real per-bus voltage-pu channel the same way."""
    mean_v = bus_voltage.mean(dim=1)  # [S]
    delta = torch.zeros_like(mean_v)
    delta[1:] = mean_v[1:] - mean_v[:-1]
    return torch.stack([mean_v, delta], dim=-1)


def chronological_chunks(n: int, fracs: Mapping[str, float]) -> dict[str, tuple[int, int]]:
    """`n` items, in existing (real timestamp) order -> non-overlapping
    `{name: (start, end)}` index ranges cut in `fracs`' iteration order.
    Used to carve ONE real file's train chunk into (train, val, calib)
    without ever reordering by time (causality preserved by construction)."""
    out: dict[str, tuple[int, int]] = {}
    start = 0
    for name, frac in fracs.items():
        end = start + int(round(frac * n))
        out[name] = (start, end)
        start = end
    return out
