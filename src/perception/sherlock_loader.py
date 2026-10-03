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

This is POWER-GRID STATE keyed by semantic component name (confirmed against
`raw/train/data-point-map.json`, which maps each real IEC-104 point address to
exactly this `element:attribute` naming), not IEC-104 message traffic with
`src`/`dest`/`activity` fields. CORRECTION (2026-09-28, after reading the
dataset paper, Sec. 3.6): it is the state a PASSIVE VANTAGE POINT reconstructs
from intercepted IEC-104 packets (initial state + packet updates, logged once
a second) -- the state AS THE NETWORK CARRIED IT, not raw simulator truth. So a
denial of service that stops updates shows up as a frozen state, and a
man-in-the-middle measurement manipulation shows up as the manipulated value.
Calling it "physical telemetry", as this docstring and exp07 originally did,
was loose.

SCENARIOS AND DATA DEFECTS (exp13, LAB_NOTEBOOK.md 2026-09-28). The same
schema holds for `02-Semiurban` (3,565 keys per record) and `03-Rural` (1,894),
but: 02's shipped state files are TRUNCATED (~3 h of the 12 h the paper states;
22 of its 29 catalogued test attacks are not in the export) and each ends in a
line cut off mid-number, which `read_ipal_tolerant` drops and reports (a
malformed line anywhere else still raises); 03's only state file is named
"train" but is the attack data (there is no clean 03 split).
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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
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


def read_ipal_tolerant(path: Path, stats: dict | None = None) -> Iterator[dict]:
    """Like `read_ipal`, but tolerates exactly ONE kind of damage: a final line
    that does not parse. Verified on the real 02-Semiurban train file
    (2026-09-28): the shipped `train.n406.state.gz` holds 11,116 complete
    records and then ends mid-number (a writer cut off; the zip member's own
    CRC-32 matches, so this is the dataset's bytes, not a bad download).
    A malformed line ANYWHERE ELSE still raises `json.JSONDecodeError` -- that
    would be corruption, not a truncated tail, and must not be skipped
    silently. When a tail is dropped, `stats["truncated_tail_chars"]` records
    how many characters, so callers can report it."""
    opener = gzip.open if str(path).endswith(".gz") else open
    pending: str | None = None
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if pending is not None:
                yield json.loads(pending)
            pending = line
    if pending is not None:
        try:
            yield json.loads(pending)
        except json.JSONDecodeError:
            if stats is not None:
                stats["truncated_tail_chars"] = len(pending)


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

    malicious, event_id = _label_from_raw(raw.get("malicious", False))
    label = SherlockLabel(slice_index=slice_index, malicious=malicious, event_id=event_id)
    return record, label


def _label_from_raw(malicious_raw) -> tuple[bool, str | None]:
    """`(is_real_attack, event_id)` from one record's raw `malicious` field.
    Single source of truth shared by `parse_state_line` and
    `stream_state_file_features`, so the two can never disagree on labels."""
    if malicious_raw is False:
        return False, None
    if isinstance(malicious_raw, str) and "benign" in malicious_raw:
        # Verified against the REAL downloaded 01-Basic scenario: the train
        # file's benign marker is the bare string "benign-event" (no event
        # id, hyphenated), while the test file's is "<n> (benign event)"
        # (spaced, with id) -- both forms confirmed present, so this checks
        # the substring "benign" alone rather than either exact phrase.
        # Caught by TestParseStateLine::test_train_style_benign_event_hyphenated
        # after this exact bug (space-only match silently mis-scored 39
        # real train slices as attacks) was found running exp07 on real
        # data and traced to this line.
        return False, None
    return True, str(malicious_raw)


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


@dataclass(frozen=True)
class StateFileFeatures:
    """Everything the all-scenario experiment needs from one `*.state.gz`,
    computed in a single streaming pass (no per-record dicts retained)."""

    features: torch.Tensor  # [S, len(SHERLOCK_GLOBAL_COLUMNS)] -- identical to build_global_features
    labels: torch.Tensor  # [S] binary, real attack only (benign events are negatives)
    timestamps: np.ndarray  # [S] float64, as shipped
    event_ids: tuple[str | None, ...]  # per slice; None unless a real attack is active
    raw_label_counts: dict[str, int]  # Counter over str(raw `malicious`) -- the real label vocabulary
    component_key_counts: dict[str, int]  # distinct state keys per component family, first record
    n_state_keys_first_record: int
    truncated_tail_chars: int = 0  # >0: the file's last line was cut off mid-record and dropped (see read_ipal_tolerant)


def stream_state_file_features(path: Path, *, max_records: int | None = None) -> StateFileFeatures:
    """One streaming pass over a real `*.state.gz`: aggregate feature vector,
    label, timestamp and label-vocabulary bookkeeping per record, WITHOUT
    keeping each record's ~400-key dict alive (the record-list route,
    `read_state_file` + `build_global_features`, holds every dict in memory
    -- fine for 01-Basic's 43,204 lines, not for the much larger
    02-Semiurban / 03-Rural files).

    Numerically identical to `build_global_features(read_state_file(path)[0])`
    BY CONSTRUCTION: values are gathered in the same (dict) order and reduced
    by the very same `_aggregate` -- only the per-key regex work is cached
    (key -> the spec columns it feeds), not re-run 11x per key per record.
    `tests/test_sherlock_loader.py::TestStreamStateFileFeatures` asserts exact
    tensor equality against the record-list route rather than trusting this
    docstring.

    `max_records` reads a real prefix (smoke runs) -- never synthetic data."""
    key_to_specs: dict[str, tuple[int, ...]] = {}
    n_specs = len(_AGG_SPECS)
    block_rows = 1 << 16
    blocks: list[np.ndarray] = []
    block = np.empty((block_rows, n_specs), dtype=np.float32)
    row = 0
    labels: list[float] = []
    timestamps: list[float] = []
    event_ids: list[str | None] = []
    label_counts: Counter[str] = Counter()
    component_key_counts: Counter[str] = Counter()
    n_keys_first = 0

    reader_stats: dict = {}
    for i, raw in enumerate(read_ipal_tolerant(path, reader_stats)):
        if max_records is not None and i >= max_records:
            break
        per_spec: list[list[float]] = [[] for _ in _AGG_SPECS]
        for k, v in raw["state"].items():
            if not isinstance(v, (int, float, bool)):
                continue
            specs = key_to_specs.get(k)
            if specs is None:
                m = STATE_KEY_RE.match(k)
                specs = tuple(
                    si for si, (_, component, attribute, _) in enumerate(_AGG_SPECS)
                    if m and m.group("component") == component and m.group("attribute") == attribute
                )
                key_to_specs[k] = specs
            if specs:
                fv = 1.0 if v is True else 0.0 if v is False else float(v)
                for si in specs:
                    per_spec[si].append(fv)
        if row == block_rows:  # numpy blocks, not a Python list of floats: keeps memory flat on 1M+ line files
            blocks.append(block)
            block = np.empty((block_rows, n_specs), dtype=np.float32)
            row = 0
        for si, (vals, spec) in enumerate(zip(per_spec, _AGG_SPECS)):
            block[row, si] = _aggregate(vals, spec[3])
        row += 1

        malicious_raw = raw.get("malicious", False)
        malicious, event_id = _label_from_raw(malicious_raw)
        labels.append(float(malicious))
        event_ids.append(event_id)
        label_counts[str(malicious_raw)] += 1
        timestamps.append(float(raw["timestamp"]))

        if i == 0:
            n_keys_first = len(raw["state"])
            for k in raw["state"]:
                m = STATE_KEY_RE.match(k)
                component_key_counts[m.group("component") if m else "<unparsed>"] += 1

    if row == 0 and not blocks:
        raise ValueError(f"no records read from {path}")
    blocks.append(block[:row])
    return StateFileFeatures(
        features=torch.from_numpy(np.concatenate(blocks, axis=0)),
        labels=torch.tensor(labels, dtype=torch.float32),
        timestamps=np.asarray(timestamps, dtype=np.float64),
        event_ids=tuple(event_ids),
        raw_label_counts=dict(label_counts),
        component_key_counts=dict(component_key_counts),
        n_state_keys_first_record=n_keys_first,
        truncated_tail_chars=int(reader_stats.get("truncated_tail_chars", 0)),
    )


def build_shared_subspace(bus_voltage: torch.Tensor) -> torch.Tensor:
    """`[S, n_bus]` per-bus voltage -> `[S, 2]` (`SHARED_TRANSFER_COLUMNS`):
    mean bus voltage (pu) and its slice-to-slice delta (0.0 for slice 0).
    Mirrors `features.py`'s twin-side counterpart
    (`twin_bus_voltage_shared_subspace`) exactly -- both reduce their own
    domain's real per-bus voltage-pu channel the same way."""
    return shared_subspace_from_mean_voltage(bus_voltage.mean(dim=1))


def shared_subspace_from_mean_voltage(mean_bus_voltage_pu: torch.Tensor) -> torch.Tensor:
    """`[S]` per-slice mean bus voltage (pu) -> `[S, 2]` (`SHARED_TRANSFER_COLUMNS`).
    Exactly the reduction `build_shared_subspace` applies after averaging over
    buses; split out so a caller that already holds the streamed
    `bus_voltage_pu_mean` aggregate (`stream_state_file_features`) need not
    keep a `[S, n_bus]` matrix around. `build_shared_subspace` now calls this,
    so the two cannot drift (`tests/test_sherlock_loader.py::TestSharedSubspace`)."""
    delta = torch.zeros_like(mean_bus_voltage_pu)
    delta[1:] = mean_bus_voltage_pu[1:] - mean_bus_voltage_pu[:-1]
    return torch.stack([mean_bus_voltage_pu, delta], dim=-1)


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
