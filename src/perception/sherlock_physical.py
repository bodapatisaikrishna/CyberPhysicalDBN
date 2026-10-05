"""Full-length 12 h physical snapshots of a Sherlock scenario -> the same 11
aggregate features exp13 uses (LAB_NOTEBOOK.md 2026-10-03).

Why this exists: 02-Semiurban's exported `*.state.gz` is truncated to ~3 h of a
12 h run (7 of 29 catalogued test attacks). The raw `raw/<split>/physical.zip`
holds one JSON snapshot per ~2 s for the whole 12 h
(`{"timestamp": float, "values": {"<component>.<N>.<GROUP>.<attribute>": v}}`),
so it is the only place the other 22 attacks can be scored.

What is DIFFERENT from `sherlock_loader.stream_state_file_features`, stated so
no caller treats the two as interchangeable:
  * Source is the simulator-side physical export (178 buses, 126 lines, ...),
    not the network-reconstructed state (a subset of components; paper Sec. 3.6).
    The same aggregate over a different component set is a different feature,
    so physical-view and state-view features must not be mixed across arms.
  * Cadence is ~2 s (median 2.0, jitter 1-4 s), not 1 Hz.
  * Snapshots carry NO `malicious` field. Labels come from the dataset's own
    event catalog (`ipal/<split>/events.json`): a snapshot is an attack slice
    iff `start <= timestamp < end` of a catalogued event whose id is not a
    "benign event". `label_agreement_with_state_file` measures how well that
    rule reproduces the state file's own labels on the time span both cover --
    verified, not assumed.

Column order/aggregation reuse `sherlock_loader._AGG_SPECS`, so the feature
columns are the same 11 named quantities.
"""

from __future__ import annotations

import json
import math
import re
import zipfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from src.perception.sherlock_loader import (
    _AGG_SPECS,
    SHERLOCK_GLOBAL_COLUMNS,
    StateFileFeatures,
    _aggregate,
)

PHYSICAL_KEY_RE = re.compile(
    r"^(?P<component>[a-z_]+)\.(?P<index>\d+)\.(?P<group>CONFIGURATION|MEASUREMENT)\.(?P<attribute>.+)$"
)

# Spec column -> the physical group its attribute lives in. Chosen by inspecting
# the real keys: the state export's `switch.N:closed` is `CONFIGURATION.closed`
# here (`MEASUREMENT.is_closed` also exists); everything else is MEASUREMENT.
_GROUP_FOR: dict[tuple[str, str], str] = {
    ("switch", "closed"): "CONFIGURATION",
}


def _group_for(component: str, attribute: str) -> str:
    return _GROUP_FOR.get((component, attribute), "MEASUREMENT")


def attack_label_from_catalog(
    timestamps: np.ndarray, catalog: Sequence[Mapping]
) -> tuple[torch.Tensor, tuple[str | None, ...]]:
    """`(labels[S], event_ids[S])`: 1 / the event id iff the snapshot lies in
    `[start, end)` of a catalogued event that is not a benign event. Benign
    events are negatives, same as `sherlock_loader._label_from_raw`."""
    labels = np.zeros(len(timestamps), dtype=np.float32)
    ids: list[str | None] = [None] * len(timestamps)
    for ev in catalog:
        eid = str(ev["id"])
        if "benign" in eid:
            continue
        hit = np.flatnonzero((timestamps >= float(ev["start"])) & (timestamps < float(ev["end"])))
        for i in hit:
            labels[i] = 1.0
            ids[i] = eid
    return torch.from_numpy(labels), tuple(ids)


def stream_physical_zip_features(
    zip_path: Path, catalog: Sequence[Mapping], *, max_records: int | None = None
) -> StateFileFeatures:
    """One pass over `physical.zip` (no extraction to disk), snapshots sorted by
    their own `timestamp`, 11 aggregates per snapshot via the SAME `_aggregate`
    reducers as the state-file route.

    Non-finite source values (the real files contain NaN bus voltages -- islanded
    buses) are EXCLUDED from the aggregate of that snapshot and counted in
    `StateFileFeatures.n_nonfinite_values`; nothing is imputed.

    A component/attribute absent from a snapshot contributes 0.0 for that
    column (same convention as `build_global_features`); a column that is empty
    in EVERY snapshot raises, since that means the key mapping is wrong rather
    than the data sparse.

    `max_records` reads the first N snapshots in file order (smoke only)."""
    z = zipfile.ZipFile(zip_path)
    members = [n for n in z.namelist() if n.startswith("physical/power_grid") and n.endswith(".json")]
    if not members:
        raise ValueError(f"no physical/power_grid*.json members in {zip_path}")
    # zip order is NOT chronological; sort by the filename's unix-second stamp so a
    # `max_records` prefix is a contiguous real stretch of time (smoke), not a scatter.
    members.sort(key=lambda n: int(re.search(r"TS(\d+)-", n).group(1)))
    if max_records is not None:
        members = members[:max_records]

    key_to_specs: dict[str, tuple[int, ...]] = {}
    n_specs = len(_AGG_SPECS)
    rows: list[tuple[float, np.ndarray]] = []
    component_key_counts: dict[str, int] = {}
    n_keys_first = 0
    nonempty = np.zeros(n_specs, dtype=bool)
    n_nonfinite = 0
    for i, name in enumerate(members):
        raw = json.loads(z.read(name))
        values = raw["values"]
        per_spec: list[list[float]] = [[] for _ in _AGG_SPECS]
        for k, v in values.items():
            if not isinstance(v, (int, float, bool)):
                continue
            specs = key_to_specs.get(k)
            if specs is None:
                m = PHYSICAL_KEY_RE.match(k)
                specs = tuple(
                    si for si, (_, comp, attr, _) in enumerate(_AGG_SPECS)
                    if m and m.group("component") == comp and m.group("attribute") == attr
                    and m.group("group") == _group_for(comp, attr)
                )
                key_to_specs[k] = specs
            if specs:
                fv = 1.0 if v is True else 0.0 if v is False else float(v)
                if not math.isfinite(fv):
                    n_nonfinite += 1  # de-energised/islanded component: pandapower reports NaN. Excluded, counted, never imputed.
                    continue
                for si in specs:
                    per_spec[si].append(fv)
        vec = np.zeros(n_specs, dtype=np.float32)
        for si, (vals, spec) in enumerate(zip(per_spec, _AGG_SPECS)):
            vec[si] = _aggregate(vals, spec[3])
            nonempty[si] |= bool(vals)
        rows.append((float(raw["timestamp"]), vec))
        if i == 0:
            n_keys_first = len(values)
            for k in values:
                m = PHYSICAL_KEY_RE.match(k)
                c = m.group("component") if m else "<unparsed>"
                component_key_counts[c] = component_key_counts.get(c, 0) + 1
    if not nonempty.all():
        empty = [SHERLOCK_GLOBAL_COLUMNS[i] for i in np.flatnonzero(~nonempty)]
        raise ValueError(f"physical key mapping produced no values for columns {empty} in any snapshot")

    rows.sort(key=lambda r: r[0])
    ts = np.asarray([r[0] for r in rows], dtype=np.float64)
    feats = torch.from_numpy(np.stack([r[1] for r in rows]))
    labels, event_ids = attack_label_from_catalog(ts, catalog)
    counts: dict[str, int] = {"False": int((labels == 0).sum())}
    for e in event_ids:
        if e is not None:
            counts[e] = counts.get(e, 0) + 1
    return StateFileFeatures(
        features=feats, labels=labels, timestamps=ts, event_ids=event_ids,
        raw_label_counts=counts, component_key_counts=component_key_counts,
        n_state_keys_first_record=n_keys_first, n_nonfinite_values=n_nonfinite,
    )


def label_agreement_with_state_file(
    phys_ts: np.ndarray, phys_labels: np.ndarray, state_ts: np.ndarray, state_labels: np.ndarray
) -> dict:
    """How well the catalog rule reproduces a real state file's own labels on
    the span both cover: each state record is matched to the nearest physical
    snapshot in time (|dt| <= 2 s) and the two binary labels compared."""
    lo, hi = max(phys_ts[0], state_ts[0]), min(phys_ts[-1], state_ts[-1])
    sel = (state_ts >= lo) & (state_ts <= hi)
    s_ts, s_y = state_ts[sel], state_labels[sel]
    idx = np.clip(np.searchsorted(phys_ts, s_ts), 1, len(phys_ts) - 1)
    nearer_left = np.abs(phys_ts[idx - 1] - s_ts) <= np.abs(phys_ts[idx] - s_ts)
    idx = np.where(nearer_left, idx - 1, idx)
    ok = np.abs(phys_ts[idx] - s_ts) <= 2.0
    p_y = phys_labels[idx]
    return {
        "n_compared": int(ok.sum()),
        "agreement": float((p_y[ok] == s_y[ok]).mean()) if ok.any() else float("nan"),
        "n_state_attack": int((s_y[ok] > 0).sum()),
        "n_phys_attack": int((p_y[ok] > 0).sum()),
        "n_disagree": int((p_y[ok] != s_y[ok]).sum()),
    }


# --- component view ---------------------------------------------------------
# Every numeric quantity of the physical export, instead of 11 grid-wide means.
# Chosen a priori from the key schema (not from any label): all MEASUREMENT keys
# plus the two CONFIGURATION keys that carry operating state (switch `closed`,
# trafo `tap_position`); the remaining CONFIGURATION keys are setpoints/profile
# scalings and static flags.
_COMPONENT_CONFIG_KEEP = {("switch", "closed"), ("trafo", "tap_position")}


def _keep_component_key(m: re.Match) -> bool:
    return m.group("group") == "MEASUREMENT" or (m.group("component"), m.group("attribute")) in _COMPONENT_CONFIG_KEEP


def stream_physical_zip_components(
    zip_path: Path, catalog: Sequence[Mapping], *, max_records: int | None = None
) -> StateFileFeatures:
    """Per-component view of `physical.zip`: one column per kept key (see
    `_keep_component_key`; column order = sorted key names of the first snapshot),
    followed by explicit non-finite-count columns (`nonfinite_total` and
    `nonfinite_<component>` per component family). NaN source values are KEPT as
    NaN in `features` (nothing imputed here) and counted -- the consumer decides
    how to fill them (the detector fits column medians on clean data only).
    A key missing from a later snapshot is NaN and counted too.

    Labels and ordering exactly as `stream_physical_zip_features`."""
    z = zipfile.ZipFile(zip_path)
    members = [n for n in z.namelist() if n.startswith("physical/power_grid") and n.endswith(".json")]
    if not members:
        raise ValueError(f"no physical/power_grid*.json members in {zip_path}")
    members.sort(key=lambda n: int(re.search(r"TS(\d+)-", n).group(1)))
    if max_records is not None:
        members = members[:max_records]

    first = json.loads(z.read(members[0]))["values"]
    cols: list[str] = []
    fam_of: dict[str, str] = {}
    for k in sorted(first):
        m = PHYSICAL_KEY_RE.match(k)
        if m and _keep_component_key(m) and isinstance(first[k], (int, float, bool)):
            cols.append(k)
            fam_of[k] = m.group("component")
    fams = sorted(set(fam_of.values()))
    names = tuple(cols) + ("nonfinite_total",) + tuple(f"nonfinite_{f}" for f in fams)
    col_idx = {k: i for i, k in enumerate(cols)}
    fam_cols = {f: np.array([col_idx[k] for k in cols if fam_of[k] == f]) for f in fams}
    n_cols = len(cols)

    X = np.full((len(members), len(names)), np.nan, dtype=np.float32)
    ts = np.empty(len(members), dtype=np.float64)
    n_nonfinite = 0
    for i, name in enumerate(members):
        raw = json.loads(z.read(name))
        vals = raw["values"]
        row = np.full(n_cols, np.nan, dtype=np.float32)
        for k, j in col_idx.items():
            v = vals.get(k)
            if isinstance(v, (int, float, bool)):
                row[j] = 1.0 if v is True else 0.0 if v is False else float(v)
        bad = ~np.isfinite(row)
        X[i, :n_cols] = row
        X[i, n_cols] = float(bad.sum())
        for fi, f in enumerate(fams):
            X[i, n_cols + 1 + fi] = float(bad[fam_cols[f]].sum())
        n_nonfinite += int(bad.sum())
        ts[i] = float(raw["timestamp"])
    order = np.argsort(ts, kind="stable")
    ts, X = ts[order], X[order]
    labels, event_ids = attack_label_from_catalog(ts, catalog)
    counts: dict[str, int] = {"False": int((labels == 0).sum())}
    for e in event_ids:
        if e is not None:
            counts[e] = counts.get(e, 0) + 1
    return StateFileFeatures(
        features=torch.from_numpy(X), labels=labels, timestamps=ts, event_ids=event_ids,
        raw_label_counts=counts, component_key_counts={f: int(len(fam_cols[f])) for f in fams},
        n_state_keys_first_record=len(first), column_names=names, n_nonfinite_values=n_nonfinite,
    )
