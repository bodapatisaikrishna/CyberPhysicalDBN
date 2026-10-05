"""Tests for src/perception/sherlock_physical.py with hand-computed expectations
on tiny hand-built zips (structure copied from the real physical.zip)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.perception.sherlock_loader import SHERLOCK_GLOBAL_COLUMNS
from src.perception.sherlock_physical import (
    attack_label_from_catalog,
    label_agreement_with_state_file,
    stream_physical_zip_features,
)


def _snapshot(ts: float, v1: float, v2: float, closed=(True, True), tap=0.0) -> dict:
    vals = {
        "bus.0.MEASUREMENT.voltage": v1, "bus.1.MEASUREMENT.voltage": v2,
        "line.0.MEASUREMENT.current_from": 10.0, "line.0.MEASUREMENT.active_power_from": 100.0,
        "load.0.MEASUREMENT.active_power": 5.0, "load.0.MEASUREMENT.reactive_power": 1.0,
        "sgen.0.MEASUREMENT.active_power": 2.0,
        "switch.0.CONFIGURATION.closed": closed[0], "switch.1.CONFIGURATION.closed": closed[1],
        "switch.0.MEASUREMENT.is_closed": True,  # must NOT feed the switch column
        "trafo.0.MEASUREMENT.tap_position": tap, "trafo.0.CONFIGURATION.tap_position": 99.0,
    }
    return {"timestamp": ts, "values": vals}


def _zip(tmp_path: Path, snaps: list[dict]) -> Path:
    p = tmp_path / "physical.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("physical/", "")
        for i, s in enumerate(snaps):
            z.writestr(f"physical/power_grid_x_TS{int(s['timestamp'])}-{i}.json", json.dumps(s))
    return p


def test_features_hand_computed_and_sorted(tmp_path):
    # written out of time order on purpose; NaN voltage must be excluded, not imputed
    snaps = [_snapshot(1002.0, 1.0, float("nan"), closed=(True, False)), _snapshot(1000.0, 1.0, 1.2)]
    f = stream_physical_zip_features(_zip(tmp_path, snaps), [])
    assert list(f.timestamps) == [1000.0, 1002.0]
    col = {c: i for i, c in enumerate(SHERLOCK_GLOBAL_COLUMNS)}
    assert f.features[0, col["bus_voltage_pu_mean"]] == pytest.approx(1.1)
    assert f.features[0, col["bus_voltage_pu_min"]] == pytest.approx(1.0)
    assert f.features[0, col["bus_voltage_pu_max"]] == pytest.approx(1.2)
    assert f.features[1, col["bus_voltage_pu_mean"]] == pytest.approx(1.0)  # NaN bus excluded
    assert f.features[1, col["switch_open_fraction"]] == pytest.approx(0.5)
    assert f.features[0, col["switch_open_fraction"]] == pytest.approx(0.0)
    assert f.features[0, col["trafo_tap_position_mean"]] == 0.0  # MEASUREMENT group, not CONFIGURATION's 99
    assert f.n_nonfinite_values == 1
    assert torch.isfinite(f.features).all()


def test_labels_follow_catalog_and_benign_is_negative(tmp_path):
    snaps = [_snapshot(t, 1.0, 1.0) for t in (1000.0, 1002.0, 1004.0, 1006.0, 1008.0)]
    cat = [
        {"id": "0", "start": 1002.0, "end": 1006.0},            # [start, end): 1002, 1004
        {"id": "1 (benign event)", "start": 1006.0, "end": 1010.0},
    ]
    f = stream_physical_zip_features(_zip(tmp_path, snaps), cat)
    assert f.labels.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0]
    assert f.event_ids == (None, "0", "0", None, None)
    assert f.raw_label_counts == {"False": 3, "0": 2}


def test_attack_label_from_catalog_direct():
    y, ids = attack_label_from_catalog(np.array([1.0, 2.0, 3.0]), [{"id": "7", "start": 2.0, "end": 3.0}])
    assert y.tolist() == [0.0, 1.0, 0.0] and ids == (None, "7", None)


def test_empty_column_raises(tmp_path):
    snap = {"timestamp": 1.0, "values": {"bus.0.MEASUREMENT.voltage": 1.0}}
    with pytest.raises(ValueError, match="no values for columns"):
        stream_physical_zip_features(_zip(tmp_path, [snap]), [])


def test_label_agreement():
    phys_ts = np.array([0.0, 2.0, 4.0, 6.0])
    phys_y = np.array([0, 1, 1, 0])
    state_ts = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    state_y = np.array([0, 0, 1, 1, 1, 0, 1])  # last record disagrees (nearest phys t=6 -> 0)
    r = label_agreement_with_state_file(phys_ts, phys_y, state_ts, state_y)
    assert r["n_compared"] == 7 and r["n_disagree"] >= 1 and 0.0 < r["agreement"] < 1.0
