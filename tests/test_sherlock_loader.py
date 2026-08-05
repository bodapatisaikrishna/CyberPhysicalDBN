"""Tests for src/perception/sherlock_loader.py.

Fixtures match the REAL schema verified against the downloaded
`data/sherlock/01-Basic/train.n302.state.gz` (see this module's own
docstring and LAB_NOTEBOOK.md 2026-08-05): one JSON object per line,
`{"timestamp":..., "state": {"bus.0:voltage":..., ...}, "malicious": false |
"<n> (benign event)" | "<n>"}`. No download is required to run these --
the fixtures are hand-built to match the verified real shape exactly.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import inspect

import pytest
import torch

from src.perception.features import BUS_DYNAMIC_COLUMNS, twin_bus_voltage_shared_subspace
from src.perception.sherlock_loader import (
    SHARED_TRANSFER_COLUMNS,
    SHERLOCK_GLOBAL_COLUMNS,
    SherlockLabel,
    SherlockStateRecord,
    build_global_features,
    build_labels,
    build_shared_subspace,
    chronological_chunks,
    component_attribute_values,
    component_indices,
    parse_state_line,
    per_bus_voltage,
    read_ipal,
    read_state_file,
)


def _raw_line(**overrides) -> dict:
    base = {
        "timestamp": 1741555756.202053,
        "state": {
            "bus.0:voltage": 1.02, "bus.0:voltage_angle": -0.0005,
            "bus.1:voltage": 0.99, "bus.1:voltage_angle": -1.2,
            "line.0:active_power_from": 3122710.5, "line.0:current_from": 98.2,
            "switch.8:closed": True,
            "load.0:active_power": 9955330.4, "load.0:reactive_power": 3044714.8,
            "sgen.5:active_power": 500000.0,
            "trafo.0:tap_position": 0,
        },
        "malicious": False,
    }
    base.update(overrides)
    return base


class TestReadIpal:
    def test_gzipped_jsonlines(self, tmp_path):
        p = tmp_path / "train.n2.state.gz"
        with gzip.open(p, "wt") as f:
            for i in range(3):
                f.write(json.dumps(_raw_line(timestamp=1000.0 + i)) + "\n")
        rows = list(read_ipal(p))
        assert len(rows) == 3
        assert rows[1]["timestamp"] == 1001.0

    def test_plain_jsonlines(self, tmp_path):
        p = tmp_path / "train.n2.state"
        p.write_text("\n".join(json.dumps(_raw_line(timestamp=float(i))) for i in range(2)))
        assert len(list(read_ipal(p))) == 2


class TestParseStateLine:
    def test_strips_malicious_from_record(self):
        record, _ = parse_state_line(_raw_line(malicious=False), slice_index=1)
        assert not hasattr(record, "malicious")
        assert set(f.name for f in dataclasses.fields(record)) == {"timestamp", "state"}

    def test_bool_state_values_cast_to_float(self):
        record, _ = parse_state_line(_raw_line(), slice_index=1)
        assert record.state["switch.8:closed"] == 1.0

    def test_malicious_false_is_negative(self):
        _, label = parse_state_line(_raw_line(malicious=False), slice_index=1)
        assert label.malicious is False
        assert label.event_id is None

    def test_benign_event_is_negative(self):
        _, label = parse_state_line(_raw_line(malicious="27 (benign event)"), slice_index=1)
        assert label.malicious is False
        assert label.event_id is None

    def test_train_style_benign_event_hyphenated(self):
        """Real bug found running exp07 on the actual downloaded data: the
        train file's benign marker is the bare hyphenated string
        "benign-event" (no id), not "<n> (benign event)" like the test
        file. A space-only substring check silently scored 39 real train
        slices as attacks (inflated train base rate from 0.0 to ~0.0009)."""
        _, label = parse_state_line(_raw_line(malicious="benign-event"), slice_index=1)
        assert label.malicious is False
        assert label.event_id is None

    def test_bare_numeric_id_is_positive(self):
        _, label = parse_state_line(_raw_line(malicious="14"), slice_index=5)
        assert label.malicious is True
        assert label.event_id == "14"
        assert label.slice_index == 5


class TestReadStateFile:
    def test_slice_index_by_file_position(self, tmp_path):
        p = tmp_path / "train.n2.state.gz"
        with gzip.open(p, "wt") as f:
            for i in range(3):
                f.write(json.dumps(_raw_line(timestamp=1000.0 + i, malicious="7" if i == 1 else False)) + "\n")
        records, labels = read_state_file(p)
        assert [l.slice_index for l in labels] == [1, 2, 3]
        assert [l.malicious for l in labels] == [False, True, False]
        assert len(records) == 3


class TestLeakGuard:
    """Mirrors tests/test_perception_features.py's SliceObservation barrier
    tests: no function that computes FEATURES may take a label-shaped
    argument, and perturbing the label must never change a feature output."""

    def test_state_record_has_no_label_fields(self):
        names = {f.name for f in dataclasses.fields(SherlockStateRecord)}
        assert "malicious" not in names
        assert "event_id" not in names

    def test_record_and_label_disjoint_except_join_key(self):
        record_fields = {f.name for f in dataclasses.fields(SherlockStateRecord)}
        label_fields = {f.name for f in dataclasses.fields(SherlockLabel)}
        assert record_fields.isdisjoint(label_fields)  # SherlockStateRecord has no slice_index at all

    def test_build_global_features_signature_carries_no_label(self):
        sig = inspect.signature(build_global_features)
        for name in sig.parameters:
            assert "label" not in name and "malicious" not in name

    def test_global_features_bitwise_invariant_to_label(self):
        records1, _ = read_state_file_from_lines([_raw_line(malicious=False), _raw_line(malicious="9")])
        records2, _ = read_state_file_from_lines([_raw_line(malicious=False), _raw_line(malicious=False)])
        # Same state values, different malicious -> identical features.
        a = build_global_features(records1)
        b = build_global_features(records2)
        assert torch.equal(a, b)


def read_state_file_from_lines(raw_lines: list[dict]):
    records, labels = [], []
    for i, raw in enumerate(raw_lines, start=1):
        r, l = parse_state_line(raw, i)
        records.append(r)
        labels.append(l)
    return tuple(records), tuple(labels)


class TestComponentHelpers:
    def test_component_attribute_values_matches_real_key_shape(self):
        record, _ = parse_state_line(_raw_line(), slice_index=1)
        values = component_attribute_values(record, "bus", "voltage")
        assert sorted(values) == sorted([1.02, 0.99])

    def test_component_indices_derived_not_hardcoded(self):
        records, _ = read_state_file_from_lines([_raw_line(), _raw_line()])
        assert component_indices(records, "bus") == (0, 1)
        assert component_indices(records, "sgen") == (5,)

    def test_component_indices_empty_for_absent_component(self):
        records, _ = read_state_file_from_lines([_raw_line()])
        assert component_indices(records, "transformer") == ()


class TestPerBusVoltage:
    def test_shape_and_values(self):
        records, _ = read_state_file_from_lines([_raw_line(), _raw_line()])
        x = per_bus_voltage(records, [0, 1])
        assert x.shape == (2, 2)
        assert x[0].tolist() == pytest.approx([1.02, 0.99])

    def test_missing_bus_raises(self):
        records, _ = read_state_file_from_lines([_raw_line()])
        with pytest.raises(ValueError, match="bus.99"):
            per_bus_voltage(records, [0, 99])


class TestBuildGlobalFeatures:
    def test_shape(self):
        records, _ = read_state_file_from_lines([_raw_line()])
        x = build_global_features(records)
        assert x.shape == (1, len(SHERLOCK_GLOBAL_COLUMNS))

    def test_bus_voltage_aggregates(self):
        records, _ = read_state_file_from_lines([_raw_line()])
        x = build_global_features(records)
        cols = {c: i for i, c in enumerate(SHERLOCK_GLOBAL_COLUMNS)}
        assert x[0, cols["bus_voltage_pu_mean"]] == pytest.approx((1.02 + 0.99) / 2)
        assert x[0, cols["bus_voltage_pu_min"]] == pytest.approx(0.99)
        assert x[0, cols["bus_voltage_pu_max"]] == pytest.approx(1.02)

    def test_switch_open_fraction(self):
        raw = _raw_line()
        raw["state"]["switch.9:closed"] = False
        records, _ = read_state_file_from_lines([raw])
        x = build_global_features(records)
        cols = {c: i for i, c in enumerate(SHERLOCK_GLOBAL_COLUMNS)}
        assert x[0, cols["switch_open_fraction"]] == pytest.approx(0.5)  # 1 of 2 switches open

    def test_absent_component_contributes_zero(self):
        raw = _raw_line()
        del raw["state"]["trafo.0:tap_position"]
        records, _ = read_state_file_from_lines([raw])
        x = build_global_features(records)
        cols = {c: i for i, c in enumerate(SHERLOCK_GLOBAL_COLUMNS)}
        assert x[0, cols["trafo_tap_position_mean"]] == 0.0


class TestBuildLabels:
    def test_matches_label_malicious_field(self):
        _, labels = read_state_file_from_lines([
            _raw_line(malicious=False), _raw_line(malicious="14"), _raw_line(malicious="3 (benign event)"),
        ])
        y = build_labels(labels)
        assert y.tolist() == [0.0, 1.0, 0.0]


class TestSharedSubspace:
    def test_shape_and_delta(self):
        bus_voltage = torch.tensor([[1.0, 1.0], [1.0, 0.9], [1.0, 1.1]])  # mean: 1.0, 0.95, 1.05
        reduced = build_shared_subspace(bus_voltage)
        assert reduced.shape == (3, 2)
        assert reduced[0].tolist() == [1.0, 0.0]
        assert reduced[1, 0].item() == pytest.approx(0.95)
        assert reduced[1, 1].item() == pytest.approx(-0.05)
        assert reduced[2, 1].item() == pytest.approx(0.10)

    def test_twin_side_matches_shape(self):
        bus_dynamic = torch.zeros(4, 3, len(BUS_DYNAMIC_COLUMNS))
        reduced = twin_bus_voltage_shared_subspace(bus_dynamic)
        assert reduced.shape == (4, 2)

    def test_twin_side_rejects_wrong_width(self):
        with pytest.raises(ValueError):
            twin_bus_voltage_shared_subspace(torch.zeros(4, 3, 2))

    def test_twin_side_selects_correct_columns(self):
        c = {name: i for i, name in enumerate(BUS_DYNAMIC_COLUMNS)}
        bus_dynamic = torch.zeros(2, 2, len(BUS_DYNAMIC_COLUMNS))
        bus_dynamic[0, :, c["vm_pu"]] = torch.tensor([1.0, 0.98])
        bus_dynamic[0, :, c["delta_vm_pu"]] = torch.tensor([0.01, -0.02])
        reduced = twin_bus_voltage_shared_subspace(bus_dynamic)
        assert reduced[0, 0].item() == pytest.approx(0.99)
        assert reduced[0, 1].item() == pytest.approx(-0.005)

    def test_both_domains_same_column_semantics(self):
        assert len(SHARED_TRANSFER_COLUMNS) == 2
        assert SHARED_TRANSFER_COLUMNS == ("mean_bus_voltage_pu", "delta_mean_bus_voltage_pu")


class TestChronologicalChunks:
    def test_fractions_partition_without_overlap(self):
        chunks = chronological_chunks(100, {"train": 0.6, "val": 0.2, "calib": 0.2})
        assert chunks["train"] == (0, 60)
        assert chunks["val"] == (60, 80)
        assert chunks["calib"] == (80, 100)

    def test_covers_full_range_for_typical_fracs(self):
        chunks = chronological_chunks(10, {"a": 0.5, "b": 0.5})
        assert chunks["a"][0] == 0
        assert chunks["b"][1] == 10
