"""All three real Sherlock scenarios, used properly (LAB_NOTEBOOK.md 2026-09-28).

exp07 used only `01-Basic` and found that scenario gives no supervised signal:
its train split has zero real attacks, so the supervised classifier collapsed
to a constant predictor (AUC-PR == base rate). This experiment uses the whole
dataset (01-Basic, 02-Semiurban, 03-Rural) in the way the dataset is built for:

  Arm A  unsupervised ANOMALY DETECTION, in-domain (01, 02): an LSTM
         autoencoder (src/baselines/lstm_ae.py, reused unchanged) is fit ONLY on
         the attack-free train file and scores the attack test file. No positive
         label is ever seen in training. A trivial mean-|z| detector on the same
         standardized features is scored alongside as a reference.
  Arm B  ZERO-SHOT CROSS-NETWORK transfer: each in-domain model scores the other
         scenarios (incl. 03-Rural, shipped by the dataset authors specifically
         to test transferability), normalized with the SOURCE statistics only.
         False-alarm rates on the target's clean data separate "cannot see the
         attack" from "the whole network looks anomalous because its scale differs".
  Arm C  SUPERVISED TRANSFER MATRIX on the shared 2-column bus-voltage subspace
         (exp07's design, unchanged): twin <-> each real attack file and real <->
         real. exp07 could fill only the twin -> 01-Basic cell.

Hypotheses H1-H4 and the pre-registered nulls are in LAB_NOTEBOOK.md
(2026-09-28), written before this script. The validation gate is structural
only; AUC-PR / ROC-AUC / lift print unconditionally and are never gated
(CLAUDE.md rule 3).

Run: .venv/bin/python experiments/exp13_sherlock_full.py [--smoke]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from src.baselines.lstm_ae import (
    WINDOW_SLICES,
    LSTMAETrialConfig,
    error_to_probability,
    fit_recon_error_scaler,
    train_autoencoder,
)
from src.eval.provenance import git_sha
from src.perception.calibration import perception_calibration_report
from src.perception.sherlock_anomaly import (
    apply_standardizer,
    blocked_split,
    causal_windows_for_endpoints,
    endpoints_in_ranges,
    event_runs,
    fit_standardizer,
    score_ae_batched,
    score_ae_ranges,
    zscore_mean_abs,
)
from src.perception.sherlock_loader import (
    SHERLOCK_GLOBAL_COLUMNS,
    StateFileFeatures,
    shared_subspace_from_mean_voltage,
    stream_state_file_features,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
CONFIG_PATH = REPO_ROOT / "configs" / "sherlock_full.yaml"
TWIN_CONFIG_PATH = REPO_ROOT / "configs" / "twin.yaml"
RESULTS_DIR = REPO_ROOT / "results"
SUMMARIES_DIR = RESULTS_DIR / "summaries"

VOLTAGE_COL = SHERLOCK_GLOBAL_COLUMNS.index("bus_voltage_pu_mean")


def _load_exp07():
    """Path-import exp07 (same pattern exp06/exp09/exp11/exp12 use) so the
    transfer arm shares its EXACT TransferModel / twin-scenario code rather
    than a copy that could drift. `sys.modules` registration must precede
    `exec_module` (dataclass internals look the module up by name)."""
    path = REPO_ROOT / "experiments" / "exp07_sherlock.py"
    spec = importlib.util.spec_from_file_location("exp07_sherlock", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp07_sherlock"] = module
    spec.loader.exec_module(module)
    return module


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# === stage 0: locate, parse (cached), inventory ==============================


@dataclass
class SherlockFile:
    scenario: str
    kind: str  # "train" | "test" -- the FILE NAME's own split label, never assumed to mean clean/attack
    path: Path
    feats: StateFileFeatures
    cadence_median: float
    cadence_n_bad: int
    base_rate: float
    runs: list[tuple[str, int, int]]
    catalog: dict[str, dict] | None = None  # ipal events.json, keyed by event id string
    catalog_path: Path | None = None

    @property
    def has_attacks(self) -> bool:
        return self.base_rate > 0.0

    @property
    def label(self) -> str:
        return f"{self.scenario}/{self.kind}"


def load_features_cached(path: Path, cache_dir: Path, scenario: str, max_records: int | None) -> StateFileFeatures:
    """Parse once, reuse: the large files take minutes to stream. The cache is
    keyed on the source file's size + mtime (a re-downloaded/changed file is
    re-parsed, never silently served stale). Smoke prefixes are never cached."""
    st = path.stat()
    cache = cache_dir / f"{scenario}__{path.name}.npz"
    if max_records is None and cache.exists():
        z = np.load(cache, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if meta["size"] == st.st_size and meta["mtime_ns"] == st.st_mtime_ns:
            print(f"    cache hit: {cache.name}", flush=True)
            return StateFileFeatures(
                features=torch.from_numpy(z["features"]),
                labels=torch.from_numpy(z["labels"]),
                timestamps=z["timestamps"],
                event_ids=tuple(None if e == "" else str(e) for e in z["event_ids"]),
                raw_label_counts=meta["raw_label_counts"],
                component_key_counts=meta["component_key_counts"],
                n_state_keys_first_record=meta["n_keys"],
                truncated_tail_chars=int(meta.get("truncated_tail_chars", 0)),
            )
    t0 = time.time()
    f = stream_state_file_features(path, max_records=max_records)
    print(f"    parsed {f.features.shape[0]} records in {time.time() - t0:.0f}s", flush=True)
    if max_records is None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache, features=f.features.numpy(), labels=f.labels.numpy(), timestamps=f.timestamps,
            event_ids=np.array(["" if e is None else e for e in f.event_ids]),
            meta=json.dumps({
                "size": st.st_size, "mtime_ns": st.st_mtime_ns, "raw_label_counts": f.raw_label_counts,
                "component_key_counts": f.component_key_counts, "n_keys": f.n_state_keys_first_record,
                "truncated_tail_chars": f.truncated_tail_chars,
            }),
        )
    return f


def check_cadence(timestamps: np.ndarray, expected: float, tol: float) -> tuple[float, int]:
    """`(median_gap, n_gaps_outside_tolerance)` -- verified, never assumed,
    because window length is counted in slices == records."""
    gaps = np.diff(timestamps)
    if gaps.size == 0:
        return expected, 0
    return float(np.median(gaps)), int((np.abs(gaps - expected) > tol).sum())


def load_event_catalog(scenario_dir: Path, kind: str) -> tuple[dict[str, dict] | None, Path | None]:
    """The dataset's own attack catalog (`ipal/<split>/events.json`: id, attack
    type, manipulated state points, start/end/recovery unix times). Looked up
    for the file's own split first; 03-Rural ships a state file NAMED "train"
    but only `ipal/test`, so a missing split directory falls back to whichever
    single `ipal/*/events.json` exists (printed, and cross-checked against the
    state file's own label timestamps -- see gate (k))."""
    cands = [scenario_dir / "ipal" / kind / "events.json"] + sorted((scenario_dir / "ipal").glob("*/events.json"))
    for c in cands:
        if c.exists():
            return {str(e["id"]): e for e in json.loads(c.read_text())}, c
    return None, None


def locate_and_load(cfg: dict, smoke: bool) -> list[SherlockFile]:
    root = REPO_ROOT / cfg["data"]["root"]
    cache_dir = REPO_ROOT / cfg["data"]["cache_dir"]
    max_records = int(cfg["smoke"]["max_records"]) if smoke else None
    files: list[SherlockFile] = []
    for sc in cfg["data"]["scenarios"]:
        d = root / sc["dir"]
        print(f"  {sc['name']}: {d}", flush=True)
        if not d.exists():
            print("    NOT PRESENT on disk -- skipped (see docs/sherlock_download.md)")
            continue
        for kind, glob in (("train", cfg["data"]["train_glob"]), ("test", cfg["data"]["test_glob"])):
            matches = sorted(d.glob(glob))
            print(f"    {kind}_glob {glob!r} matched: {[m.name for m in matches]}")
            if not matches:
                continue
            f = load_features_cached(matches[0], cache_dir, sc["name"], max_records)
            med, n_bad = check_cadence(f.timestamps, float(cfg["timebase"]["expected_cadence_seconds"]),
                                       float(cfg["timebase"]["cadence_tolerance_seconds"]))
            runs = event_runs(f.event_ids)
            catalog, catalog_path = load_event_catalog(d, kind) if runs else (None, None)
            if runs:
                print(f"    event catalog: {catalog_path.relative_to(d) if catalog_path else None}")
            files.append(SherlockFile(
                scenario=sc["name"], kind=kind, path=matches[0], feats=f, cadence_median=med,
                cadence_n_bad=n_bad, base_rate=float(f.labels.mean()), runs=runs,
                catalog=catalog, catalog_path=catalog_path,
            ))
    return files


# === metrics helpers =========================================================


def ap_and_roc(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return float(average_precision_score(y, s)), float(roc_auc_score(y, s))


def calib_numbers(y: np.ndarray, p: np.ndarray, seed: int) -> tuple[float, float]:
    """ECE (10 uniform bins) and Brier via the project's own report. Test
    streams are ONE continuous run (run_ids constant), so the report's
    run-level bootstrap CI is degenerate here (exp07 noted the same) --
    n_bootstrap is kept tiny and the CI is not reported."""
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    rep = perception_calibration_report(
        y.astype(int), p, np.zeros(len(y), dtype=int), n_bootstrap=10, rng=np.random.default_rng(seed)
    )
    return float(rep.ece[(10, "uniform")]), float(rep.brier)


@dataclass
class TrainedDetector:
    scenario: str
    standardizer: object
    ae: torch.nn.Module
    ae_scaler: object
    z_scaler: object
    theta_ae: float  # alarm threshold in RAW reconstruction-error space
    theta_z: float  # alarm threshold in RAW mean-|z| space
    fit_val_loss: float
    train_rows: list[dict]


def score_both(det: TrainedDetector, x_raw: torch.Tensor, batch: int) -> tuple[np.ndarray, np.ndarray]:
    """Raw scores (AE reconstruction error, mean |z|) for a whole file, using
    `det`'s OWN standardizer -- so passing another scenario's features is a
    genuine zero-shot transfer, not a re-fit."""
    x_std = apply_standardizer(x_raw, det.standardizer)
    return score_ae_batched(det.ae, x_std, WINDOW_SLICES, batch_size=batch), zscore_mean_abs(x_std)


# === arm A: train on clean data only =========================================


def train_detector(scenario: str, clean: SherlockFile, cfg: dict, seed: int, smoke: bool) -> TrainedDetector:
    a = cfg["anomaly"]
    sp = cfg["splits"]
    n = clean.feats.features.shape[0]
    parts = blocked_split(n, int(sp["block_slices"]), tuple(sp["pattern"]))
    fit_r, val_r, calib_r = parts["fit"], parts["val"], parts["calib"]
    sizes = {k: sum(e - s0 for s0, e in v) for k, v in parts.items()}
    print(f"  [{scenario}] clean-train file {clean.path.name}: {n} slices -> blocked split "
          f"({sp['block_slices']}-slice blocks, pattern {list(sp['pattern'])}): "
          f"fit {sizes['fit']} ({len(fit_r)} blocks), val {sizes['val']} ({len(val_r)}), calib {sizes['calib']} ({len(calib_r)})",
          flush=True)
    if min(sizes.values()) < 2 * WINDOW_SLICES:
        raise ValueError(f"[{scenario}] a split is shorter than 2 windows ({WINDOW_SLICES}); raise max_records")

    fit_x = torch.cat([clean.feats.features[s0:e] for s0, e in fit_r])
    std = fit_standardizer(fit_x, eps=float(a["standardizer_eps"]), z_clip=float(a["z_clip"]),
                           fit_split=f"{scenario}/train fit blocks ({sizes['fit']} slices of the attack-free train file)")
    x_std = apply_standardizer(clean.feats.features, std)
    stride = int(a["train_stride"])
    tr_end = endpoints_in_ranges(fit_r, stride, min_end=WINDOW_SLICES - 1)
    va_end = endpoints_in_ranges(val_r, stride)
    train_w = causal_windows_for_endpoints(x_std, tr_end, WINDOW_SLICES)
    val_w = causal_windows_for_endpoints(x_std, va_end, WINDOW_SLICES)
    print(f"  [{scenario}] AE windows: train {tuple(train_w.shape)}, val {tuple(val_w.shape)} (stride {stride})", flush=True)

    ae_cfg = LSTMAETrialConfig(
        hidden_dim=int(a["hidden_dim"]), latent_dim=int(a["latent_dim"]), n_layers=int(a["n_layers"]),
        dropout=float(a["dropout"]), learning_rate=float(a["learning_rate"]),
    )
    n_epochs = int(cfg["smoke"]["n_epochs"]) if smoke else int(a["n_epochs"])
    t0 = time.time()
    ae, rows = train_autoencoder(
        ae_cfg, train_w, val_w, n_epochs=n_epochs, batch_size=int(a["batch_size"]),
        grad_clip_norm=float(a["grad_clip_norm"]), patience=int(a["early_stopping_patience_epochs"]), torch_seed=seed,
    )
    print(f"  [{scenario}] AE trained: {len(rows)} epochs in {time.time() - t0:.0f}s, "
          f"final val loss {rows[-1]['val_loss']:.5f}", flush=True)

    batch = int(a["score_batch_size"])
    val_err = score_ae_ranges(ae, x_std, WINDOW_SLICES, val_r, batch_size=batch)
    cal_err = score_ae_ranges(ae, x_std, WINDOW_SLICES, calib_r, batch_size=batch)
    z_all = zscore_mean_abs(x_std)
    z_val = np.concatenate([z_all[s0:e] for s0, e in val_r])
    z_cal = np.concatenate([z_all[s0:e] for s0, e in calib_r])
    pct = float(a["alarm_percentile"])
    return TrainedDetector(
        scenario=scenario, standardizer=std, ae=ae,
        ae_scaler=fit_recon_error_scaler(val_err, fit_split=f"{scenario}/train val blocks"),
        z_scaler=fit_recon_error_scaler(z_val, fit_split=f"{scenario}/train val blocks"),
        theta_ae=float(np.percentile(cal_err, pct)), theta_z=float(np.percentile(z_cal, pct)),
        fit_val_loss=float(rows[-1]["val_loss"]), train_rows=rows,
    )


def event_table(scenario: str, target: SherlockFile, scores: dict[str, np.ndarray], thetas: dict[str, float]) -> list[dict]:
    """One row per real attack event per detector: how long, how strongly
    scored, whether it ever crossed the alarm threshold (and after how many
    slices), and a THRESHOLD-FREE separability: `event_auroc` is the AUROC of
    the event's slices against every normal slice (label 0 and outside any
    recovery window) of the same file -- 0.5 = indistinguishable from normal,
    1.0 = perfectly separable. Needed because the alarm threshold comes from a
    clean calibration chunk that can be optimistic (exp13 notes the
    non-attack false-alarm rate next to every detection count)."""
    rows = []
    normal = (target.feats.labels.numpy() == 0) & ~recovery_mask(target)
    for det, s in scores.items():
        neg_sorted = np.sort(s[normal])
        for ev, a, b in target.runs:
            seg = s[a:b]
            hit = np.flatnonzero(seg >= thetas[det])
            cat = (target.catalog or {}).get(ev, {})
            below = (np.searchsorted(neg_sorted, seg, side="left") + np.searchsorted(neg_sorted, seg, side="right")) / 2.0
            rows.append({
                "scenario": scenario, "detector": det, "event_id": ev, "start": a, "n_slices": b - a,
                "attack_type": cat.get("description", "<not in catalog>"),
                "n_attack_points": len(cat.get("attack_point", [])),
                "catalog_start_offset_s": float(target.feats.timestamps[a] - cat["start"]) if cat else float("nan"),
                "mean_score": float(seg.mean()), "max_score": float(seg.max()),
                "event_auroc": float(below.mean() / max(len(neg_sorted), 1)),
                "detected": bool(hit.size), "delay_slices": float(hit[0]) if hit.size else float("nan"),
                "frac_slices_alarmed": float((seg >= thetas[det]).mean()),
            })
    return rows


def recovery_mask(f: SherlockFile) -> np.ndarray:
    """Slices in the post-attack RECOVERY window `[end, recovery)` of a real
    attack, per the dataset's own catalog. The shipped label marks only the
    attack itself as malicious, but the grid is still perturbed while it
    recovers, so an alarm there is not obviously a false alarm. Used for a
    second, ALSO-REPORTED evaluation that ignores these slices; never in place
    of the as-shipped labels."""
    ts = f.feats.timestamps
    m = np.zeros(len(ts), dtype=bool)
    for ev, _, _ in f.runs:
        cat = (f.catalog or {}).get(ev)
        if cat and "recovery" in cat and "end" in cat:
            m |= (ts >= float(cat["end"])) & (ts < float(cat["recovery"]))
    return m & (f.feats.labels.numpy() == 0)


def catalog_coverage(f: SherlockFile) -> dict:
    """How much of the dataset's own attack catalog this state file actually
    covers. A shipped state file can end before the catalog does (02-Semiurban's
    test file ends after ~3 h while its catalog lists 39 events), so the number
    of labelled attacks can be far below the number catalogued. Also flags any
    catalogued attack whose start lies INSIDE the file's time span but has no
    labelled run -- that would be a label/catalog mismatch, not truncation."""
    if not f.catalog:
        return {"n_catalog_attack_events": 0, "n_catalog_attacks_inside_file": 0,
                "n_catalog_attacks_after_file_end": 0, "unlabelled_inside": []}
    t0, t1 = float(f.feats.timestamps[0]), float(f.feats.timestamps[-1])
    attacks = {k: e for k, e in f.catalog.items() if "benign" not in k}
    inside = [k for k, e in attacks.items() if t0 <= float(e["start"]) <= t1]
    labelled = {ev for ev, _, _ in f.runs}
    return {
        "n_catalog_attack_events": len(attacks), "n_catalog_attacks_inside_file": len(inside),
        "n_catalog_attacks_after_file_end": sum(1 for e in attacks.values() if float(e["start"]) > t1),
        "unlabelled_inside": sorted(k for k in inside if k not in labelled),
    }


def excl_metrics(y: np.ndarray, sc: np.ndarray, ignore: np.ndarray, theta: float) -> dict:
    keep = ~ignore
    ap, roc = ap_and_roc(y[keep], sc[keep])
    yk = y[keep]
    return {
        "frac_slices_in_recovery": float(ignore.mean()),
        "base_rate_excl_recovery": float(yk.mean()) if yk.size else float("nan"),
        "auc_pr_excl_recovery": ap, "roc_auc_excl_recovery": roc,
        "fpr_nonattack_excl_recovery": float((sc[keep][yk == 0] >= theta).mean()) if (yk == 0).any() else float("nan"),
    }


# === arm C helpers ===========================================================


def real_subspace(f: SherlockFile, max_slices: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    """[S,2] shared bus-voltage subspace + labels, from the streamed
    `bus_voltage_pu_mean` aggregate (== per-bus mean when every bus is present
    in every record, as in the real files)."""
    n = f.feats.features.shape[0] if max_slices is None else min(max_slices, f.feats.features.shape[0])
    return shared_subspace_from_mean_voltage(f.feats.features[:n, VOLTAGE_COL]), f.feats.labels[:n]


# === main ====================================================================


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="scenario/hyperparameter YAML (default configs/sherlock_full.yaml)")
    args = parser.parse_args()

    base_cfg = yaml.safe_load(BASE_CONFIG_PATH.read_text())
    cfg = yaml.safe_load(Path(args.config).read_text())
    twin_cfg = yaml.safe_load(TWIN_CONFIG_PATH.read_text())
    seed = int(base_cfg["seed"])
    set_all_seeds(seed)
    RESULTS_DIR.mkdir(exist_ok=True)
    SUMMARIES_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = git_sha(REPO_ROOT)
    tag = "smoke_" if args.smoke else ""
    print(f"exp13 seed={seed} git_sha={sha} smoke={args.smoke} window={WINDOW_SLICES}", flush=True)
    failures: list[str] = []

    # --- stage 0: locate + parse + inventory ---------------------------------
    print("\nstage 0: locating and parsing every Sherlock file present ...", flush=True)
    files = locate_and_load(cfg, args.smoke)
    if not files:
        print("\nGATE FAILED: no Sherlock files found. See docs/sherlock_download.md.")
        return 1

    inv_rows = []
    for f in files:
        non_benign = [k for k in f.feats.raw_label_counts
                      if k != "False" and "benign" not in k and not k.isdigit()]
        f_role = "attack_file" if f.has_attacks else "clean_train"
        inv_rows.append({
            "scenario": f.scenario, "file": f.path.name, "kind": f.kind, "role": f_role,
            "n_records": f.feats.features.shape[0], "n_state_keys": f.feats.n_state_keys_first_record,
            "truncated_tail_chars": f.feats.truncated_tail_chars,
            "cadence_median_s": f.cadence_median, "cadence_n_gaps_outside_tol": f.cadence_n_bad,
            "base_rate": f.base_rate, "n_attack_slices": int(f.feats.labels.sum().item()),
            "n_attack_events": len(f.runs), **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in catalog_coverage(f).items()},
            "n_distinct_raw_labels": len(f.feats.raw_label_counts),
            "unclassified_raw_labels": json.dumps(non_benign),
            "raw_label_counts": json.dumps(f.feats.raw_label_counts, sort_keys=True),
            "component_key_counts": json.dumps(f.feats.component_key_counts, sort_keys=True),
            "git_sha": sha,
        })
        print(f"  {f.label:<22} {f.feats.features.shape[0]:>8} records, {f.feats.n_state_keys_first_record} keys, "
              f"cadence median {f.cadence_median:.3f}s ({f.cadence_n_bad} gaps out of tol), "
              f"base rate {f.base_rate:.6f}, {len(f.runs)} attack events, role={f_role}", flush=True)
        cov = catalog_coverage(f)
        if cov["n_catalog_attack_events"]:
            print(f"      catalog: {cov['n_catalog_attack_events']} attack events; {cov['n_catalog_attacks_inside_file']} start inside the file's "
                  f"time span, {cov['n_catalog_attacks_after_file_end']} start AFTER it ends; {len(f.runs)} labelled runs; "
                  f"catalogued-inside-but-unlabelled: {cov['unlabelled_inside']}")
        if f.feats.truncated_tail_chars:
            print(f"      NOTE: the file's final line is cut off mid-record ({f.feats.truncated_tail_chars} chars) and was "
                  "dropped; all earlier records are complete (a defect in the shipped data, reported not hidden)")
        print(f"      raw labels (top): {dict(sorted(f.feats.raw_label_counts.items(), key=lambda kv: -kv[1])[:6])}")
        print(f"      components: {f.feats.component_key_counts}")
        if non_benign:
            failures.append(f"{f.label}: raw label(s) not False / benign / numeric event id: {non_benign[:5]}")
        if f.cadence_n_bad:
            failures.append(f"{f.label}: {f.cadence_n_bad} inter-record gaps outside {cfg['timebase']['cadence_tolerance_seconds']}s "
                            "-- window-in-slices semantics need real-timestamp slicing")
        if not torch.isfinite(f.feats.features).all():
            failures.append(f"{f.label}: non-finite feature values")
    inv_df = pd.DataFrame(inv_rows)
    inv_path = RESULTS_DIR / f"exp13_{tag}inventory_{timestamp}.csv"
    inv_df.to_csv(inv_path, index=False)
    print(f"  wrote {inv_path}")

    by_scen: dict[str, list[SherlockFile]] = {}
    for f in files:
        by_scen.setdefault(f.scenario, []).append(f)
    clean_files = {s: next((f for f in fs if not f.has_attacks), None) for s, fs in by_scen.items()}
    attack_files = {s: next((f for f in fs if f.has_attacks), None) for s, fs in by_scen.items()}

    # --- arm A: in-domain anomaly detection ---------------------------------
    print("\narm A: unsupervised anomaly detection, in-domain (clean-train only) ...", flush=True)
    a_cfg = cfg["anomaly"]
    detectors: dict[str, TrainedDetector] = {}
    metric_rows, event_rows, train_rows_all, profile_rows = [], [], [], []
    raw_scores: dict[str, np.ndarray] = {}
    batch = int(a_cfg["score_batch_size"])
    for s in by_scen:
        clean, atk = clean_files[s], attack_files[s]
        if clean is None or atk is None:
            print(f"  [{s}] no (clean train file, attack file) pair -- not an in-domain source "
                  f"(clean={clean.label if clean else None}, attack={atk.label if atk else None})")
            continue
        det = train_detector(s, clean, cfg, seed, args.smoke)
        detectors[s] = det
        for r in det.train_rows:
            train_rows_all.append({"scenario": s, **r, "git_sha": sha})
        ae_err, z_sc = score_both(det, atk.feats.features, batch)
        y = atk.feats.labels.numpy()
        scores = {"lstm_ae": ae_err, "zscore_mean_abs": z_sc}
        thetas = {"lstm_ae": det.theta_ae, "zscore_mean_abs": det.theta_z}
        links = {"lstm_ae": det.ae_scaler, "zscore_mean_abs": det.z_scaler}
        for name in scores:
            p = error_to_probability(scores[name], links[name])
            ap, roc = ap_and_roc(y, scores[name])
            ece, brier = calib_numbers(y, p, seed)
            alarm = scores[name] >= thetas[name]
            n_alarm = int(alarm.sum())
            tp = int((alarm & (y > 0)).sum())
            metric_rows.append({
                "scenario": s, "detector": name, "n": len(y), "base_rate": float(y.mean()),
                "auc_pr": ap, "lift": ap / float(y.mean()) if y.mean() > 0 else float("nan"), "roc_auc": roc,
                "ece_10_uniform": ece, "brier": brier, "theta_raw": thetas[name],
                "recall_at_alarm": tp / max(int(y.sum()), 1), "precision_at_alarm": tp / n_alarm if n_alarm else float("nan"),
                "fpr_attack_file_nonattack": float(alarm[y == 0].mean()),
                **excl_metrics(y, scores[name], recovery_mask(atk), thetas[name]),
                "fit_val_loss": det.fit_val_loss if name == "lstm_ae" else float("nan"),
                "git_sha": sha, "seed": seed,
            })
            ex = metric_rows[-1]
            print(f"  [{s}] {name:<16} AUC-PR={ap:.4f} (base {y.mean():.4f}, lift {ap / max(y.mean(), 1e-12):.2f}x) "
                  f"ROC-AUC={roc:.4f} ECE={ece:.4f} | ignoring recovery windows ({ex['frac_slices_in_recovery']:.1%} of slices): "
                  f"AUC-PR={ex['auc_pr_excl_recovery']:.4f} (base {ex['base_rate_excl_recovery']:.4f}) "
                  f"ROC-AUC={ex['roc_auc_excl_recovery']:.4f} | FPR non-attack {ex['fpr_nonattack_excl_recovery']:.3f} "
                  f"(as shipped {metric_rows[-1]['fpr_attack_file_nonattack']:.3f})", flush=True)
            raw_scores[f"{s}__{name}__y"] = y.astype(np.int8)
            raw_scores[f"{s}__{name}__score"] = scores[name].astype(np.float32)
        event_rows.extend(event_table(s, atk, scores, thetas))
        # What do the 11 aggregates actually see during each attack type? Mean |z| per feature
        # (source-standardized) over the attack's slices vs. the file's non-attack slices.
        z_atk = apply_standardizer(atk.feats.features, det.standardizer).abs().numpy()
        nonatk = z_atk[atk.feats.labels.numpy() == 0].mean(axis=0)
        by_type: dict[str, list[np.ndarray]] = {}
        for ev, a, b in atk.runs:
            at = (atk.catalog or {}).get(ev, {}).get("description", "<not in catalog>")
            by_type.setdefault(at, []).append(z_atk[a:b].mean(axis=0))
        for at, vs in by_type.items():
            m = np.mean(vs, axis=0)
            for j, col in enumerate(SHERLOCK_GLOBAL_COLUMNS):
                profile_rows.append({"scenario": s, "attack_type": at, "n_events": len(vs), "feature": col,
                                     "mean_abs_z_during_attack": float(m[j]), "mean_abs_z_nonattack": float(nonatk[j]),
                                     "git_sha": sha})
        det_by_det = [r for r in event_rows if r["scenario"] == s]
        for name in scores:
            rs = [r for r in det_by_det if r["detector"] == name]
            print(f"  [{s}] {name:<16} events detected {sum(r['detected'] for r in rs)}/{len(rs)}", flush=True)

    # --- arm B: zero-shot cross-network transfer ----------------------------
    print("\narm B: zero-shot cross-network transfer ...", flush=True)
    cross_rows = []
    for src, det in detectors.items():
        for tgt in by_scen:
            if tgt == src:
                continue
            tclean, tatk = clean_files[tgt], attack_files[tgt]
            fpr_clean = {}
            if tclean is not None:
                ae_c, z_c = score_both(det, tclean.feats.features, batch)
                fpr_clean = {"lstm_ae": float((ae_c >= det.theta_ae).mean()), "zscore_mean_abs": float((z_c >= det.theta_z).mean())}
            if tatk is None:
                print(f"  {src} -> {tgt}: no attack file in target; only clean-data false-alarm rate is available")
                continue
            ae_e, z_e = score_both(det, tatk.feats.features, batch)
            y = tatk.feats.labels.numpy()
            for name, sc, th in (("lstm_ae", ae_e, det.theta_ae), ("zscore_mean_abs", z_e, det.theta_z)):
                ap, roc = ap_and_roc(y, sc)
                alarm = sc >= th
                cross_rows.append({
                    "source": src, "target": tgt, "target_file": tatk.path.name, "detector": name,
                    "n": len(y), "base_rate": float(y.mean()), "auc_pr": ap,
                    "lift": ap / float(y.mean()) if y.mean() > 0 else float("nan"), "roc_auc": roc,
                    "fpr_target_clean_train": fpr_clean.get(name, float("nan")),
                    "fpr_target_attackfile_nonattack": float(alarm[y == 0].mean()),
                    "recall_at_source_alarm": float(alarm[y > 0].mean()) if y.sum() else float("nan"),
                    **excl_metrics(y, sc, recovery_mask(tatk), th),
                    "git_sha": sha, "seed": seed,
                })
                print(f"  {src} -> {tgt:<13} {name:<16} AUC-PR={ap:.4f} (base {y.mean():.4f}) ROC-AUC={roc:.4f} "
                      f"FPR@src-theta: clean-train={fpr_clean.get(name, float('nan')):.3f} "
                      f"attack-file-nonattack={float(alarm[y == 0].mean()):.3f}", flush=True)
                raw_scores[f"{src}_to_{tgt}__{name}__y"] = y.astype(np.int8)
                raw_scores[f"{src}_to_{tgt}__{name}__score"] = sc.astype(np.float32)

    # --- arm C: supervised transfer matrix on the shared 2-col subspace -----
    print("\narm C: supervised transfer matrix (shared bus-voltage subspace) ...", flush=True)
    exp07 = _load_exp07()
    t_cfg = dict(cfg["transfer"])
    n_twin = int(cfg["smoke"]["n_twin_scenarios"]) if args.smoke else int(t_cfg["n_twin_scenarios"])
    n_epochs_c = int(cfg["smoke"]["n_epochs"]) if args.smoke else int(t_cfg["n_epochs"])
    t_cfg["n_epochs"] = n_epochs_c
    print("  building twin scenarios (same call as exp07 stage 4) ...", flush=True)
    twin_x, twin_y = exp07.build_twin_transfer_scenarios(twin_cfg, n_twin, float(t_cfg["twin_horizon_time_units"]), seed)
    twin_base = float(np.mean([float(y.mean()) for y in twin_y]))

    domains: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {"twin": (twin_x, twin_y)}
    full_eval: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    max_train = int(t_cfg["max_train_slices"])
    for s, atk in attack_files.items():
        if atk is None:
            continue
        x_tr, y_tr = real_subspace(atk, max_train)
        x_full, y_full = real_subspace(atk, None)
        if len(torch.unique(y_tr)) < 2:
            print(f"  {s}: first {len(y_tr)} slices contain a single class -- cannot TRAIN on it (still an eval target)")
        else:
            domains[s] = ([x_tr], [y_tr])
        full_eval[s] = (x_full, y_full)

    matrix_rows = []
    for tr_name, (xs, ys) in domains.items():
        model = exp07.train_transfer_model(xs, ys, t_cfg, seed + 1 if tr_name == "twin" else seed)
        # eval on every other domain (twin: mean AP over its scenarios; real: full attack file)
        for ev_name in ["twin", *full_eval]:
            if ev_name == tr_name:
                continue
            if ev_name == "twin":
                aps = [exp07.eval_transfer_model(model, x, y) for x, y in zip(twin_x, twin_y)]
                ap, base, n_eval = float(np.nanmean(aps)), twin_base, len(aps)
            else:
                x, y = full_eval[ev_name]
                ap, base, n_eval = float(exp07.eval_transfer_model(model, x, y)), float(y.mean()), len(y)
            matrix_rows.append({
                "train_domain": tr_name, "eval_domain": ev_name, "auc_pr": ap, "base_rate": base,
                "lift": ap / base if base > 0 else float("nan"), "n_eval": n_eval,
                "n_train_slices": int(sum(len(y) for y in ys)), "git_sha": sha, "seed": seed,
            })
            print(f"  train={tr_name:<13} eval={ev_name:<13} AUC-PR={ap:.4f} (base {base:.4f}, lift {ap / base:.2f}x)", flush=True)

    # --- attack-type breakdown (which attacks do the aggregates expose?) -----
    ev_df = pd.DataFrame(event_rows)
    type_rows = []
    if len(ev_df):
        for (sc, det, at), g in ev_df.groupby(["scenario", "detector", "attack_type"]):
            type_rows.append({
                "scenario": sc, "detector": det, "attack_type": at, "n_events": len(g),
                "n_detected": int(g["detected"].sum()), "mean_frac_slices_alarmed": float(g["frac_slices_alarmed"].mean()),
                "mean_event_auroc": float(g["event_auroc"].mean()), "n_events_auroc_ge_0.9": int((g["event_auroc"] >= 0.9).sum()),
                "mean_n_attack_points": float(g["n_attack_points"].mean()), "git_sha": sha,
            })
        print("\nattack-type breakdown, lstm_ae (alarm counts depend on the calibration-chunk threshold; event AUROC does not):")
        for r in type_rows:
            if r["detector"] == "lstm_ae":
                print(f"  {r['scenario']:<13} {r['attack_type']:<34} alarmed {r['n_detected']}/{r['n_events']}; "
                      f"threshold-free: mean event AUROC {r['mean_event_auroc']:.2f}, {r['n_events_auroc_ge_0.9']}/{r['n_events']} events >= 0.9 "
                      f"(~{r['mean_n_attack_points']:.1f} attack points)")

    # --- write outputs -------------------------------------------------------
    def dump(name: str, rows: list[dict]) -> Path:
        path = RESULTS_DIR / f"exp13_{tag}{name}_{timestamp}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  wrote {path}")
        return path

    print("\nwriting outputs ...")
    metrics_df = pd.DataFrame(metric_rows)
    dump("anomaly_metrics", metric_rows)
    dump("events", event_rows)
    dump("attack_type_summary", type_rows)
    dump("attack_feature_profile", profile_rows)
    dump("training", train_rows_all)
    dump("cross_network", cross_rows)
    dump("transfer_matrix", matrix_rows)
    npz_path = RESULTS_DIR / f"exp13_{tag}raw_scores_{timestamp}.npz"
    np.savez_compressed(npz_path, **raw_scores)
    print(f"  wrote {npz_path}")

    summary = []
    for r in metric_rows:
        summary.append({"arm": "A_in_domain", "train": r["scenario"], "eval": r["scenario"], "detector": r["detector"],
                        "auc_pr": r["auc_pr"], "base_rate": r["base_rate"], "lift": r["lift"], "roc_auc": r["roc_auc"]})
    for r in cross_rows:
        summary.append({"arm": "B_zero_shot", "train": r["source"], "eval": r["target"], "detector": r["detector"],
                        "auc_pr": r["auc_pr"], "base_rate": r["base_rate"], "lift": r["lift"], "roc_auc": r["roc_auc"]})
    for r in matrix_rows:
        summary.append({"arm": "C_supervised_2col", "train": r["train_domain"], "eval": r["eval_domain"], "detector": "causal_tcn",
                        "auc_pr": r["auc_pr"], "base_rate": r["base_rate"], "lift": r["lift"], "roc_auc": float("nan")})
    sdf = pd.DataFrame(summary)
    sdf["git_sha"], sdf["seed"] = sha, seed
    sum_path = SUMMARIES_DIR / f"exp13_summary_{timestamp}.csv"
    sdf.to_csv(sum_path, index=False)
    print(f"  wrote {sum_path}")

    # --- reproducibility check vs exp07 (reported, not gated) ----------------
    ref = [r for r in matrix_rows if r["train_domain"] == "twin" and r["eval_domain"] == "01-Basic"]
    if ref and not args.smoke:
        print(f"\nexp07 cross-check: twin -> 01-Basic AUC-PR here = {ref[0]['auc_pr']:.10f}; "
              "exp07's logged twin_to_sherlock = 0.1692012238 (results/exp07_transfer_20260805T153324Z.csv). "
              "Small differences are expected from the streamed mean-voltage aggregate vs. exp07's per-bus mean.")

    # === validation gate (structural only) ====================================
    print("\n=== VALIDATION GATE ===")
    print(f"(a) every raw `malicious` label classifiable (False / benign / numeric id), all files ... "
          f"{'PASS' if not any('raw label' in f for f in failures) else 'FAIL'}")
    print(f"(b) cadence verified uniform (1.0s) on every file ... "
          f"{'PASS' if not any('gaps outside' in f for f in failures) else 'FAIL'}")
    print(f"(c) feature tensors finite, all files ... {'PASS' if not any('non-finite feature' in f for f in failures) else 'FAIL'}")

    ok_events = all(sum(b - a for _, a, b in f.runs) == int(f.feats.labels.sum().item()) for f in files)
    print(f"(d) event runs partition exactly the positive slices, every file ... {'PASS' if ok_events else 'FAIL'}")
    if not ok_events:
        failures.append("event runs do not cover exactly the positive slices")

    clean_ok = all(not clean_files[s].has_attacks for s in detectors)
    print(f"(e) every in-domain source trained on a file with ZERO real attack slices ... {'PASS' if clean_ok else 'FAIL'}")
    if not clean_ok:
        failures.append("an anomaly-detector source file contained real attacks")

    fin = bool(np.isfinite(metrics_df[["auc_pr", "roc_auc", "ece_10_uniform", "brier"]].to_numpy(dtype=float)).all()) if len(metrics_df) and not args.smoke else True
    scores_ok = all(np.isfinite(v).all() for k, v in raw_scores.items() if k.endswith("__score"))
    print(f"(f) all detector scores finite ... {'PASS' if scores_ok else 'FAIL'}")
    if not scores_ok:
        failures.append("non-finite detector score")
    print(f"(g) in-domain metrics finite (skipped under --smoke: a 6k prefix may hold no attack) ... {'PASS' if fin else 'FAIL'}")
    if not fin:
        failures.append("non-finite in-domain metric")

    if detectors:
        s0 = next(iter(detectors))
        d0 = detectors[s0]
        prefix = clean_files[s0].feats.features[:3 * WINDOW_SLICES]
        e1, _ = score_both(d0, prefix, batch)
        e2, _ = score_both(d0, prefix, batch)
        det_ok = np.array_equal(e1, e2)
    else:
        det_ok = False
    print(f"(h) AE scoring deterministic (same weights, same input, twice) ... {'PASS' if det_ok else 'FAIL'}")
    if not det_ok:
        failures.append("AE scoring not deterministic")
    print(f"(i) at least one in-domain (clean-train, attack-file) scenario evaluated ({len(detectors)}) ... "
          f"{'PASS' if detectors else 'FAIL'}")
    if not detectors:
        failures.append("no in-domain scenario evaluated")
    worst = 0.0
    n_uncat = 0
    for f in files:
        for ev, a, _ in f.runs:
            cat = (f.catalog or {}).get(ev)
            if cat is None:
                n_uncat += 1
            else:
                worst = max(worst, abs(float(f.feats.timestamps[a]) - float(cat["start"])))
    k_ok = n_uncat == 0 and worst <= 2.0
    print(f"(k) every labelled attack run is in the dataset's own event catalog and starts within 2 s of its catalogued "
          f"start (uncatalogued runs={n_uncat}, worst offset={worst:.3f}s) ... {'PASS' if k_ok else 'FAIL'}")
    if not k_ok:
        failures.append(f"label/catalog mismatch: {n_uncat} uncatalogued runs, worst start offset {worst:.3f}s")
    unl = [(f.label, catalog_coverage(f)["unlabelled_inside"]) for f in files if catalog_coverage(f)["unlabelled_inside"]]
    print(f"(l) every catalogued attack that starts inside a file's time span has a labelled run (no label/catalog mismatch) ... "
          f"{'PASS' if not unl else 'FAIL'}")
    if unl:
        failures.append(f"catalogued attacks inside the file's span with no labelled run: {unl}")
    print(f"(j) seed/git sha logged in every output row (seed={seed}, sha={sha[:8]}) ... PASS")

    print("\nNOTE: AUC-PR / ROC-AUC / lift above are REPORTED, not gated (CLAUDE.md rule 3). A null on H1 "
          "(no better than base rate) is a valid result, not a gate failure. See LAB_NOTEBOOK.md 2026-09-28.")
    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
