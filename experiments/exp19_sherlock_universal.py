"""exp19: scenario-agnostic detector with a self-calibrating causal threshold, evaluated
leave-one-scenario-out; 03-Rural is the untouched confirmatory test
(LAB_NOTEBOOK.md 2026-10-09, design + criteria U-1..U-4 pre-registered there).

Features per 2 s bin (26, all network-size-agnostic): the 23 exp17 IEC-104 traffic counts,
`n_iec104_commands` (ASDU types 45..69), `n_switch_changes` and `nonfinite_total` from the
LOCF physical grid. Per-run causal robust z with a self-estimated scale floor (needs no clean
data of the evaluated network). PCA fit on clean-train fit blocks of the OTHER scenarios.
Thresholds: static (fit scenarios' calib 99th pct) and adaptive (rolling median + c * MAD of
the run's own past scores; c set for 1% false alarms on the fit scenarios' clean calib blocks).
Results are REPORTED, never gated (CLAUDE.md rule 3).

Run: .venv/bin/python experiments/exp19_sherlock_universal.py [--config PATH]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml

from src.eval.provenance import git_sha
from src.perception.sherlock_anomaly import blocked_split, event_runs
from src.perception.sherlock_component_detectors import causal_rolling_robust_z, component_scores, fit_component_model
from src.perception.sherlock_loader import StateFileFeatures
from src.perception.sherlock_network import FEATURE_NAMES

RESULTS_DIR = REPO_ROOT / "results"
SUMMARIES_DIR = RESULTS_DIR / "summaries"


def _path_import(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "experiments" / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


NET_IDX = list(range(23)) + [FEATURE_NAMES.index("n_iec104_commands")]
FEATURES = tuple(FEATURE_NAMES[i] for i in NET_IDX) + ("n_switch_changes", "nonfinite_total")


def physical_events(xp: torch.Tensor, phys_names: list[str]) -> torch.Tensor:
    """[S,2]: switch `closed` states changed since the previous grid point; NaN-count total."""
    sw = [i for i, n in enumerate(phys_names) if n.startswith("switch.") and n.endswith(".CONFIGURATION.closed")]
    s = np.nan_to_num(xp[:, sw].numpy().astype(np.float64), nan=-1.0)
    ch = np.zeros(len(s))
    ch[1:] = (s[1:] != s[:-1]).sum(axis=1)
    nf = xp[:, phys_names.index("nonfinite_total")].numpy()
    return torch.from_numpy(np.stack([ch, nf], axis=1).astype(np.float32))


def adaptive_norm(s: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """(s_t - m_t) / (1.4826 * d_t): m = rolling median of PAST scores, d = rolling median of
    past |s - m| floored by its own expanding median (causal). NaN where history is short."""
    ser = pd.Series(s.astype(np.float64))
    m = ser.shift(1).rolling(window, min_periods=min_periods).median()
    dev = (ser - m).abs()
    d = dev.shift(1).rolling(window, min_periods=min_periods).median()
    dfl = d.expanding(min_periods=1).median()
    d = np.maximum(d.to_numpy(), np.maximum(np.nan_to_num(dfl.to_numpy(), nan=0.0), 1e-12))
    return ((ser - m).to_numpy()) / (1.4826 * d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "sherlock_universal.yaml"))
    args = ap.parse_args()
    e13 = _path_import("exp13_sherlock_full", "exp13_sherlock_full.py")
    e17 = _path_import("exp17_sherlock_network", "exp17_sherlock_network.py")
    base_cfg = yaml.safe_load((REPO_ROOT / "configs" / "base.yaml").read_text())
    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = int(base_cfg["seed"])
    e13.set_all_seeds(seed)
    ts_run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = git_sha(REPO_ROOT)
    bin_s = float(cfg["grid"]["bin_seconds"])
    rb, c, th = cfg["rolling_baseline"], cfg["component"], cfg["threshold"]
    print(f"exp19 seed={seed} git_sha={sha}; {len(FEATURES)} features: {FEATURES}", flush=True)

    def view(scenario: str, split: str):
        xp, counts, y, ids, t_end, catalog, info = e17.load_split(scenario, split, bin_s)
        phys = np.load(e17.CACHE / f"{scenario}__physical_component__{split}__physical.zip.npz", allow_pickle=True)
        names = list(json.loads(str(phys["meta"]))["column_names"])
        x = torch.cat([counts[:, NET_IDX], physical_events(xp, names)], dim=1)
        z, _ = causal_rolling_robust_z(x, window=int(rb["window"]), min_history=int(rb["min_history"]),
                                       scale_floor=None, z_clip=float(c["z_clip"]), self_floor=True)
        return z, y, ids, t_end, catalog, info

    clean = {sc: view(sc, "train") for sc in cfg["clean_scenarios"]}
    for sc, v in clean.items():
        print(f"  clean {sc}/train: {tuple(v[0].shape)}, attack labels {int(v[1].sum())}", flush=True)

    sp = cfg["splits"]
    metric_rows, ev_all, type_rows, crit_rows, raw = [], [], [], [], {}
    for target, fit_from, role in cfg["evaluations"]:
        fit_z, cal_s, cal_a = [], {}, {}
        parts = {sc: blocked_split(clean[sc][0].shape[0], int(sp["block_slices"]), tuple(sp["pattern"])) for sc in fit_from}
        for sc in fit_from:
            fit_z.append(torch.cat([clean[sc][0][a:b] for a, b in parts[sc]["fit"]]))
        mdl = fit_component_model(torch.cat(fit_z), eps=float(c["standardizer_eps"]), z_clip=float(c["z_clip"]),
                                  explained_variance_target=float(c["explained_variance_target"]),
                                  max_components=len(FEATURES), fit_split=f"clean fit blocks of {fit_from}")
        for sc in fit_from:
            sc_scores = component_scores(mdl, clean[sc][0], int(c["top_k"]), 4096)
            for d, s in sc_scores.items():
                if d not in ("pca_spe", "top_k_mean_abs_z"):
                    continue
                a_n = adaptive_norm(s, int(th["window"]), int(th["min_periods"]))
                cal_s.setdefault(d, []).append(np.concatenate([s[a:b] for a, b in parts[sc]["calib"]]))
                cal_a.setdefault(d, []).append(np.concatenate([a_n[a:b] for a, b in parts[sc]["calib"]]))
        thetas = {d: float(np.percentile(np.concatenate(v), float(th["percentile"]))) for d, v in cal_s.items()}
        for d, v in cal_a.items():
            vv = np.concatenate(v)
            thetas[f"{d}_adaptive"] = float(np.nanpercentile(vv, float(th["percentile"])))
        print(f"\n[{target}] ({role}) fit on {fit_from}: PCA k={mdl.components.shape[1]} ({mdl.explained_variance_ratio:.4f}); "
              f"thresholds {json.dumps({k: round(v, 3) for k, v in thetas.items()})}", flush=True)

        zt, y_t, ids_t, t_end, catalog, info = view(target, "test")
        sc_t = component_scores(mdl, zt, int(c["top_k"]), 4096)
        scores = {}
        for d in ("pca_spe", "top_k_mean_abs_z"):
            scores[d] = sc_t[d]
            a_n = adaptive_norm(sc_t[d], int(th["window"]), int(th["min_periods"]))
            # before enough history the adaptive score is undefined: fall back to the static decision
            fallback = (sc_t[d] - thetas[d]) / max(abs(thetas[d]), 1e-12) + thetas[f"{d}_adaptive"]
            scores[f"{d}_adaptive"] = np.where(np.isfinite(a_n), a_n, fallback)
        runs = event_runs(ids_t)
        feats = StateFileFeatures(features=zt, labels=y_t, timestamps=t_end, event_ids=ids_t, raw_label_counts={},
                                  component_key_counts={}, n_state_keys_first_record=0)
        atk = e13.SherlockFile(scenario=target, kind="test", path=Path("grid"), feats=feats, cadence_median=bin_s,
                               cadence_n_bad=0, base_rate=float(y_t.mean()), runs=runs,
                               catalog={str(e["id"]): e for e in catalog}, catalog_path=None)
        y = y_t.numpy()
        rec = e13.recovery_mask(atk)
        for d, s in scores.items():
            ap_, roc = e13.ap_and_roc(y, s)
            alarm = s >= thetas[d]
            metric_rows.append({"target": target, "role": role, "fit_from": "+".join(fit_from), "detector": d, "n": len(y),
                                "base_rate": float(y.mean()), "auc_pr": ap_, "lift": ap_ / float(y.mean()), "roc_auc": roc,
                                "theta": thetas[d], "fpr_nonattack": float(alarm[y == 0].mean()),
                                "recall_at_alarm": float(alarm[y > 0].mean()),
                                **e13.excl_metrics(y, s, rec, thetas[d]), "git_sha": sha, "seed": seed})
            print(f"  {d:<26} AUC-PR={ap_:.4f} (base {y.mean():.4f}, lift {ap_ / y.mean():.2f}x) ROC-AUC={roc:.4f} | "
                  f"non-attack FPR {float(alarm[y == 0].mean()):.3f}, recall {float(alarm[y > 0].mean()):.3f}", flush=True)
            raw[f"{target}__{d}__y"] = y.astype(np.int8)
            raw[f"{target}__{d}__score"] = s.astype(np.float32)
        ev = pd.DataFrame(e13.event_table(target, atk, scores, thetas))
        ev["role"] = role
        ev_all.append(ev)
        for (d, at), g in ev.groupby(["detector", "attack_type"]):
            type_rows.append({"target": target, "role": role, "detector": d, "attack_type": at, "n_events": len(g),
                              "n_detected": int(g["detected"].sum()), "mean_event_auroc": float(g["event_auroc"].mean()),
                              "n_events_auroc_ge_0.9": int((g["event_auroc"] >= 0.9).sum()), "git_sha": sha})
        tr = [r for r in type_rows if r["target"] == target and r["detector"] == "pca_spe"]
        for r in tr:
            print(f"    pca_spe {r['attack_type']:<38} {r['mean_event_auroc']:.2f} ({r['n_events_auroc_ge_0.9']}/{r['n_events']} >= 0.9), alarmed {r['n_detected']}/{r['n_events']}")

        prim = next(r for r in metric_rows if r["target"] == target and r["detector"] == "pca_spe")
        adap = next(r for r in metric_rows if r["target"] == target and r["detector"] == "pca_spe_adaptive")
        fam = {r["attack_type"].split(":")[0]: r for r in tr}
        ind, arp = fam.get("industroyer"), fam.get("arp-spoof")
        crit = {
            "U-1 pca_spe pooled ROC>=0.70 and lift>=2.0": (prim["roc_auc"] >= 0.70 and prim["lift"] >= 2.0, f"ROC {prim['roc_auc']:.3f}, lift {prim['lift']:.2f}x"),
            "U-2 adaptive-threshold non-attack FPR<=0.05": (adap["fpr_nonattack"] <= 0.05, f"{adap['fpr_nonattack']:.3f} (static {prim['fpr_nonattack']:.3f}); recall {adap['recall_at_alarm']:.3f}"),
            "U-3 industroyer >=5 events AUROC>=0.9 (03: of 8)": (ind is not None and ind["n_events_auroc_ge_0.9"] >= 5, f"{ind['n_events_auroc_ge_0.9']}/{ind['n_events']}" if ind else "n/a"),
            "U-4 arp-spoof mean event AUROC>=0.80": (arp is not None and arp["mean_event_auroc"] >= 0.80, f"{arp['mean_event_auroc']:.3f}" if arp else "n/a"),
        }
        print(f"  criteria ({role}):")
        for k, (ok, dtl) in crit.items():
            print(f"    {'MET   ' if ok else 'MISSED'} {k}: {dtl}")
            crit_rows.append({"target": target, "role": role, "criterion": k, "met": bool(ok), "detail": dtl, "git_sha": sha, "seed": seed})

    def out(name):
        return RESULTS_DIR / f"exp19_{name}_{ts_run}.csv"
    pd.DataFrame(metric_rows).to_csv(out("anomaly_metrics"), index=False)
    pd.concat(ev_all).to_csv(out("events"), index=False)
    pd.DataFrame(type_rows).to_csv(out("attack_type_summary"), index=False)
    pd.DataFrame(crit_rows).to_csv(out("criteria"), index=False)
    np.savez_compressed(RESULTS_DIR / f"exp19_raw_scores_{ts_run}.npz", **raw)
    pd.DataFrame([{"arm": f"loso_{r['role']}", "train": r["fit_from"], "eval": r["target"], "detector": r["detector"],
                   "auc_pr": r["auc_pr"], "base_rate": r["base_rate"], "lift": r["lift"], "roc_auc": r["roc_auc"],
                   "git_sha": sha, "seed": seed} for r in metric_rows]).to_csv(SUMMARIES_DIR / f"exp19_summary_{ts_run}.csv", index=False)
    print(f"\nwrote exp19_*_{ts_run}.csv")
    ok = all(np.isfinite(r["roc_auc"]) for r in metric_rows) and all(v[1].sum() == 0 for v in clean.values())
    print("\n=== VALIDATION GATE (structural only) ===")
    print(f"  clean fit data has zero attack labels; all metrics finite ... {'PASS' if ok else 'FAIL'}")
    print(f"  every evaluation's fit set excludes its target scenario ... "
          f"{'PASS' if all(t not in f for t, f, _ in cfg['evaluations']) else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
