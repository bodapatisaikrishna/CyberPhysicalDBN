"""exp17: network-capture view + uniform-grid cadence sensitivity on 02-Semiurban's full
12 h (LAB_NOTEBOOK.md 2026-10-05; design and criteria N-0..N-5 pre-registered there).

Views on a common 2 s grid per split: (P) physical component snapshots carried onto the
grid by last-observation-carried-forward; (N) IEC-104 traffic counts from the six switch
pcaps; (P+N) fused. All detectors label-free, fit on attack-free train blocks only,
thresholds from held-out clean calib blocks, per-column causal rolling baseline (exp15).
Results are REPORTED, never gated (CLAUDE.md rule 3).

Needs the caches written by scripts/build_sherlock_network_features.py.
Run: .venv/bin/python experiments/exp17_sherlock_network.py [--smoke] [--config PATH]
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
from src.perception.sherlock_grid import grid_end_times, locf_index, max_locf_staleness, n_before_first
from src.perception.sherlock_loader import StateFileFeatures
from src.perception.sherlock_physical import attack_label_from_catalog

RESULTS_DIR = REPO_ROOT / "results"
SUMMARIES_DIR = RESULTS_DIR / "summaries"
CACHE = REPO_ROOT / "data/sherlock/_feature_cache"


def _load_exp13():
    path = REPO_ROOT / "experiments" / "exp13_sherlock_full.py"
    spec = importlib.util.spec_from_file_location("exp13_sherlock_full", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp13_sherlock_full"] = module
    spec.loader.exec_module(module)
    return module


def load_split(scenario: str, split: str, bin_s: float):
    net = np.load(CACHE / f"{scenario}__network__{split}.npz", allow_pickle=True)
    phys = np.load(CACHE / f"{scenario}__physical_component__{split}__physical.zip.npz", allow_pickle=True)
    counts = torch.from_numpy(net["counts"])
    t0 = float(net["t0"])
    assert float(net["bin_s"]) == bin_s
    pts = phys["timestamps"]
    t_end = grid_end_times(t0, counts.shape[0], bin_s)
    idx = locf_index(pts, t_end)
    xp = torch.from_numpy(phys["features"])[torch.from_numpy(idx)]
    catalog = json.loads((REPO_ROOT / "data/sherlock" / scenario / scenario / "ipal" / split / "events.json").read_text())
    y, ids = attack_label_from_catalog(t_end, catalog)
    info = {"n_before_first": n_before_first(pts, t_end), "max_staleness_s": max_locf_staleness(pts, t_end),
            "n_bins": int(counts.shape[0]), "names_net": [str(n) for n in net["names"]]}
    return xp, counts, y, ids, t_end, catalog, info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "sherlock_network.yaml"))
    args = parser.parse_args()
    e13 = _load_exp13()
    base_cfg = yaml.safe_load((REPO_ROOT / "configs" / "base.yaml").read_text())
    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = int(base_cfg["seed"])
    e13.set_all_seeds(seed)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = git_sha(REPO_ROOT)
    tag = "smoke_" if args.smoke else ""
    c, rb = cfg["component"], cfg["rolling_baseline"]
    bin_s = float(cfg["grid"]["bin_seconds"])
    print(f"{cfg.get('experiment', 'exp17')} seed={seed} git_sha={sha} smoke={args.smoke}", flush=True)
    failures: list[str] = []

    SC = cfg.get("scenario", "02-Semiurban")
    EXP = cfg.get("experiment", "exp17")
    n_base = int(cfg.get("n_network_features", 23))
    xp_tr, xn_tr_all, y_tr, _, _, _, info_tr = load_split(SC, "train", bin_s)
    xp_te, xn_te_all, y_te, ids_te, t_end_te, catalog, info_te = load_split(SC, "test", bin_s)
    xn_tr, xn_te = xn_tr_all[:, :n_base], xn_te_all[:, :n_base]  # exp17 = first 23 network features
    extended = bool(cfg.get("extended_network", False))
    print(f"  scenario {SC}; network features: {n_base} base" + (f" + {xn_te_all.shape[1] - n_base} extended (view N+)" if extended else ""), flush=True)
    if args.smoke:
        m = int(cfg["smoke"]["max_bins"])
        xp_tr, xn_tr, xn_tr_all, y_tr = xp_tr[:m], xn_tr[:m], xn_tr_all[:m], y_tr[:m]
        xp_te, xn_te, xn_te_all, y_te, ids_te, t_end_te = xp_te[:m], xn_te[:m], xn_te_all[:m], y_te[:m], ids_te[:m], t_end_te[:m]
    for nm, inf in (("train", info_tr), ("test", info_te)):
        print(f"  {nm}: {inf['n_bins']} uniform {bin_s:g}s bins; {inf['n_before_first']} grid points before the first snapshot; "
              f"max LOCF staleness {inf['max_staleness_s']:.2f}s", flush=True)
    names_net = info_te["names_net"]
    print(f"  attack-free train bins carry {int(y_tr.sum())} attack labels; test base rate {float(y_te.mean()):.4f}", flush=True)

    w, mh, zc = int(rb["window"]), int(rb["min_history"]), float(c["z_clip"])

    def normalise(x_tr, x_te):
        _, floor = causal_rolling_robust_z(x_tr, window=w, min_history=mh, scale_floor=None, z_clip=zc)
        z_tr, _ = causal_rolling_robust_z(x_tr, window=w, min_history=mh, scale_floor=floor, z_clip=zc)
        z_te, _ = causal_rolling_robust_z(x_te, window=w, min_history=mh, scale_floor=floor, z_clip=zc)
        return z_tr, z_te

    sp = cfg["splits"]
    parts = blocked_split(xp_tr.shape[0], int(sp["block_slices"]), tuple(sp["pattern"]))
    fit_r, calib_r = parts["fit"], parts["calib"]

    def fit_view(x_tr, x_te, max_comp, label):
        z_tr, z_te = normalise(x_tr, x_te)
        fit = torch.cat([z_tr[a:b] for a, b in fit_r])
        mdl = fit_component_model(fit, eps=float(c["standardizer_eps"]), z_clip=zc,
                                  explained_variance_target=float(c["explained_variance_target"]),
                                  max_components=max_comp, fit_split=f"{SC}/train fit blocks, view {label}")
        print(f"  view {label}: k={mdl.components.shape[1]} ({mdl.explained_variance_ratio:.4f} of fit variance), "
              f"{mdl.n_constant_in_fit}/{mdl.n_columns} constant in fit", flush=True)
        k_top = int(c["top_k"]) if label == "P" else int(c["top_k_net"])
        return (component_scores(mdl, z_tr, k_top, int(c["score_batch_size"])),
                component_scores(mdl, z_te, k_top, int(c["score_batch_size"])))

    p_tr, p_te = fit_view(xp_tr, xp_te, int(c["max_components_physical"]), "P")
    n_tr, n_te = fit_view(xn_tr, xn_te, int(c["max_components_net"]), "N")
    if extended:
        np_tr, np_te = fit_view(xn_tr_all, xn_te_all, int(c["max_components_net"]) + 2, "N+")

    def calib(s):
        return float(np.percentile(np.concatenate([s[a:b] for a, b in calib_r]), float(c["alarm_percentile"])))

    cal = {"P_pca_spe": calib(p_tr["pca_spe"]), "N_pca_spe": calib(n_tr["pca_spe"]), "N_top_k_mean_abs_z": calib(n_tr["top_k_mean_abs_z"])}
    tr_scores = {"P_pca_spe": p_tr["pca_spe"], "N_pca_spe": n_tr["pca_spe"], "N_top_k_mean_abs_z": n_tr["top_k_mean_abs_z"],
                 "PN_max": np.maximum(p_tr["pca_spe"] / cal["P_pca_spe"], n_tr["pca_spe"] / cal["N_pca_spe"])}
    te_scores = {"P_pca_spe": p_te["pca_spe"], "N_pca_spe": n_te["pca_spe"], "N_top_k_mean_abs_z": n_te["top_k_mean_abs_z"],
                 "PN_max": np.maximum(p_te["pca_spe"] / cal["P_pca_spe"], n_te["pca_spe"] / cal["N_pca_spe"])}
    # exp18 F-a: fusion by MEAN of the calibration-normalised scores (added rows; exp17's rows unchanged)
    tr_scores["PN_mean"] = 0.5 * (p_tr["pca_spe"] / cal["P_pca_spe"] + n_tr["pca_spe"] / cal["N_pca_spe"])
    te_scores["PN_mean"] = 0.5 * (p_te["pca_spe"] / cal["P_pca_spe"] + n_te["pca_spe"] / cal["N_pca_spe"])
    if extended:
        cal["Nplus_pca_spe"] = calib(np_tr["pca_spe"])
        tr_scores["Nplus_pca_spe"], te_scores["Nplus_pca_spe"] = np_tr["pca_spe"], np_te["pca_spe"]
        tr_scores["PNplus_mean"] = 0.5 * (p_tr["pca_spe"] / cal["P_pca_spe"] + np_tr["pca_spe"] / cal["Nplus_pca_spe"])
        te_scores["PNplus_mean"] = 0.5 * (p_te["pca_spe"] / cal["P_pca_spe"] + np_te["pca_spe"] / cal["Nplus_pca_spe"])
    thetas = {d: calib(s) for d, s in tr_scores.items()}

    # a SherlockFile on the grid so exp13's event/metric helpers apply unchanged
    runs = event_runs(ids_te)
    feats = StateFileFeatures(features=xn_te_all, labels=y_te, timestamps=t_end_te, event_ids=ids_te, raw_label_counts={},
                              component_key_counts={}, n_state_keys_first_record=0)
    atk = e13.SherlockFile(scenario=SC, kind="test", path=Path("grid"), feats=feats, cadence_median=bin_s,
                           cadence_n_bad=0, base_rate=float(y_te.mean()), runs=runs,
                           catalog={str(e["id"]): e for e in catalog}, catalog_path=None)
    y = y_te.numpy()
    rec = e13.recovery_mask(atk)
    metric_rows, raw = [], {}
    for d, s in te_scores.items():
        if not np.isfinite(s).all():
            failures.append(f"non-finite scores from {d}")
        ap, roc = e13.ap_and_roc(y, s)
        alarm = s >= thetas[d]
        ex = e13.excl_metrics(y, s, rec, thetas[d])
        tp = int((alarm & (y > 0)).sum())
        metric_rows.append({"scenario": SC, "detector": d, "n": len(y), "base_rate": float(y.mean()), "auc_pr": ap,
                            "lift": ap / float(y.mean()), "roc_auc": roc, "theta_raw": thetas[d],
                            "recall_at_alarm": tp / max(int(y.sum()), 1),
                            "fpr_attack_file_nonattack": float(alarm[y == 0].mean()), **ex, "git_sha": sha, "seed": seed})
        print(f"  {d:<20} AUC-PR={ap:.4f} (base {y.mean():.4f}, lift {ap / y.mean():.2f}x) ROC-AUC={roc:.4f} | "
              f"FPR non-attack {float(alarm[y == 0].mean()):.3f}", flush=True)
        raw[f"{d}__y"] = y.astype(np.int8)
        raw[f"{d}__score"] = s.astype(np.float32)

    ev = pd.DataFrame(e13.event_table(SC, atk, te_scores, thetas))
    type_rows = []
    for (d, at), g in ev.groupby(["detector", "attack_type"]):
        type_rows.append({"scenario": SC, "detector": d, "attack_type": at, "n_events": len(g),
                          "n_detected": int(g["detected"].sum()), "mean_event_auroc": float(g["event_auroc"].mean()),
                          "n_events_auroc_ge_0.9": int((g["event_auroc"] >= 0.9).sum()), "git_sha": sha})
    print("\nattack-type breakdown (mean event AUROC; events >= 0.9):")
    for r in type_rows:
        print(f"  {r['detector']:<20} {r['attack_type']:<38} {r['mean_event_auroc']:.2f}  ({r['n_events_auroc_ge_0.9']}/{r['n_events']})")

    def fam_auroc(det, fam):
        g = ev[(ev.detector == det) & ev.attack_type.str.startswith(fam)]
        return float(g.event_auroc.mean()), int((g.event_auroc >= 0.9).sum()), len(g)

    p_ind = fam_auroc("P_pca_spe", "industroyer")
    pn_ind = fam_auroc("PN_max", "industroyer")
    prm = {r["detector"]: r for r in metric_rows}
    if EXP == "exp17":
        crit = {
            "N-0 cadence sensitivity: P pooled ROC within 0.05 of 0.501 and industroyer>=0.9 events within 1 of 6/9":
                (abs(prm["P_pca_spe"]["roc_auc"] - 0.501) <= 0.05 and abs(p_ind[1] - 6) <= 1, f"ROC {prm['P_pca_spe']['roc_auc']:.3f}, industroyer {p_ind[1]}/{p_ind[2]}"),
            "N-1 arp-spoof mean event AUROC>=0.70 (N pca_spe)": (fam_auroc("N_pca_spe", "arp-spoof")[0] >= 0.70, f"{fam_auroc('N_pca_spe', 'arp-spoof')[0]:.3f}"),
            "N-2 control-and-freeze mean event AUROC>=0.70 (N pca_spe)": (fam_auroc("N_pca_spe", "control-and-freeze")[0] >= 0.70, f"{fam_auroc('N_pca_spe', 'control-and-freeze')[0]:.3f}"),
            "N-3 drift-off mean event AUROC>=0.70 (N pca_spe)": (fam_auroc("N_pca_spe", "drift-off")[0] >= 0.70, f"{fam_auroc('N_pca_spe', 'drift-off')[0]:.3f}"),
            "N-4 fused: industroyer>=7/9 events AUROC>=0.9 and pooled ROC>=0.70 and lift>=2.0":
                (pn_ind[1] >= 7 and prm["PN_max"]["roc_auc"] >= 0.70 and prm["PN_max"]["lift"] >= 2.0,
                 f"industroyer {pn_ind[1]}/{pn_ind[2]}, ROC {prm['PN_max']['roc_auc']:.3f}, lift {prm['PN_max']['lift']:.2f}x"),
            "N-5 non-attack FPR<=0.05 for N and for P+N": (prm["N_pca_spe"]["fpr_attack_file_nonattack"] <= 0.05 and prm["PN_max"]["fpr_attack_file_nonattack"] <= 0.05,
                                                           f"N {prm['N_pca_spe']['fpr_attack_file_nonattack']:.3f}, PN {prm['PN_max']['fpr_attack_file_nonattack']:.3f}"),
        }
    else:  # exp18 criteria (LAB_NOTEBOOK.md 2026-10-09; R = frozen replication, F = fixes)
        pn_ind = fam_auroc("PN_max", "industroyer")
        crit = {
            "R-1 fused PN_max pooled ROC>=0.70 and lift>=2.0": (prm["PN_max"]["roc_auc"] >= 0.70 and prm["PN_max"]["lift"] >= 2.0, f"ROC {prm['PN_max']['roc_auc']:.3f}, lift {prm['PN_max']['lift']:.2f}x"),
            "R-2 industroyer PN_max >=5 events AUROC>=0.9": (pn_ind[1] >= 5, f"{pn_ind[1]}/{pn_ind[2]}"),
            "R-3 arp-spoof N_pca_spe mean event AUROC>=0.70": (fam_auroc("N_pca_spe", "arp-spoof")[0] >= 0.70, f"{fam_auroc('N_pca_spe', 'arp-spoof')[0]:.3f} over {fam_auroc('N_pca_spe', 'arp-spoof')[2]} events"),
            "F-1 PN_mean non-attack FPR<=0.05": (prm["PN_mean"]["fpr_attack_file_nonattack"] <= 0.05, f"PN_mean {prm['PN_mean']['fpr_attack_file_nonattack']:.3f} vs PN_max {prm['PN_max']['fpr_attack_file_nonattack']:.3f}"),
        }
        if extended:
            crit["F-2 control-and-freeze Nplus_pca_spe mean event AUROC>=0.70"] = (fam_auroc("Nplus_pca_spe", "control-and-freeze")[0] >= 0.70, f"{fam_auroc('Nplus_pca_spe', 'control-and-freeze')[0]:.3f} (N: {fam_auroc('N_pca_spe', 'control-and-freeze')[0]:.3f})")
            crit["F-3 PNplus_mean pooled ROC>=0.70 and lift>=2.0"] = (prm["PNplus_mean"]["roc_auc"] >= 0.70 and prm["PNplus_mean"]["lift"] >= 2.0, f"ROC {prm['PNplus_mean']['roc_auc']:.3f}, lift {prm['PNplus_mean']['lift']:.2f}x")
    print("\npre-registered criteria (REPORTED, not gated):")
    for k, (ok, d) in crit.items():
        print(f"  {'MET   ' if ok else 'MISSED'} {k}: {d}")

    def out(name):
        return RESULTS_DIR / f"{EXP}_{tag}{name}_{timestamp}.csv"
    pd.DataFrame(metric_rows).to_csv(out("anomaly_metrics"), index=False)
    ev.to_csv(out("events"), index=False)
    pd.DataFrame(type_rows).to_csv(out("attack_type_summary"), index=False)
    pd.DataFrame([{"criterion": k, "met": bool(o), "detail": d, "git_sha": sha, "seed": seed} for k, (o, d) in crit.items()]).to_csv(out("criteria"), index=False)
    pd.DataFrame([{**info_te, "split": "test", "names_net": ""}, {**info_tr, "split": "train", "names_net": ""}]).to_csv(out("grid_info"), index=False)
    npz = RESULTS_DIR / f"{EXP}_{tag}raw_scores_{timestamp}.npz"
    np.savez_compressed(npz, **raw)
    pd.DataFrame([{"arm": "A_in_domain_network_grid", "train": SC, "eval": SC, "detector": r["detector"],
                   "auc_pr": r["auc_pr"], "base_rate": r["base_rate"], "lift": r["lift"], "roc_auc": r["roc_auc"], "git_sha": sha, "seed": seed}
                  for r in metric_rows]).to_csv(SUMMARIES_DIR / f"{EXP}_summary_{timestamp}.csv", index=False)
    print(f"\nwrote {EXP}_{tag}*_{timestamp}.csv")

    print("\n=== VALIDATION GATE (structural only) ===")
    chk = {
        "train grid has zero attack labels": float(y_tr.sum()) == 0.0,
        "all scores finite": not any("non-finite" in f for f in failures),
        "grid uniform by construction (bin_s constant; staleness reported above)": True,
        "event runs partition positive slices": args.smoke or sum(b - a for _, a, b in runs) == int(y_te.sum().item()),
        "every catalogued attack labelled": args.smoke or len(runs) == e13.catalog_coverage(atk)["n_catalog_attack_events"],
        "network features finite": bool(torch.isfinite(xn_tr).all() and torch.isfinite(xn_te).all()),
    }
    for k, ok in chk.items():
        print(f"  {k} ... {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(k)
    if failures:
        print("\nGATE FAILED:\n  - " + "\n  - ".join(failures))
        return 1
    print("\nGATE PASSED (structural). Metrics above are reported, not gated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
