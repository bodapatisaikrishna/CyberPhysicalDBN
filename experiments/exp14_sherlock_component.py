"""exp14: per-component view + label-free multivariate detectors on 02-Semiurban's
full 12 h physical snapshots (LAB_NOTEBOOK.md 2026-10-03, plan step 1).

Why: exp13's physical variant (11 grid-wide aggregates) gave lift 1.06-1.24x, ROC-AUC
~0.46. This tests whether keeping every component (5,722 columns + explicit NaN counts)
exposes the attacks the aggregates average away. Hypotheses/criteria were written to the
notebook BEFORE this script ran. Detectors are fit on attack-free blocks only; thresholds
come from held-out clean calib blocks; nothing is searched on test labels. Results are
REPORTED, never gated (CLAUDE.md rule 3); the structural gate only checks plumbing.

Run: .venv/bin/python experiments/exp14_sherlock_component.py [--smoke] [--config PATH]
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
from src.perception.sherlock_anomaly import blocked_split
from src.perception.sherlock_component_detectors import causal_rolling_robust_z, component_scores, fit_component_model


def _load_exp13():
    """Path-import exp13 (same pattern as exp13's own exp07 import) to reuse its loader
    and metric helpers rather than copy them. Registered in sys.modules before exec."""
    path = REPO_ROOT / "experiments" / "exp13_sherlock_full.py"
    spec = importlib.util.spec_from_file_location("exp13_sherlock_full", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp13_sherlock_full"] = module
    spec.loader.exec_module(module)
    return module


RESULTS_DIR = REPO_ROOT / "results"
SUMMARIES_DIR = RESULTS_DIR / "summaries"
PRIMARY = "pca_spe"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "sherlock_component.yaml"))
    args = parser.parse_args()

    e13 = _load_exp13()
    base_cfg = yaml.safe_load((REPO_ROOT / "configs" / "base.yaml").read_text())
    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = int(base_cfg["seed"])
    e13.set_all_seeds(seed)
    RESULTS_DIR.mkdir(exist_ok=True)
    SUMMARIES_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = git_sha(REPO_ROOT)
    c = cfg["component"]
    print(f"{cfg.get('experiment', 'exp14')} seed={seed} git_sha={sha} smoke={args.smoke}", flush=True)
    failures: list[str] = []

    print("\nstage 0: locating and parsing (component view) ...", flush=True)
    files = e13.locate_and_load(cfg, args.smoke)
    clean = next((f for f in files if not f.has_attacks), None)
    atk = next((f for f in files if f.has_attacks), None)
    if clean is None or atk is None:
        print("GATE FAILED: need one clean train file and one attack file")
        return 1
    names = clean.feats.column_names
    n_cols = clean.feats.features.shape[1]
    EXP = cfg.get("experiment", "exp14")
    tag = f"{'smoke_' if args.smoke else ''}"
    clean_x, atk_x = clean.feats.features, atk.feats.features
    rb = c.get("rolling_baseline")
    if rb:
        # exp15: score each snapshot against the file's own trailing baseline (see
        # causal_rolling_robust_z). The scale floor is fit on CLEAN TRAIN ONLY.
        print(f"  rolling baseline: window {rb['window']}, min_history {rb['min_history']} (causal, label-free)", flush=True)
        w, mh, zc = int(rb["window"]), int(rb["min_history"]), float(c["z_clip"])
        _, floor = causal_rolling_robust_z(clean_x, window=w, min_history=mh, scale_floor=None, z_clip=zc)
        clean_x, _ = causal_rolling_robust_z(clean_x, window=w, min_history=mh, scale_floor=floor, z_clip=zc)
        atk_x, _ = causal_rolling_robust_z(atk_x, window=w, min_history=mh, scale_floor=floor, z_clip=zc)
        nf_col_idx = list(names).index("nonfinite_total")
        print(f"  baseline-normalised features ready: train {tuple(clean_x.shape)}, test {tuple(atk_x.shape)}", flush=True)
    print(f"  clean {clean.label}: {tuple(clean.feats.features.shape)}; attack {atk.label}: "
          f"{tuple(atk.feats.features.shape)}, base rate {atk.base_rate:.4f}, {len(atk.runs)} attack events", flush=True)
    cad_med, cad_bad = clean.cadence_median, clean.cadence_n_bad
    print(f"  cadence: train median {cad_med:.3f}s ({cad_bad} gaps out of tol), "
          f"test median {atk.cadence_median:.3f}s ({atk.cadence_n_bad} gaps out of tol)", flush=True)

    sp = cfg["splits"]
    parts = blocked_split(clean.feats.features.shape[0], int(sp["block_slices"]), tuple(sp["pattern"]))
    fit_r, calib_r = parts["fit"], parts["calib"]
    fit_x = torch.cat([clean_x[a:b] for a, b in fit_r])
    print(f"  fit {fit_x.shape[0]} slices, calib {sum(b - a for a, b in calib_r)} slices (interleaved blocks)", flush=True)

    model = fit_component_model(
        fit_x, eps=float(c["standardizer_eps"]), z_clip=float(c["z_clip"]),
        explained_variance_target=float(c["explained_variance_target"]), max_components=int(c["max_components"]),
        fit_split=f"02-Semiurban/train fit blocks ({fit_x.shape[0]} slices)",
    )
    print(f"  PCA: k={model.components.shape[1]} components capture {model.explained_variance_ratio:.4f} of fit variance; "
          f"{model.n_constant_in_fit}/{model.n_columns} columns constant in the fit chunk", flush=True)

    batch, top_k = int(c["score_batch_size"]), int(c["top_k"])
    clean_scores = component_scores(model, clean_x, top_k, batch)
    atk_scores = component_scores(model, atk_x, top_k, batch)
    nf_col = list(names).index("nonfinite_total")
    clean_scores["nonfinite_total"] = clean.feats.features[:, nf_col].numpy().astype(np.float32)
    atk_scores["nonfinite_total"] = atk.feats.features[:, nf_col].numpy().astype(np.float32)

    pct = float(c["alarm_percentile"])
    thetas = {d: float(np.percentile(np.concatenate([s[a:b] for a, b in calib_r]), pct)) for d, s in clean_scores.items()}

    y = atk.feats.labels.numpy()
    rec = e13.recovery_mask(atk)
    metric_rows, raw = [], {}
    for d, s in atk_scores.items():
        if not np.isfinite(s).all():
            failures.append(f"non-finite scores from {d}")
        ap, roc = e13.ap_and_roc(y, s)
        alarm = s >= thetas[d]
        tp = int((alarm & (y > 0)).sum())
        ex = e13.excl_metrics(y, s, rec, thetas[d])
        metric_rows.append({
            "scenario": "02-Semiurban", "detector": d, "view": "component", "n": len(y), "base_rate": float(y.mean()),
            "auc_pr": ap, "lift": ap / float(y.mean()), "roc_auc": roc, "theta_raw": thetas[d],
            "recall_at_alarm": tp / max(int(y.sum()), 1), "precision_at_alarm": tp / int(alarm.sum()) if alarm.any() else float("nan"),
            "fpr_attack_file_nonattack": float(alarm[y == 0].mean()), **ex,
            "pca_k": int(model.components.shape[1]), "pca_explained_variance": model.explained_variance_ratio,
            "n_columns": model.n_columns, "n_constant_in_fit": model.n_constant_in_fit,
            "git_sha": sha, "seed": seed,
        })
        print(f"  {d:<17} AUC-PR={ap:.4f} (base {y.mean():.4f}, lift {ap / y.mean():.2f}x) ROC-AUC={roc:.4f} | "
              f"excl. recovery: AUC-PR={ex['auc_pr_excl_recovery']:.4f} ROC-AUC={ex['roc_auc_excl_recovery']:.4f} | "
              f"FPR non-attack {float(alarm[y == 0].mean()):.3f}; calib-clean FPR "
              f"{float((np.concatenate([clean_scores[d][a:b] for a, b in calib_r]) >= thetas[d]).mean()):.3f}", flush=True)
        raw[f"{d}__y"] = y.astype(np.int8)
        raw[f"{d}__score"] = s.astype(np.float32)

    event_rows = e13.event_table("02-Semiurban", atk, atk_scores, thetas)
    ev = pd.DataFrame(event_rows)
    type_rows = []
    for (d, at), g in ev.groupby(["detector", "attack_type"]):
        type_rows.append({
            "scenario": "02-Semiurban", "detector": d, "attack_type": at, "n_events": len(g),
            "n_detected": int(g["detected"].sum()), "mean_frac_slices_alarmed": float(g["frac_slices_alarmed"].mean()),
            "mean_event_auroc": float(g["event_auroc"].mean()), "n_events_auroc_ge_0.9": int((g["event_auroc"] >= 0.9).sum()),
            "git_sha": sha,
        })
    print("\nattack-type breakdown (threshold-free event AUROC; 0.5 = indistinguishable from normal):")
    for r in type_rows:
        print(f"  {r['detector']:<17} {r['attack_type']:<36} alarmed {r['n_detected']}/{r['n_events']}; "
              f"mean event AUROC {r['mean_event_auroc']:.2f}, {r['n_events_auroc_ge_0.9']}/{r['n_events']} >= 0.9")

    # --- pre-registered criteria (notebook 2026-10-03, exp14) -- REPORTED, not gated ---
    prim = next(r for r in metric_rows if r["detector"] == PRIMARY)
    ind = ev[(ev.detector == PRIMARY) & ev.attack_type.str.startswith("industroyer")]
    fam = {f: float(ev[(ev.detector == PRIMARY) & ev.attack_type.str.startswith(f)]["event_auroc"].mean())
           for f in ("arp-spoof", "control-and-freeze", "drift-off")}
    ind_nf = ev[(ev.detector == "nonfinite_total") & ev.attack_type.str.startswith("industroyer")]
    n_ind = int((ind.event_auroc >= 0.9).sum())
    if EXP == "exp14":
        crit = {
            "H-C1 pca_spe ROC-AUC>=0.70 and lift>=2.0": (prim["roc_auc"] >= 0.70 and prim["lift"] >= 2.0, f"ROC-AUC {prim['roc_auc']:.3f}, lift {prim['lift']:.2f}x"),
            "H-C2 pca_spe industroyer >=7/9 events AUROC>=0.9": (n_ind >= 7, f"{n_ind}/{len(ind)}"),
            "H-C3 pca_spe mean event AUROC>=0.60 for each other family": (all(v >= 0.60 for v in fam.values()), json.dumps({k: round(v, 3) for k, v in fam.items()})),
            "H-C4 nonfinite_total industroyer >=5/9 events AUROC>=0.9": (int((ind_nf.event_auroc >= 0.9).sum()) >= 5, f"{int((ind_nf.event_auroc >= 0.9).sum())}/{len(ind_nf)}"),
        }
    else:  # exp15 criteria, pre-registered in the notebook (2026-10-03, exp15)
        crit = {
            "E-1 pca_spe ROC-AUC>=0.70 and lift>=2.0": (prim["roc_auc"] >= 0.70 and prim["lift"] >= 2.0, f"ROC-AUC {prim['roc_auc']:.3f}, lift {prim['lift']:.2f}x"),
            "E-2 pca_spe non-attack FPR<=0.05 at calib threshold": (prim["fpr_attack_file_nonattack"] <= 0.05, f"{prim['fpr_attack_file_nonattack']:.3f}"),
            "E-3 pca_spe industroyer >=7/9 events AUROC>=0.9": (n_ind >= 7, f"{n_ind}/{len(ind)}"),
            "E-4 pca_spe mean event AUROC>=0.60 for each other family": (all(v >= 0.60 for v in fam.values()), json.dumps({k: round(v, 3) for k, v in fam.items()})),
        }
    print("\npre-registered criteria (REPORTED, not gated):")
    for k, (ok, detail) in crit.items():
        print(f"  {'MET   ' if ok else 'MISSED'} {k}: {detail}")

    out = lambda name: RESULTS_DIR / f"{EXP}_{tag}{name}_{timestamp}.csv"
    pd.DataFrame(metric_rows).to_csv(out("anomaly_metrics"), index=False)
    ev.to_csv(out("events"), index=False)
    pd.DataFrame(type_rows).to_csv(out("attack_type_summary"), index=False)
    pd.DataFrame([{"criterion": k, "met": bool(ok), "detail": d, "git_sha": sha, "seed": seed}
                  for k, (ok, d) in crit.items()]).to_csv(out("criteria"), index=False)
    npz = RESULTS_DIR / f"{EXP}_{tag}raw_scores_{timestamp}.npz"
    np.savez_compressed(npz, **raw)
    sdf = pd.DataFrame([{"arm": "A_in_domain_component", "train": "02-Semiurban", "eval": "02-Semiurban", "detector": r["detector"],
                         "auc_pr": r["auc_pr"], "base_rate": r["base_rate"], "lift": r["lift"], "roc_auc": r["roc_auc"],
                         "git_sha": sha, "seed": seed} for r in metric_rows])
    sdf.to_csv(SUMMARIES_DIR / f"{EXP}_summary_{timestamp}.csv", index=False)
    print(f"\nwrote {EXP}_{tag}*_{timestamp}.csv, {npz.name}, summaries/{EXP}_summary_{timestamp}.csv")

    # --- structural gate ---
    print("\n=== VALIDATION GATE (structural only) ===")
    chk = {
        "fit/calib data has zero attack slices": float(clean.feats.labels.sum()) == 0.0,
        "all scores finite": not any("non-finite" in f for f in failures),
        "every catalogued attack labelled (skipped under --smoke: a prefix holds few)": args.smoke or len(atk.runs) == e13.catalog_coverage(atk)["n_catalog_attack_events"],
        "event runs partition positive slices": sum(b - a for _, a, b in atk.runs) == int(atk.feats.labels.sum().item()),
        "seed/git sha logged": bool(sha) and seed is not None,
    }
    for k, ok in chk.items():
        print(f"  {k} ... {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(k)
    print(f"  (cadence, reported as in exp13 gate b: train {cad_bad} / test {atk.cadence_n_bad} gaps outside tolerance)")
    if failures:
        print("\nGATE FAILED:\n  - " + "\n  - ".join(failures))
        return 1
    print("\nGATE PASSED (structural). Metrics above are reported, not gated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
