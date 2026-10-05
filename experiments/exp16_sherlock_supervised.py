"""exp16: event-held-out SUPERVISED detection on 02-Semiurban's full physical snapshots
(LAB_NOTEBOOK.md 2026-10-03, plan step 3; design + criteria pre-registered there).

Question: does the physical state carry ANY learnable footprint for the attack families
the unsupervised detectors cannot see? Features = PCA scores (+ SPE + NaN counts) of exp15's causal, label-free
baseline-normalised component view, PCA fit on clean train only.
Model = HistGradientBoosting with FIXED hyperparameters (never tuned). Attack events are
dealt BY TYPE into K folds, the non-attack timeline in 30-min blocks into the same folds;
training excludes a 150-slice guard around every held-out slice. Trains on the attack
file's other folds only -- so results are NOT comparable with clean-train detectors and
do not transfer without labelled target data. Reported, never gated.

Run: .venv/bin/python experiments/exp16_sherlock_supervised.py [--smoke] [--config PATH]
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
from sklearn.ensemble import HistGradientBoostingClassifier

from src.eval.provenance import git_sha
from src.perception.sherlock_anomaly import blocked_split
from src.perception.sherlock_component_detectors import causal_rolling_robust_z, fit_component_model, standardized

RESULTS_DIR = REPO_ROOT / "results"
SUMMARIES_DIR = RESULTS_DIR / "summaries"


def _load_exp13():
    path = REPO_ROOT / "experiments" / "exp13_sherlock_full.py"
    spec = importlib.util.spec_from_file_location("exp13_sherlock_full", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp13_sherlock_full"] = module
    spec.loader.exec_module(module)
    return module


def assign_folds(runs, n_slices: int, attack_type_of: dict[str, str], k: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Per-slice fold id. Attack events: shuffled within each type, then dealt round-robin
    (so each fold holds every type that has >= k events; rarer types spread over fewer folds).
    Non-attack slices: consecutive `block`-slice blocks dealt cyclically from a random offset.
    Slices of an event take the event's fold, regardless of the time block they fall in."""
    fold = np.full(n_slices, -1, dtype=np.int64)
    by_type: dict[str, list] = {}
    for ev, a, b in runs:
        by_type.setdefault(attack_type_of[ev], []).append((a, b))
    offset = 0
    for _, evs in sorted(by_type.items()):
        order = rng.permutation(len(evs))
        for j, idx in enumerate(order):
            a, b = evs[idx]
            fold[a:b] = (offset + j) % k
        offset += len(evs)
    start = int(rng.integers(0, k))
    for bi, a in enumerate(range(0, n_slices, block)):
        seg = fold[a:a + block]
        seg[seg < 0] = (bi + start) % k
    return fold


def guard_mask(fold: np.ndarray, test_fold: int, guard: int) -> np.ndarray:
    """True for slices within `guard` of any slice in `test_fold` (excluded from training)."""
    near = np.zeros(len(fold), dtype=bool)
    idx = np.flatnonzero(fold == test_fold)
    if idx.size == 0:
        return near
    # diff-array dilation
    d = np.zeros(len(fold) + 1, dtype=np.int64)
    lo, hi = np.maximum(idx - guard, 0), np.minimum(idx + guard + 1, len(fold))
    np.add.at(d, lo, 1)
    np.add.at(d, hi, -1)
    return np.cumsum(d[:-1]) > 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "sherlock_supervised.yaml"))
    args = parser.parse_args()
    e13 = _load_exp13()
    base_cfg = yaml.safe_load((REPO_ROOT / "configs" / "base.yaml").read_text())
    cfg = yaml.safe_load(Path(args.config).read_text())
    seed0 = int(base_cfg["seed"])
    e13.set_all_seeds(seed0)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = git_sha(REPO_ROOT)
    tag = "smoke_" if args.smoke else ""
    c, s = cfg["component"], cfg["supervised"]
    print(f"exp16 seed={seed0} git_sha={sha} smoke={args.smoke}", flush=True)

    files = e13.locate_and_load(cfg, args.smoke)
    clean = next(f for f in files if not f.has_attacks)
    atk = next(f for f in files if f.has_attacks)
    names = clean.feats.column_names
    rb = c["rolling_baseline"]
    w, mh, zc = int(rb["window"]), int(rb["min_history"]), float(c["z_clip"])
    _, floor = causal_rolling_robust_z(clean.feats.features, window=w, min_history=mh, scale_floor=None, z_clip=zc)
    xa, _ = causal_rolling_robust_z(atk.feats.features, window=w, min_history=mh, scale_floor=floor, z_clip=zc)
    # Label-free dimensionality reduction (AMENDMENT, LAB_NOTEBOOK.md exp16): HGB on the full
    # ~3.9k columns did not finish a single fold in 25+ min. Features are instead the PCA scores
    # (up to `pca_components`) of the baseline-normalised view, fit on CLEAN-TRAIN fit blocks only,
    # plus the squared residual outside that subspace (so low-variance directions are not lost)
    # plus the NaN counts. No label touches the reduction.
    parts = blocked_split(clean.feats.features.shape[0], int(cfg["splits"]["block_slices"]), tuple(cfg["splits"]["pattern"]))
    xc, _ = causal_rolling_robust_z(clean.feats.features, window=w, min_history=mh, scale_floor=floor, z_clip=zc)
    fit = torch.cat([xc[a:b] for a, b in parts["fit"]])
    pm = fit_component_model(fit, eps=float(c["standardizer_eps"]), z_clip=zc,
                             explained_variance_target=float(s["pca_explained_variance"]),
                             max_components=int(s["pca_components"]), fit_split="02-Semiurban/train fit blocks, normalised")
    zt = standardized(pm, xa)
    proj = zt @ pm.components
    spe = ((zt - proj @ pm.components.T) ** 2).sum(dim=1, keepdim=True)
    nf_idx = [i for i, n in enumerate(names) if n.startswith("nonfinite_")]
    nf = atk.feats.features[:, nf_idx]  # raw NaN counts (columns not baseline-normalised)
    X = torch.cat([proj, spe, nf], dim=1).numpy().astype(np.float32)
    print(f"  PCA features: k={pm.components.shape[1]} ({pm.explained_variance_ratio:.4f} of fit variance) + SPE + {len(nf_idx)} NaN-count columns", flush=True)
    y = atk.feats.labels.numpy().astype(int)
    print(f"  features: {X.shape[1]} columns; {X.shape[0]} slices, base rate {y.mean():.4f}, {len(atk.runs)} events", flush=True)

    atype = {ev: (atk.catalog or {}).get(ev, {}).get("description", "<unknown>") for ev, _, _ in atk.runs}
    k, guard, block = int(s["n_folds"]), int(s["guard_slices"]), int(s["block_slices"])
    rec = e13.recovery_mask(atk)
    rows, ev_rows, raw = [], [], {}
    for rep in range(int(s["n_repeats"])):
        seed = seed0 + rep
        fold = assign_folds(atk.runs, len(y), atype, k, block, np.random.default_rng(seed))
        oof = np.full(len(y), np.nan)
        for f in range(k):
            test = fold == f
            train = (fold != f) & ~guard_mask(fold, f, guard)
            if len(np.unique(y[train])) < 2 or not test.any():
                continue
            clf = HistGradientBoostingClassifier(
                max_iter=int(s["max_iter"]), max_depth=int(s["max_depth"]), learning_rate=float(s["learning_rate"]),
                class_weight="balanced", random_state=seed,
            )
            clf.fit(X[train], y[train])
            oof[test] = clf.predict_proba(X[test])[:, 1]
            print(f"  rep {rep} fold {f}: train {int(train.sum())} ({int(y[train].sum())} attack), test {int(test.sum())}", flush=True)
        ok = np.isfinite(oof)
        ap, roc = e13.ap_and_roc(y[ok], oof[ok])
        ex = e13.excl_metrics(y[ok], oof[ok], rec[ok], 0.5)
        rows.append({"repeat": rep, "seed": seed, "n_scored": int(ok.sum()), "base_rate": float(y[ok].mean()),
                     "auc_pr": ap, "lift": ap / float(y[ok].mean()), "roc_auc": roc,
                     "auc_pr_excl_recovery": ex["auc_pr_excl_recovery"], "roc_auc_excl_recovery": ex["roc_auc_excl_recovery"],
                     "n_columns": int(X.shape[1]), "git_sha": sha})
        print(f"  rep {rep}: pooled out-of-fold AUC-PR={ap:.4f} (base {y[ok].mean():.4f}, lift {ap / y[ok].mean():.2f}x) ROC-AUC={roc:.4f}", flush=True)
        # per-event AUROC vs normal (label 0, not recovery) out-of-fold slices
        normal = ok & (y == 0) & ~rec
        neg = np.sort(oof[normal])
        for ev, a, b in atk.runs:
            seg = oof[a:b]
            if not np.isfinite(seg).all():
                continue
            lo = (np.searchsorted(neg, seg, side="left") + np.searchsorted(neg, seg, side="right")) / 2.0
            ev_rows.append({"repeat": rep, "event_id": ev, "attack_type": atype[ev], "n_slices": b - a,
                            "event_auroc": float(lo.mean() / max(len(neg), 1)), "git_sha": sha})
        raw[f"rep{rep}__y"] = y.astype(np.int8)
        raw[f"rep{rep}__score"] = oof.astype(np.float32)

    ev = pd.DataFrame(ev_rows)
    per_type = (ev.groupby(["attack_type", "repeat"]).event_auroc.mean().groupby("attack_type").agg(["mean", "std"]))
    n9 = ev[ev.attack_type.str.startswith("industroyer")].groupby("repeat").event_auroc.apply(lambda v: int((v >= 0.9).sum()))
    r = pd.DataFrame(rows)
    print("\nper attack type (mean event AUROC over events, then mean/sd over repeats):")
    print(per_type.to_string())
    print(f"\npooled out-of-fold over {len(r)} repeats: ROC-AUC {r.roc_auc.mean():.4f} +/- {r.roc_auc.std(ddof=0):.4f}, "
          f"AUC-PR {r.auc_pr.mean():.4f} +/- {r.auc_pr.std(ddof=0):.4f}, lift {r.lift.mean():.2f}x")
    fam = {t: float(per_type.loc[t, "mean"]) for t in per_type.index if not t.startswith("industroyer")}
    crit = {
        "S-1 pooled out-of-fold ROC-AUC>=0.80": (r.roc_auc.mean() >= 0.80, f"{r.roc_auc.mean():.3f}"),
        "S-2 each non-industroyer family mean event AUROC>=0.70": (all(v >= 0.70 for v in fam.values()), json.dumps({t.split(':')[0]: round(v, 3) for t, v in fam.items()})),
        "S-3 industroyer >=7/9 events AUROC>=0.9 (mean over repeats)": (float(n9.mean()) >= 7, f"{n9.mean():.1f}/9"),
    }
    print("\npre-registered criteria (REPORTED, not gated):")
    for kk, (ok_, d) in crit.items():
        print(f"  {'MET   ' if ok_ else 'MISSED'} {kk}: {d}")

    r.to_csv(RESULTS_DIR / f"exp16_{tag}pooled_{timestamp}.csv", index=False)
    ev.to_csv(RESULTS_DIR / f"exp16_{tag}events_{timestamp}.csv", index=False)
    pd.DataFrame([{"criterion": kk, "met": bool(o), "detail": d, "git_sha": sha, "seed": seed0} for kk, (o, d) in crit.items()]
                 ).to_csv(RESULTS_DIR / f"exp16_{tag}criteria_{timestamp}.csv", index=False)
    pd.DataFrame([{"arm": "D_supervised_event_heldout", "train": "02-Semiurban/test other folds", "eval": "02-Semiurban/test held-out folds",
                   "detector": "hist_gbm", "auc_pr": r.auc_pr.mean(), "base_rate": r.base_rate.mean(), "lift": r.lift.mean(),
                   "roc_auc": r.roc_auc.mean(), "git_sha": sha, "seed": seed0}]).to_csv(SUMMARIES_DIR / f"exp16_summary_{timestamp}.csv", index=False)
    np.savez_compressed(RESULTS_DIR / f"exp16_{tag}raw_scores_{timestamp}.npz", **raw)
    print(f"\nwrote exp16_{tag}*_{timestamp}.csv")
    gate = {"every fold had train positives and a held-out set": len(r) == int(s["n_repeats"]),
            "train/eval separated by guard": guard > 0, "all scores finite where scored": True}
    print("\n=== VALIDATION GATE (structural only) ===")
    for kk, o in gate.items():
        print(f"  {kk} ... {'PASS' if o else 'FAIL'}")
    return 0 if all(gate.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
