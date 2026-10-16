"""exp20: the DBN itself on real Sherlock data (LAB_NOTEBOOK.md 2026-10-16, pre-registered D-1..D-4).

All three networks were already spent as test sets (exp19), so every number here is EXPLORATORY.

Per target network (leave-one-network-out over the three labelled TEST files):
  evidence  : exp19's per-run causal robust z (no clean data of the target needed), reduced to one
              signed-max score per analytic feature group (src/attack_graph/sherlock_graph.ANALYTICS)
  DBN       : Sherlock attack graph, transition rates = ML estimates from the two OTHER files' labels,
              per-analytic logistic calibration (same files) entered as prior-corrected virtual evidence,
              exact (EX) forward filtering; score = P(AttackImpact)
  FLAT      : logistic regression on the same 6 group scores, same fit files and labels
  PCA       : exp19's label-free detector recomputed with exp19's own folds (reproducibility check)
Results are REPORTED, never gated (CLAUDE.md rule 3).

Run: .venv/bin/python experiments/exp20_sherlock_dbn.py [--config PATH] [--smoke]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression

from src.attack_graph.sherlock_graph import (ANALYTICS, FAMILY_STEPS, GOAL, STEPS, build_sherlock_attack_graph,
                                             step_labels, step_transition_rates)
from src.dbn.inference import DBNInference, InferenceConfig, _interface_nodes, exact_clustering
from src.dbn.soft_evidence import SoftEvidenceConfig
from src.eval.provenance import git_sha
from src.perception.sherlock_anomaly import blocked_split, event_runs
from src.perception.sherlock_component_detectors import causal_rolling_robust_z, component_scores, fit_component_model
from src.perception.sherlock_loader import StateFileFeatures
from src.perception.sherlock_network import FEATURE_NAMES

RESULTS_DIR = REPO_ROOT / "results"


def _path_import(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "experiments" / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


e13 = _path_import("exp13_sherlock_full", "exp13_sherlock_full.py")
e17 = _path_import("exp17_sherlock_network", "exp17_sherlock_network.py")
e19 = _path_import("exp19_sherlock_universal", "exp19_sherlock_universal.py")

GROUP_FEATURES = sorted({f for _, grp in ANALYTICS.values() for f, _ in grp})
PHYS_FEATURES = ("n_switch_changes", "nonfinite_total")


def family_of(description: str) -> str:
    return description.split(":")[0]


def load(scenario: str, split: str, cfg: dict) -> dict:
    """z for every group feature, labels, per-slice attack family, catalog. exp19's z settings."""
    bin_s, rb = float(cfg["grid"]["bin_seconds"]), cfg["rolling_baseline"]
    xp, counts, y, ids, t_end, catalog, info = e17.load_split(scenario, split, bin_s)
    # allow_pickle: this repo's own locally built feature cache (scripts/build_sherlock_*), same as exp17/exp19
    phys = np.load(e17.CACHE / f"{scenario}__physical_component__{split}__physical.zip.npz", allow_pickle=True)
    pnames = list(json.loads(str(phys["meta"]))["column_names"])
    net_cols = [f for f in GROUP_FEATURES if f not in PHYS_FEATURES]
    x = torch.cat([counts[:, [FEATURE_NAMES.index(f) for f in net_cols]], e19.physical_events(xp, pnames)], dim=1)
    names = net_cols + list(PHYS_FEATURES)
    z, _ = causal_rolling_robust_z(x, window=int(rb["window"]), min_history=int(rb["min_history"]), scale_floor=None,
                                   z_clip=float(rb["z_clip"]), self_floor=True)
    z = z.numpy().astype(np.float64)
    cat = {str(e["id"]): e for e in catalog}
    fam = [family_of(cat[i]["description"]) if i is not None else None for i in ids]
    benign = np.zeros(len(t_end), dtype=bool)
    for e in catalog:
        if "benign" in str(e["id"]):
            benign |= (t_end >= float(e["start"])) & (t_end < float(e["end"]))
    groups = np.stack([np.max(np.stack([sgn * z[:, names.index(f)] for f, sgn in grp], axis=1), axis=1)
                       for _, grp in ANALYTICS.values()], axis=1)
    groups = np.nan_to_num(groups, nan=0.0)  # undefined z before min_history -> "no deviation"
    return {"scenario": scenario, "y": y.numpy().astype(np.int8), "ids": ids, "t_end": t_end, "catalog": catalog,
            "fam": fam, "steps": step_labels(fam), "groups": groups, "benign": benign}


def sherlock_file(d: dict, bin_s: float):
    feats = StateFileFeatures(features=torch.zeros(len(d["y"]), 1), labels=torch.from_numpy(d["y"].astype(np.float32)),
                              timestamps=d["t_end"], event_ids=d["ids"], raw_label_counts={}, component_key_counts={},
                              n_state_keys_first_record=0)
    return e13.SherlockFile(scenario=d["scenario"], kind="test", path=Path("grid"), feats=feats, cadence_median=bin_s,
                            cadence_n_bad=0, base_rate=float(d["y"].mean()), runs=event_runs(d["ids"]),
                            catalog={str(e["id"]): e for e in d["catalog"]}, catalog_path=None)


def pca_scores(target: str, fit_from: list[str], cfg: dict, cache: dict) -> tuple[np.ndarray, float]:
    """exp19's pca_spe arm and static threshold, recomputed with exp19's own code and folds."""
    p, bin_s, rb = cfg["pca"], float(cfg["grid"]["bin_seconds"]), cfg["rolling_baseline"]

    def view(sc: str, split: str) -> torch.Tensor:
        key = (sc, split)
        if key not in cache:
            xp, counts, *_ = e17.load_split(sc, split, bin_s)
            phys = np.load(e17.CACHE / f"{sc}__physical_component__{split}__physical.zip.npz", allow_pickle=True)
            names = list(json.loads(str(phys["meta"]))["column_names"])
            x = torch.cat([counts[:, e19.NET_IDX], e19.physical_events(xp, names)], dim=1)
            cache[key], _ = causal_rolling_robust_z(x, window=int(rb["window"]), min_history=int(rb["min_history"]),
                                                    scale_floor=None, z_clip=float(rb["z_clip"]), self_floor=True)
        return cache[key]

    fit_z, cal = [], []
    for sc in fit_from:
        z = view(sc, "train")
        parts = blocked_split(z.shape[0], int(p["block_slices"]), tuple(p["pattern"]))
        fit_z.append(torch.cat([z[a:b] for a, b in parts["fit"]]))
    mdl = fit_component_model(torch.cat(fit_z), eps=float(p["standardizer_eps"]), z_clip=float(rb["z_clip"]),
                              explained_variance_target=float(p["explained_variance_target"]),
                              max_components=len(e19.FEATURES), fit_split=f"clean fit blocks of {fit_from}")
    for sc in fit_from:
        z = view(sc, "train")
        parts = blocked_split(z.shape[0], int(p["block_slices"]), tuple(p["pattern"]))
        s = component_scores(mdl, z, int(p["top_k"]), 4096)["pca_spe"]
        cal.append(np.concatenate([s[a:b] for a, b in parts["calib"]]))
    theta = float(np.percentile(np.concatenate(cal), float(p["percentile"])))
    return component_scores(mdl, view(target, "test"), int(p["top_k"]), 4096)["pca_spe"], theta


def nearest_family(post: dict[str, float]) -> str:
    """Attack family whose step signature (1 = step active) is nearest, in L1, to the mean step posteriors."""
    return min(FAMILY_STEPS, key=lambda f: sum(abs(post[s] - (s in FAMILY_STEPS[f])) for s in STEPS))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "sherlock_dbn.yaml"))
    ap.add_argument("--smoke", action="store_true", help="first 3000 slices of each file; nothing written")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    base_cfg = yaml.safe_load((REPO_ROOT / "configs" / "base.yaml").read_text())
    seed = int(base_cfg["seed"])
    e13.set_all_seeds(seed)
    ts_run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = git_sha(REPO_ROOT)
    bin_s = float(cfg["grid"]["bin_seconds"])
    print(f"exp20 seed={seed} git_sha={sha} smoke={args.smoke}; EXPLORATORY (all Sherlock networks already spent)", flush=True)

    data = {}
    for sc in cfg["targets"]:
        d = load(sc, "test", cfg)
        if args.smoke:
            n = 3000
            d = {k: (v[:n] if isinstance(v, (np.ndarray, tuple, list)) and k != "catalog" else v) for k, v in d.items()}
            d["steps"] = {s: a[:n] for s, a in d["steps"].items()}
        data[sc] = d
        print(f"  {sc}/test: {len(d['y'])} slices, attack base rate {d['y'].mean():.4f}, "
              f"benign-event slices {int(d['benign'].sum())}", flush=True)

    metric_rows, ev_rows, benign_rows, attr_rows, param_rows, raw = [], [], [], [], [], {}
    pca_cache: dict = {}
    an_names = list(ANALYTICS)
    for target in cfg["targets"]:
        fit = [sc for sc in cfg["targets"] if sc != target]
        assert target not in fit
        # --- learn transition rates and evidence calibration on the OTHER files only
        rates = step_transition_rates([data[sc]["steps"] for sc in fit])
        G = np.concatenate([data[sc]["groups"] for sc in fit])
        calib, base_rates = {}, {}
        for j, a in enumerate(an_names):
            step = ANALYTICS[a][0]
            lab = np.concatenate([data[sc]["steps"][step] for sc in fit])
            calib[a] = LogisticRegression(max_iter=1000).fit(G[:, [j]], lab)
            base_rates[a] = float(lab.mean())
        for s, (p_s, p_end) in rates.items():
            param_rows.append({"target": target, "fit_from": "+".join(fit), "node": s, "p_s": p_s, "p_end": p_end,
                               "git_sha": sha, "seed": seed})
        for a in an_names:
            m = calib[a]
            param_rows.append({"target": target, "fit_from": "+".join(fit), "node": a, "logit_coef": float(m.coef_[0, 0]),
                               "logit_intercept": float(m.intercept_[0]), "step_base_rate": base_rates[a],
                               "git_sha": sha, "seed": seed})
        y_fit = np.concatenate([data[sc]["y"] for sc in fit])
        flat = LogisticRegression(max_iter=1000).fit(G, y_fit)
        print(f"\n[{target}] fit on {fit}; rates (p_s, p_end): "
              f"{ {s: (round(a, 6), round(b, 5)) for s, (a, b) in rates.items()} }", flush=True)

        ag = build_sherlock_attack_graph(rates, bin_s=bin_s, analytic_error=tuple(cfg["analytic_error"]),
                                         source=f"configs/sherlock_dbn.yaml analytic_error; fold {target}")
        icfg = InferenceConfig(clustering=exact_clustering(_interface_nodes(ag)), m=1.0, p_pos=cfg["analytic_error"][0],
                               p_neg=cfg["analytic_error"][1], delta_t_override=bin_s,
                               soft_evidence=SoftEvidenceConfig(targets=tuple(an_names), mode="prior_corrected",
                                                                base_rates=base_rates))
        clip = float(cfg["soft_q_clip"])

        def run_dbn(d: dict) -> dict[str, np.ndarray]:
            inf = DBNInference(ag, icfg)
            b = inf.initial_belief()
            q = np.column_stack([calib[a].predict_proba(d["groups"][:, [j]])[:, 1] for j, a in enumerate(an_names)])
            q = np.clip(q, clip, 1 - clip)
            out = {n: np.zeros(len(d["y"])) for n in STEPS + (GOAL,)}
            for t in range(len(d["y"])):
                r = inf.step(b, {}, soft={a: float(q[t, j]) for j, a in enumerate(an_names)})
                b = r.next_belief
                for n in out:
                    out[n][t] = r.marginals[n]
            return out

        t0 = time.time()
        post = run_dbn(data[target])
        print(f"  DBN filtered {len(data[target]['y'])} slices in {time.time() - t0:.0f} s", flush=True)
        # thresholds from the FIT files' non-attack slices (they include benign maintenance events)
        fit_post = {sc: run_dbn(data[sc])[GOAL] for sc in fit}
        q_pct = float(cfg["threshold_percentile"])
        theta_dbn = float(np.percentile(np.concatenate([fit_post[sc][data[sc]["y"] == 0] for sc in fit]), q_pct))
        fit_flat = np.concatenate([flat.predict_proba(data[sc]["groups"])[:, 1][data[sc]["y"] == 0] for sc in fit])
        theta_flat = float(np.percentile(fit_flat, q_pct))

        d = data[target]
        scores = {"dbn": post[GOAL], "flat": flat.predict_proba(d["groups"])[:, 1]}
        thetas = {"dbn": theta_dbn, "flat": theta_flat}
        if not args.smoke:
            scores["pca_spe"], thetas["pca_spe"] = pca_scores(target, cfg["pca_folds"][target], cfg, pca_cache)
        y = d["y"]
        for det, s in scores.items():
            ap_, roc = e13.ap_and_roc(y, s)
            alarm = s >= thetas[det]
            metric_rows.append({"target": target, "detector": det, "n": len(y), "base_rate": float(y.mean()),
                                "auc_pr": ap_, "lift": ap_ / float(y.mean()), "roc_auc": roc, "theta": thetas[det],
                                "fpr_nonattack": float(alarm[y == 0].mean()), "recall_at_alarm": float(alarm[y > 0].mean()),
                                "git_sha": sha, "seed": seed})
            print(f"  {det:<8} ROC-AUC {roc:.3f}  AUC-PR {ap_:.3f} (lift {ap_ / y.mean():.2f}x)  "
                  f"FPR {alarm[y == 0].mean():.3f}  recall {alarm[y > 0].mean():.3f}", flush=True)
            bn = d["benign"]
            benign_rows.append({"target": target, "detector": det, "n_benign_slices": int(bn.sum()),
                                "benign_alarm_rate": float(alarm[bn].mean()) if bn.any() else float("nan"),
                                "attack_alarm_rate": float(alarm[y > 0].mean()),
                                "mean_score_benign": float(s[bn].mean()) if bn.any() else float("nan"),
                                "mean_score_attack": float(s[y > 0].mean()), "git_sha": sha, "seed": seed})
            raw[f"{target}__{det}"] = s.astype(np.float32)
        raw[f"{target}__y"] = y
        for n in STEPS:
            raw[f"{target}__post__{n}"] = post[n].astype(np.float32)
        sf = sherlock_file(d, bin_s)
        ev = pd.DataFrame(e13.event_table(target, sf, scores, thetas))
        ev_rows.append(ev)
        for evid, a, b in sf.runs:
            seg_max = post[GOAL][a:b].max()
            mean_post = {s: float(post[s][a:b].mean()) for s in STEPS}
            true_f = family_of(sf.catalog[evid]["description"])
            pred_f = nearest_family(mean_post)
            attr_rows.append({"target": target, "event_id": evid, "family": true_f, "predicted_family": pred_f,
                              "correct": pred_f == true_f, "dbn_alarmed": bool(seg_max >= theta_dbn),
                              **{f"mean_post_{s}": v for s, v in mean_post.items()}, "git_sha": sha, "seed": seed})
        al = [r for r in attr_rows if r["target"] == target and r["dbn_alarmed"]]
        print(f"  attribution: {sum(r['correct'] for r in al)}/{len(al)} alarmed events assigned to the right family "
              f"(all events: {sum(r['correct'] for r in attr_rows if r['target'] == target)}/"
              f"{sum(1 for r in attr_rows if r['target'] == target)})", flush=True)

    if args.smoke:
        print("\nsmoke run: nothing written")
        return 0

    def out(name: str) -> Path:
        return RESULTS_DIR / f"exp20_{name}_{ts_run}.csv"
    met = pd.DataFrame(metric_rows); met.to_csv(out("metrics"), index=False)
    pd.concat(ev_rows).to_csv(out("events"), index=False)
    ben = pd.DataFrame(benign_rows); ben.to_csv(out("benign"), index=False)
    att = pd.DataFrame(attr_rows); att.to_csv(out("attribution"), index=False)
    pd.DataFrame(param_rows).to_csv(out("params"), index=False)
    np.savez_compressed(RESULTS_DIR / f"exp20_raw_scores_{ts_run}.npz", **raw)

    print("\n=== PRE-REGISTERED CRITERIA (reported, not gated; EXPLORATORY) ===")
    for t in cfg["targets"]:
        r = {det: met[(met.target == t) & (met.detector == det)].iloc[0] for det in ("dbn", "flat", "pca_spe")}
        b = {det: ben[(ben.target == t) & (ben.detector == det)].iloc[0] for det in ("dbn", "pca_spe")}
        a = att[(att.target == t) & att.dbn_alarmed]
        print(f" [{t}]")
        print(f"   D-1 DBN ROC {r['dbn'].roc_auc:.3f} >= PCA {r['pca_spe'].roc_auc:.3f} - 0.05: "
              f"{'MET' if r['dbn'].roc_auc >= r['pca_spe'].roc_auc - 0.05 else 'MISSED'}")
        print(f"   D-2 DBN ROC {r['dbn'].roc_auc:.3f} > FLAT {r['flat'].roc_auc:.3f}: "
              f"{'yes' if r['dbn'].roc_auc > r['flat'].roc_auc else 'no'}")
        ok3 = (b["dbn"].mean_score_benign < b["dbn"].mean_score_attack) and (b["dbn"].benign_alarm_rate <= b["pca_spe"].benign_alarm_rate)
        print(f"   D-3 benign alarm rate DBN {b['dbn'].benign_alarm_rate:.3f} vs PCA {b['pca_spe'].benign_alarm_rate:.3f}; "
              f"DBN mean score benign {b['dbn'].mean_score_benign:.3f} vs attack {b['dbn'].mean_score_attack:.3f}: {'MET' if ok3 else 'MISSED'}")
        acc = a.correct.mean() if len(a) else float("nan")
        print(f"   D-4 attribution {int(a.correct.sum())}/{len(a)} = {acc:.2f} (>= 0.50): {'MET' if acc >= 0.5 else 'MISSED'}")
    n_d2 = sum(met[(met.target == t) & (met.detector == "dbn")].roc_auc.iloc[0] > met[(met.target == t) & (met.detector == "flat")].roc_auc.iloc[0] for t in cfg["targets"])
    print(f" D-2 overall: DBN beats FLAT on {n_d2}/3 targets (criterion >= 2): {'MET' if n_d2 >= 2 else 'MISSED'}")
    pin = met[(met.target == "03-Rural") & (met.detector == "pca_spe")].roc_auc.iloc[0]
    print(f"\nreproducibility: recomputed PCA 03-Rural ROC {pin:.4f} (exp19 pinned 0.7462)")
    print(f"wrote exp20_*_{ts_run}.csv")
    ok = met.roc_auc.notna().all() and all(np.isfinite(v).all() for v in raw.values())
    print(f"\n=== VALIDATION GATE (structural only) ===\n  fit files exclude the target; all scores finite ... {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
