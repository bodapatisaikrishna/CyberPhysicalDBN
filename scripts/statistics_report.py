"""Uncertainty and significance for the pinned results (LAB_NOTEBOOK.md 2026-10-16 analysis plan, fix #4).

No experiment is rerun. Every input is a pinned results file; every point estimate is recomputed first and
asserted equal to the pinned value before any interval or test is added. Outputs: results/summary/stats_*.csv.

Run: .venv/bin/python scripts/statistics_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

from build_summary_tables import CANONICAL, RESULTS_DIR, SUMMARY_DIR
from src.eval.calibration import expected_calibration_error
from src.eval.lead_time import evaluate_run

B = 2000
THETA = 0.49  # the headline threshold used by exp06 / exp09 / claims_summary
EXP06 = RESULTS_DIR / "exp06_raw_test_scores_20261010T075400Z.csv"
EXP06_SUMMARY = RESULTS_DIR / "exp06_comparison_summary_20261010T075400Z.csv"
EXP09 = RESULTS_DIR / "exp09_raw_eval_scores_20261010T075400Z.csv"
EXP20 = {"exp20": "20261016T092142Z", "exp20b": "20261016T142515Z"}
BLOCK = 450  # 15 min of 2 s slices


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj.tolist()


def ci(x: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def lower_median(v: np.ndarray) -> float:
    """src/eval/lead_time.summarize's median convention: sorted[int(0.5 * (n - 1))]."""
    f = np.sort(v[np.isfinite(v)])
    return float(f[int(0.5 * (len(f) - 1))]) if len(f) else float("nan")


def ece10(y, p) -> float:
    return expected_calibration_error(y, p, n_bins=10, strategy="uniform")[0]


def per_run_leads(df: pd.DataFrame) -> dict[int, float]:
    """lead (slices) per run at THETA; NaN when the run has no defined lead (missed / no event)."""
    out = {}
    for rid, g in df.groupby("run_id", sort=True):
        r = evaluate_run(g["y_prob"].to_numpy(), g["y_true"].to_numpy().astype(bool), THETA, 1.0)
        out[int(rid)] = float(r.lead_time_slices) if r.lead_time_slices is not None else np.nan
    return out


def c2(rng) -> pd.DataFrame:
    ev = pd.read_csv(CANONICAL["exp08_transfer_eval"])
    mk = ev.drop_duplicates(["graph_id", "arm"]).pivot(index="graph_id", columns="arm", values="m_kl")
    d = (mk["amortized"] - mk["table3"]).to_numpy()
    assert abs(mk["amortized"].mean() - 0.2260) < 5e-4 and abs(mk["table3"].mean() - 0.2960) < 5e-4, "C2 point estimate drifted"
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    w = wilcoxon(mk["amortized"], mk["table3"], zero_method="wilcox")
    det = ev[np.isclose(ev.threshold, 0.5)].pivot(index="graph_id", columns="arm", values="outcome")
    hit = det.isin(["detected_before", "detected_after"])
    event = det["amortized"].ne("no_event") | det["table3"].ne("no_event")
    b = int((hit.amortized & ~hit.table3 & event).sum()); c = int((~hit.amortized & hit.table3 & event).sum())
    mcn = binomtest(b, b + c, 0.5).pvalue if b + c else float("nan")
    return pd.DataFrame([
        {"claim": "C2", "metric": "mean M_KL learned - expert (25 graphs)", "estimate": float(d.mean()),
         "ci_lo": ci(np.array(boot))[0], "ci_hi": ci(np.array(boot))[1], "test": "Wilcoxon signed-rank", "p": float(w.pvalue),
         "n": len(d), "note": f"learned {mk['amortized'].mean():.4f}, expert {mk['table3'].mean():.4f}; "
                              f"graphs where learned < expert: {int((d < 0).sum())}, equal: {int((d == 0).sum())}"},
        {"claim": "C2", "metric": "detection at theta=0.5, discordant event graphs (learned-only / expert-only)",
         "estimate": float(b - c), "ci_lo": np.nan, "ci_hi": np.nan, "test": "exact McNemar", "p": float(mcn),
         "n": int(event.sum()), "note": f"learned-only {b}, expert-only {c}"},
    ])


def baselines(rng) -> pd.DataFrame:
    raw = pd.read_csv(EXP06); pinned = pd.read_csv(EXP06_SUMMARY).set_index("system")
    pinned_lt = pd.read_csv(RESULTS_DIR / "exp06_comparison_leadtime_20261010T075400Z.csv")
    runs = np.sort(raw.run_id.unique()); rows = []
    by = {s: {int(r): g for r, g in d.groupby("run_id")} for s, d in raw.groupby("system")}
    leads = {s: per_run_leads(d) for s, d in raw.groupby("system")}
    for s, d in raw.groupby("system"):
        ap, e = average_precision_score(d.y_true, d.y_prob), ece10(d.y_true, d.y_prob)
        assert abs(ap - pinned.loc[s, "auc_pr"]) < 1e-9 and abs(e - pinned.loc[s, "ece_10_uniform"]) < 1e-9, f"exp06 {s} drifted"
        bap, bece = [], []
        for _ in range(B):
            g = pd.concat([by[s][int(r)] for r in rng.choice(runs, len(runs))])
            bap.append(average_precision_score(g.y_true, g.y_prob)); bece.append(ece10(g.y_true, g.y_prob))
        lv = np.array([leads[s][int(r)] for r in runs])
        pl = pinned_lt[(pinned_lt.system == s) & np.isclose(pinned_lt.threshold, THETA)].iloc[0]
        assert lower_median(lv) == pl.lead_median_slices and abs(np.nanmean(lv) - pl.lead_mean_slices) < 1e-9, f"exp06 {s} lead drifted"
        rows += [{"claim": "baselines", "metric": f"{s} AUC-PR", "estimate": ap, "ci_lo": ci(np.array(bap))[0], "ci_hi": ci(np.array(bap))[1],
                  "test": "run-cluster bootstrap", "p": np.nan, "n": len(runs), "note": ""},
                 {"claim": "baselines", "metric": f"{s} ECE (10-bin)", "estimate": e, "ci_lo": ci(np.array(bece))[0], "ci_hi": ci(np.array(bece))[1],
                  "test": "run-cluster bootstrap", "p": np.nan, "n": len(runs), "note": ""},
                 {"claim": "baselines", "metric": f"{s} median lead (slices), theta={THETA}", "estimate": lower_median(lv),
                  "ci_lo": np.nan, "ci_hi": np.nan, "test": "", "p": np.nan, "n": int(np.isfinite(lv).sum()),
                  "note": f"mean {np.nanmean(lv):.3f}"}]
    dbn = "dbn_soft_calibrated"; tests = []
    for s in leads:
        if s == dbn:
            continue
        a = np.array([leads[s][int(r)] for r in runs]); b = np.array([leads[dbn][int(r)] for r in runs])
        ok = np.isfinite(a) & np.isfinite(b); diff = a[ok] - b[ok]
        p = wilcoxon(a[ok], b[ok]).pvalue if np.any(diff != 0) else 1.0
        boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(B)]
        tests.append({"claim": "baselines", "metric": f"lead {s} - DBN (slices), theta={THETA}", "estimate": float(diff.mean()),
                      "ci_lo": ci(np.array(boot))[0], "ci_hi": ci(np.array(boot))[1], "test": "paired Wilcoxon", "p": float(p),
                      "n": int(ok.sum()), "note": ""})
    for t, padj in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = padj
    return pd.DataFrame(rows + tests)


def c3(rng) -> pd.DataFrame:
    raw = pd.read_csv(EXP09); rows, tests = [], []
    pinned = pd.read_csv(CANONICAL["exp09_robustness_curve"])
    leads = {(s, k): per_run_leads(d) for (s, k), d in raw.groupby(["system", "knowledge_level"])}
    for (s, k), lv in leads.items():
        v = np.array(list(lv.values())); f = v[np.isfinite(v)]
        pin = pinned[(pinned.system == s) & (pinned.knowledge_level == k) & np.isclose(pinned.threshold, THETA)].lead_mean_slices
        if len(f) and len(pin) and np.isfinite(pin.iloc[0]):
            assert abs(f.mean() - pin.iloc[0]) < 1e-9, f"exp09 {s}/{k} lead drifted"
        boot = [f[rng.integers(0, len(f), len(f))].mean() for _ in range(B)] if len(f) else [np.nan]
        rows.append({"claim": "C3", "metric": f"{s} / {k} mean lead (slices), theta={THETA}", "estimate": float(f.mean()) if len(f) else np.nan,
                     "ci_lo": ci(np.array(boot))[0] if len(f) else np.nan, "ci_hi": ci(np.array(boot))[1] if len(f) else np.nan,
                     "test": "bootstrap over runs", "p": np.nan, "n": len(f), "note": f"{len(v) - len(f)} runs without a defined lead"})
    for k in ("blind", "analytics", "full_dbn"):
        a, b = leads[("lstm_ae", k)], leads[("dbn_hard_evidence", k)]
        ids = sorted(set(a) & set(b)); x = np.array([a[i] for i in ids]); y = np.array([b[i] for i in ids])
        ok = np.isfinite(x) & np.isfinite(y); diff = x[ok] - y[ok]
        p = wilcoxon(x[ok], y[ok]).pvalue if np.any(diff != 0) else 1.0
        boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(B)] if len(diff) else [np.nan]
        tests.append({"claim": "C3", "metric": f"lead LSTM-AE - DBN, {k} attacker", "estimate": float(diff.mean()) if len(diff) else np.nan,
                      "ci_lo": ci(np.array(boot))[0], "ci_hi": ci(np.array(boot))[1], "test": "paired Wilcoxon (same run ids)",
                      "p": float(p), "n": int(ok.sum()), "note": ""})
    for t, padj in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = padj
    bl = np.array([v for v in leads[("dbn_hard_evidence", "blind")].values() if np.isfinite(v)])
    fd = np.array([v for v in leads[("dbn_hard_evidence", "full_dbn")].values() if np.isfinite(v)])
    boot = [fd[rng.integers(0, len(fd), len(fd))].mean() - bl[rng.integers(0, len(bl), len(bl))].mean() for _ in range(B)]
    tests.append({"claim": "C3", "metric": "DBN lead full-DBN attacker - blind attacker", "estimate": float(fd.mean() - bl.mean()),
                  "ci_lo": ci(np.array(boot))[0], "ci_hi": ci(np.array(boot))[1], "test": "unpaired bootstrap", "p": np.nan,
                  "n": len(bl) + len(fd), "note": "different attackers -> unpaired"})
    return pd.DataFrame(rows + tests)


def real_data(rng) -> pd.DataFrame:
    rows = []
    for tag, ts in EXP20.items():
        raw = np.load(RESULTS_DIR / f"{tag}_raw_scores_{ts}.npz")
        met = pd.read_csv(RESULTS_DIR / f"{tag}_metrics_{ts}.csv")
        tests = []
        for t in ("01-Basic", "02-Semiurban", "03-Rural"):
            y = raw[f"{t}__y"]; n = len(y); nb = int(np.ceil(n / BLOCK))
            sc = {d: raw[f"{t}__{d}"].astype(np.float64) for d in ("dbn", "flat", "pca_spe")}
            point = {d: roc_auc_score(y, s) for d, s in sc.items()}
            for d in sc:
                pin = met[(met.target == t) & (met.detector == d)].roc_auc.iloc[0]
                assert abs(point[d] - pin) < 1e-6, f"{tag} {t} {d} ROC drifted"
            boots = {k: [] for k in ("dbn", "flat", "pca_spe", "dbn-pca_spe", "dbn-flat")}
            for _ in range(B):
                starts = rng.integers(0, n - BLOCK + 1, nb)
                idx = (starts[:, None] + np.arange(BLOCK)[None, :]).ravel()[:n]
                yy = y[idx]
                if yy.min() == yy.max():
                    continue
                r = {d: roc_auc_score(yy, s[idx]) for d, s in sc.items()}
                for d in sc:
                    boots[d].append(r[d])
                boots["dbn-pca_spe"].append(r["dbn"] - r["pca_spe"]); boots["dbn-flat"].append(r["dbn"] - r["flat"])
            for d in sc:
                rows.append({"claim": f"real data ({tag})", "metric": f"{t} ROC-AUC {d}", "estimate": point[d],
                             "ci_lo": ci(np.array(boots[d]))[0], "ci_hi": ci(np.array(boots[d]))[1],
                             "test": "moving-block bootstrap (15 min)", "p": np.nan, "n": n, "note": ""})
            for k in ("dbn-pca_spe", "dbn-flat"):
                bb = np.array(boots[k]); est = point["dbn"] - point[k.split("-", 1)[1]]
                p = float(min(1.0, 2 * min((bb <= 0).mean(), (bb >= 0).mean())))
                tests.append({"claim": f"real data ({tag})", "metric": f"{t} ROC-AUC {k}", "estimate": est,
                              "ci_lo": ci(bb)[0], "ci_hi": ci(bb)[1], "test": "block-bootstrap two-sided", "p": p, "n": n, "note": ""})
        for t, padj in zip(tests, holm([t["p"] for t in tests])):
            t["p_holm"] = padj
        rows += tests
    return pd.DataFrame(rows)


def main() -> int:
    seed = int(yaml.safe_load((REPO_ROOT / "configs" / "base.yaml").read_text())["seed"])
    rng = np.random.default_rng(seed)
    parts = {"c2": c2(rng), "baselines": baselines(rng), "c3": c3(rng), "real_data": real_data(rng)}
    for name, df in parts.items():
        df["bootstrap_reps"] = B; df["seed"] = seed
        df.to_csv(SUMMARY_DIR / f"stats_{name}.csv", index=False)
        print(f"\n== {name} ==")
        with pd.option_context("display.width", 220, "display.max_colwidth", 70):
            print(df.drop(columns=["bootstrap_reps", "seed"]).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
