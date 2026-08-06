"""Reproducibility check (Session 10, task 2): re-run experiments/exp01_
reproduce_paper.py and experiments/exp03_twin_open_loop.py fresh, from
their existing configs/seeds, and diff every numeric column against the
canonical logged CSVs from their original sessions, at a strict tolerance
(rtol=1e-9, atol=1e-12). Neither script has a torch dependency or any
sampling that affects its output (exp01: pure deterministic forward
filtering; exp03: seeded SeedSequence/default_rng, single-threaded SimPy),
so bit-for-bit reproduction is the expectation, not merely "close."

exp08_transfer_c2.py is deliberately excluded here (~40 min wall-clock,
trains a torch MLP, and its own CSVs have no `seed` column at all -- see
LAB_NOTEBOOK.md Session 10 entry / this session's audit findings).

Fresh output files are matched to their canonical references by CONTENT
TYPE (e.g. "scenario1_ff", "summary"), not by exact filename: exp01's
filename convention has itself drifted since the canonical run (a later
session added a `_{reaction_mode}_` segment), so exact-name matching would
spuriously fail here. Matching is done by (a) restricting to files with
mtime after this script's own invocation, and (b) a small per-experiment
regex extracting the stable "type key" from each filename.

Run: .venv/bin/python scripts/verify_reproducibility.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
SUMMARIES_DIR = RESULTS_DIR / "summaries"
PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")

RTOL = 1e-9
ATOL = 1e-12

# Non-value columns: identifiers/keys (compared for exact equality of ROW
# SET, not "numeric drift") or expected-to-differ-by-construction (git_sha
# is a different commit's dirty-tree diff each run; timestamps differ by
# definition). Never diffed as "drift".
IGNORE_COLUMNS = {"git_sha"}

# Wall-clock/resource-usage columns are inherently non-reproducible (depend
# on machine load/scheduling/allocator behavior at run time, not on the
# computation itself) -- reported for visibility but never counted as a
# reproducibility FAILURE. This is what distinguishes "the science
# reproduces" (p_active, m_kl, argmax slice -- all deterministic given a
# fixed seed/config) from "the wall-clock happened to match" (never
# expected, not what this check is for).
SOFT_COLUMNS = {"latency_s", "mean_latency_s", "peak_tracemalloc_bytes", "rss_delta_bytes"}

EXP01_CANONICAL = {
    "scenario1_ff": RESULTS_DIR / "exp01_scenario1_ff_20260731T164052Z.csv",
    "scenario1_ex": RESULTS_DIR / "exp01_scenario1_ex_20260731T164052Z.csv",
    "scenario2_ff": RESULTS_DIR / "exp01_scenario2_ff_20260731T164052Z.csv",
    "scenario2_ex": RESULTS_DIR / "exp01_scenario2_ex_20260731T164052Z.csv",
    "summary": SUMMARIES_DIR / "exp01_summary_20260731T164052Z.csv",
}
EXP01_KEY_COLUMNS = {
    "scenario1_ff": ["node", "slice"], "scenario1_ex": ["node", "slice"],
    "scenario2_ff": ["node", "slice"], "scenario2_ex": ["node", "slice"],
    "summary": ["scenario", "node"],
}
EXP01_FRESH_PATTERN = re.compile(r"^exp01_(scenario[12]_(?:ff|ex))_\w*_?\d{8}T\d{6}Z\.csv$")
EXP01_SUMMARY_PATTERN = re.compile(r"^exp01_summary_\w*_?\d{8}T\d{6}Z\.csv$")

EXP03_CANONICAL = {
    "grid_sweep": RESULTS_DIR / "exp03_grid_sweep_20260801T123239Z.csv",
    "twin_slices": RESULTS_DIR / "exp03_twin_slices_20260801T123239Z.csv",
    "summary": SUMMARIES_DIR / "exp03_twin_summary_20260801T123239Z.csv",
}
EXP03_KEY_COLUMNS = {
    "grid_sweep": ["p_mw_per_der"],
    "twin_slices": ["arm", "replicate", "slice", "var_kind", "var_name"],
    "summary": ["metric", "arm", "node"],
}


def _import_pandas_numpy():
    import numpy as np
    import pandas as pd
    return pd, np


def run_fresh(script: str) -> float:
    # NOTE: exp01/exp03's own exit code reflects THEIR OWN internal
    # validation gate (e.g. exp01 compares against the source paper's
    # published targets, some of which are already known -- from prior
    # sessions -- to not match exactly; LAB_NOTEBOOK.md records this as an
    # accepted, investigated condition, not a bug). That gate is a
    # different question from "did this run reproduce its own prior
    # numbers." A nonzero exit here is expected and must NOT abort the
    # reproducibility check -- only a crash before any CSV is written
    # would actually prevent diffing, and that would show up downstream as
    # "fresh file not found."
    print(f"running {script} fresh ...", flush=True)
    t0 = time.time()
    result = subprocess.run([PYTHON, str(REPO_ROOT / "experiments" / script)], cwd=REPO_ROOT)
    elapsed = time.time() - t0
    print(f"  {script} finished in {elapsed:.1f}s (exit code {result.returncode}, "
          f"see note above on why this is not itself a repro failure)", flush=True)
    return t0


def newest_matching(pattern: re.Pattern, directory: Path, after_ts: float) -> list[Path]:
    out = []
    for p in directory.glob("*.csv"):
        if p.stat().st_mtime >= after_ts and pattern.match(p.name):
            out.append(p)
    return out


def diff_frame(
    fresh_path: Path, canonical_path: Path, key_columns: list[str] | None, exclude_node_pattern: str | None = None
) -> dict:
    pd, np = _import_pandas_numpy()
    fresh = pd.read_csv(fresh_path)
    canon = pd.read_csv(canonical_path)

    if exclude_node_pattern is not None and "node" in fresh.columns:
        # exp01's summary CSV reuses the m_kl/argmax_t COLUMNS for two
        # different quantities: real KL values (deterministic, node in
        # QUERY_NODES) and mean-latency/peak-memory pseudo-rows (node ==
        # "__{clustering}_perf__", inherently non-reproducible wall-clock
        # measurements already covered by SOFT_COLUMNS at the scenario-CSV
        # level). Diffing the pseudo-rows here under the m_kl/argmax_t
        # names would misreport a wall-clock difference as a KL/argmax
        # mismatch -- exclude them entirely rather than mislabel them.
        fresh = fresh[~fresh["node"].str.match(exclude_node_pattern)]
        canon = canon[~canon["node"].str.match(exclude_node_pattern)]

    report = {"fresh": str(fresh_path), "canonical": str(canonical_path)}

    if set(fresh.columns) != set(canon.columns):
        report["column_mismatch"] = {
            "fresh_only": sorted(set(fresh.columns) - set(canon.columns)),
            "canonical_only": sorted(set(canon.columns) - set(fresh.columns)),
        }
        report["ok"] = False
        return report

    if key_columns:
        fresh = fresh.sort_values(key_columns).reset_index(drop=True)
        canon = canon.sort_values(key_columns).reset_index(drop=True)

    if len(fresh) != len(canon):
        report["row_count_mismatch"] = {"fresh": len(fresh), "canonical": len(canon)}
        report["ok"] = False
        return report

    numeric_cols = [
        c for c in fresh.columns
        if c not in IGNORE_COLUMNS and pd.api.types.is_numeric_dtype(fresh[c]) and pd.api.types.is_numeric_dtype(canon[c])
    ]
    col_reports = {}
    ok = True
    for c in numeric_cols:
        a = fresh[c].to_numpy(dtype=float)
        b = canon[c].to_numpy(dtype=float)
        is_soft = c in SOFT_COLUMNS
        finite = np.isfinite(a) & np.isfinite(b)
        if not finite.all() and (np.isfinite(a) != np.isfinite(b)).any():
            col_reports[c] = {"max_abs_diff": float("inf"), "max_rel_diff": float("inf"), "note": "finiteness mismatch", "soft": is_soft}
            if not is_soft:
                ok = False
            continue
        abs_diff = np.abs(a[finite] - b[finite])
        rel_diff = abs_diff / np.maximum(np.abs(b[finite]), 1e-300)
        max_abs = float(abs_diff.max()) if abs_diff.size else 0.0
        max_rel = float(rel_diff.max()) if rel_diff.size else 0.0
        col_reports[c] = {"max_abs_diff": max_abs, "max_rel_diff": max_rel, "soft": is_soft}
        within_tol = np.allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True)
        if not within_tol and not is_soft:
            ok = False

    report["columns"] = col_reports
    report["ok"] = ok
    return report


def main() -> int:
    RESULTS_DIR.mkdir(exist_ok=True)
    reports = []
    overall_ok = True

    # --- exp01 ---
    t0 = run_fresh("exp01_reproduce_paper.py")
    scenario_files = newest_matching(EXP01_FRESH_PATTERN, RESULTS_DIR, t0)
    summary_files = newest_matching(EXP01_SUMMARY_PATTERN, SUMMARIES_DIR, t0)

    by_type = {}
    for p in scenario_files:
        m = EXP01_FRESH_PATTERN.match(p.name)
        by_type[m.group(1)] = p
    if summary_files:
        by_type["summary"] = max(summary_files, key=lambda p: p.stat().st_mtime)

    for type_key, canonical_path in EXP01_CANONICAL.items():
        if type_key not in by_type:
            reports.append({"experiment": "exp01", "type": type_key, "ok": False, "error": "fresh file not found"})
            overall_ok = False
            continue
        exclude_pat = r"__.*_perf__" if type_key == "summary" else None
        r = diff_frame(by_type[type_key], canonical_path, EXP01_KEY_COLUMNS[type_key], exclude_pat)
        r["experiment"] = "exp01"
        r["type"] = type_key
        reports.append(r)
        overall_ok = overall_ok and r["ok"]

    # --- exp03 ---
    t0 = run_fresh("exp03_twin_open_loop.py")
    exp03_fresh = {}
    for p in RESULTS_DIR.glob("exp03_grid_sweep_*.csv"):
        if p.stat().st_mtime >= t0:
            exp03_fresh["grid_sweep"] = p
    for p in RESULTS_DIR.glob("exp03_twin_slices_*.csv"):
        if p.stat().st_mtime >= t0:
            exp03_fresh["twin_slices"] = p
    for p in SUMMARIES_DIR.glob("exp03_twin_summary_*.csv"):
        if p.stat().st_mtime >= t0:
            exp03_fresh["summary"] = p

    for type_key, canonical_path in EXP03_CANONICAL.items():
        if type_key not in exp03_fresh:
            reports.append({"experiment": "exp03", "type": type_key, "ok": False, "error": "fresh file not found"})
            overall_ok = False
            continue
        r = diff_frame(exp03_fresh[type_key], canonical_path, EXP03_KEY_COLUMNS[type_key])
        r["experiment"] = "exp03"
        r["type"] = type_key
        reports.append(r)
        overall_ok = overall_ok and r["ok"]

    # --- report ---
    print("\n=== REPRODUCIBILITY CHECK ===")
    for r in reports:
        status = "PASS" if r.get("ok") else "FAIL"
        print(f"\n[{r['experiment']}/{r['type']}] {status}")
        print(f"  fresh:     {r.get('fresh')}")
        print(f"  canonical: {r.get('canonical')}")
        if "error" in r:
            print(f"  error: {r['error']}")
        if "column_mismatch" in r:
            print(f"  column mismatch: {r['column_mismatch']}")
        if "row_count_mismatch" in r:
            print(f"  row count mismatch: {r['row_count_mismatch']}")
        for c, stats in r.get("columns", {}).items():
            is_zero = stats.get("max_abs_diff", 0) == 0 and stats.get("max_rel_diff", 0) == 0
            if stats.get("soft"):
                flag = "  (wall-clock/memory, not gated)" if not is_zero else "  (wall-clock/memory, not gated)"
            else:
                flag = "" if is_zero else "  <-- non-zero diff (GATED)"
            print(f"  {c:<24} max_abs_diff={stats.get('max_abs_diff'):.3e} max_rel_diff={stats.get('max_rel_diff'):.3e}{flag}")

    print(f"\n{'PASS' if overall_ok else 'FAIL'}: reproducibility check {'passed' if overall_ok else 'FAILED'} "
          f"(rtol={RTOL}, atol={ATOL})")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
