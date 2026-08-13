"""GNN-derived bus clustering vs. the project's heuristic zone map: measuring
"accuracy loss" via KL divergence (Session 12, faculty-requested, new scope
beyond CLAUDE.md's Sessions 0-10).

WHAT THIS COMPARES, PRECISELY. The project already has one heuristic bus
grouping: `ZoneMap` (`src/twin/consequence.py`), built from
`voltage_sensitivity()`'s DER-x-bus linearized sensitivity via a dominance-
share threshold rule, used by `classify()` to turn violated buses into
`PHYS_LOCAL_DER` (one zone hit) / `PHYS_WIDE_AREA` (multiple zones hit)
evidence. The project's GNN (`SpatialEncoder`) already produces a learned
per-bus embedding as an intermediate of the perception pipeline
(`h_dict["bus"]`); nothing in the codebase has ever clustered it. This script
does exactly that and compares the two groupings.

WHAT "KL DIVERGENCE" MEANS HERE, STATED EXPLICITLY. Two hard partitions of
the same 33 buses do NOT have a well-defined raw KL divergence against each
other -- there is no canonical correspondence between arbitrary cluster-index
labels from two INDEPENDENT clusterings (this is why the literature uses
Adjusted Rand Index / NMI for comparing partitions, not KL). Computing a KL
on raw cluster indices would be a meaningless number dressed up as a metric.
What DOES have a well-defined KL divergence: the DOWNSTREAM OBSERVABLE
DISTRIBUTIONS each zoning induces via the SAME `classify()` function on the
SAME twin traces (only the `zones` argument differs; nothing is
re-simulated). Three measures, reusing this project's own tested machinery:
  1. `adjusted_rand_score` (sklearn) -- the methodologically correct
     complement to KL for comparing partitions directly.
  2. `binary_kl`/`m_kl` (`src/eval/metrics.py`, already tested to 1e-12) on
     the pooled per-slice `PHYS_LOCAL_DER`/`PHYS_WIDE_AREA` rates, heuristic
     zoning as P (this project's established EX-as-P convention), GNN
     zoning as Q -- exactly mirrors exp01's `KL(EX||FF)` construction.
  3. Per-scenario `M_KL` on the downstream hard-evidence DBN's `UnstablePS`
     posterior (heuristic-zoned evidence as P), mean across scenarios --
     exp08's own "mean M_KL across N held-out graphs" convention.

WHAT GETS CLUSTERED, AND WHEN. `ZoneMap` is a STATIC object, computed once
from a small-perturbation linearization around the NOMINAL operating point
-- independent of any attack trajectory. For a fair comparison, the GNN
embeddings clustered must also represent nominal bus behavior. Per test
scenario, only the PRE-ATTACK slice prefix (every attack-step node's
ground_truth still 0 -- attack-step activation is monotonic/persistent in
this model, so this is simply the prefix up to the first slice where
anything activates) is used; embeddings are averaged over that window and
then across all 30 test scenarios, giving one static [33, 64] matrix.

k=2 for KMeans, matching ZoneMap's own zone count (n_der=2) exactly -- the
only apples-to-apples cardinality. Sensitivity to k is NOT swept this
session (stated limitation, not silently avoided).

A REAL, EXPECTED STRUCTURAL ASYMMETRY, reported not fixed: ZoneMap's
dominance-share rule can leave buses in `unassigned_buses`; KMeans assigns
every one of the 33 buses to a cluster by construction. The two partitions
differ in COVERAGE as well as membership.

No trained-model checkpoint exists anywhere in this project (neither exp05
nor exp11 calls `torch.save`) -- this script retrains a fresh
`PerceptionEncoder(encoder_type="gnn")` via exp05_perception.py's own
module-level functions (imported by path, the same pattern
exp06_baselines.py/exp11_perception_ablation.py already use), same
60/20/20/30 split and hyperparameters as every prior perception result.

Run: .venv/bin/python experiments/exp12_gnn_cluster_vs_heuristic.py [--smoke]
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from src.attack_graph.graph import PHYS_LOCAL_DER, PHYS_WIDE_AREA
from src.dbn.inference import DBNInference, InferenceConfig, _interface_nodes, fully_factorized_clustering
from src.eval.metrics import binary_kl, m_kl
from src.eval.provenance import git_sha
from src.perception.asset_graph import build_asset_graph, nonempty_metadata, observed_endpoints
from src.perception.encoder import (
    EncoderConfig,
    PerceptionEncoder,
    combine_static_dynamic,
    flatten_for_spatial,
    replicate_static_graph,
    unflatten_from_spatial,
)
from src.perception.features import DynamicFeatureConfig, build_dynamic_features
from src.twin.attacker import AttackerConfig, DelayLaw
from src.twin.consequence import DerZone, ZoneMap, observe_local, observe_wide
from src.twin.grid import GridConfig, GridModel
from src.twin.runner import ContinuousTrace, DiscreteTrace, TwinConfig, TwinRunner, discretize, validate_trace

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
TWIN_CONFIG_PATH = REPO_ROOT / "configs" / "twin.yaml"
PERCEPTION_CONFIG_PATH = REPO_ROOT / "configs" / "perception.yaml"
RESULTS_DIR = REPO_ROOT / "results"
SUMMARIES_DIR = RESULTS_DIR / "summaries"

K_CLUSTERS = 2  # matches ZoneMap's n_der=2 exactly -- see module docstring.

M = 1.0
P_POS = 1e-4
P_NEG = 1e-4
CYBER_ANALYTICS = (
    "FileAccess", "FileIntegrity", "MeasureCoherence", "CommandCoherence",
    "SWIntegrityDER", "NewServiceStarted", "SWIntegritySCADA", "SuspArg",
)


def _load_exp05():
    """Import experiments/exp05_perception.py by path -- the same pattern
    exp06_baselines.py/exp11_perception_ablation.py already use, so this
    script shares exp05's exact scenario-generation/training helpers
    rather than re-deriving them and risking drift."""
    path = REPO_ROOT / "experiments" / "exp05_perception.py"
    spec = importlib.util.spec_from_file_location("exp05_perception", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp05_perception"] = module
    spec.loader.exec_module(module)
    return module


exp05 = _load_exp05()
T_TIME_UNITS = exp05.T_TIME_UNITS
DELTA_T_OVERRIDE = exp05.DELTA_T_OVERRIDE
N_SLICES = exp05.N_SLICES


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Bundle:
    """Superset of exp05.ScenarioData: adds `trace` (the underlying
    ContinuousTrace), which exp05.run_scenario discards after building
    `discrete`. Needed here to re-discretize the SAME twin run under a
    second zoning without re-simulating (the whole point of this
    comparison -- isolate the zoning variable, not twin stochasticity)."""
    run_id: int
    trace: ContinuousTrace
    discrete: DiscreteTrace  # heuristic-zoned
    discretize_seed: np.random.SeedSequence  # for reproducing the SAME cyber-evidence draws under a second zoning
    x_dict: dict[str, torch.Tensor]
    globals_: torch.Tensor
    labels: dict[str, torch.Tensor]
    mask: dict[str, torch.Tensor]
    violations: list


def run_scenario_with_trace(
    run_id: int, ag_physical, grid_cfg, twin_cfg, zones, asset_graph, overlay, nonempty_types,
    seed: np.random.SeedSequence, *, dyn_config: DynamicFeatureConfig,
) -> Bundle:
    tw_config = TwinConfig(
        grid=grid_cfg,
        attacker=AttackerConfig(delay_law=DelayLaw.EXPONENTIAL),
        dispatch_period_time_units=float(twin_cfg["control_centre"]["dispatch_period_time_units"]),
        comms_latency_time_units=float(twin_cfg["comms"]["latency_time_units"]),
        horizon_time_units=float(T_TIME_UNITS),
    )
    trace = TwinRunner(ag_physical, tw_config, seed).run()
    violations = validate_trace(trace, ag_physical)
    discretize_seed = seed.spawn(1)[0]
    discrete = discretize(
        trace, ag_physical, DELTA_T_OVERRIDE, N_SLICES,
        np.random.default_rng(discretize_seed), P_POS, P_NEG, zones=zones,
    )
    dyn = build_dynamic_features(
        trace, DELTA_T_OVERRIDE, N_SLICES, asset_graph, overlay, grid_cfg,
        dispatch_period_time_units=tw_config.dispatch_period_time_units,
        config=dyn_config, rng=None,
    )
    static_x = {t: asset_graph.data[t].x for t in nonempty_types}
    x_dict = combine_static_dynamic(static_x, {t: dyn[t] for t in nonempty_types})
    globals_ = dyn["host"][:, 0, :]
    labels, mask = exp05.extract_labels(discrete)
    return Bundle(
        run_id=run_id, trace=trace, discrete=discrete, discretize_seed=discretize_seed,
        x_dict=x_dict, globals_=globals_, labels=labels, mask=mask, violations=violations,
    )


def generate_split(label: str, seeds: list[np.random.SeedSequence], **kwargs) -> list[Bundle]:
    print(f"  {label}: {len(seeds)} scenarios ...", flush=True)
    out = []
    for i, seed in enumerate(seeds):
        out.append(run_scenario_with_trace(i, seed=seed, **kwargs))
        print(f"    {label} {i + 1}/{len(seeds)} done", flush=True)
    return out


def pre_attack_slice_count(discrete: DiscreteTrace) -> int:
    """First slice index (0-based) where any attack-step node's ground_truth
    is 1; activation is monotonic/persistent (self-loop), so this is exactly
    the pre-attack prefix length. Falls back to 1 slice if the very first
    slice is already non-nominal (rare: a fast root completing before the
    first slice), rather than an empty window."""
    for i, r in enumerate(discrete.records):
        if any(r.ground_truth.values()):
            return max(i, 1)
    return len(discrete.records)


def bus_embedding(model: PerceptionEncoder, bundle: Bundle, edge_index_dict, n_pre_attack: int) -> np.ndarray:
    """Mean-pooled [n_bus, hidden] embedding over the pre-attack slice
    prefix, from the trained model's own SpatialEncoder submodule directly
    (bypassing Readout/TCN/heads -- those are irrelevant to what the GNN
    learned about BUS structure specifically)."""
    x_dict_1 = {t: x[:n_pre_attack][None] for t, x in bundle.x_dict.items()}  # [1, S, N_t, F_t]
    with torch.no_grad():
        flat, num_nodes, B, S = flatten_for_spatial(x_dict_1)
        rep_edges = replicate_static_graph(edge_index_dict, num_nodes, B * S)
        h_flat = model.spatial(flat, rep_edges)
        h = unflatten_from_spatial(h_flat, num_nodes, B, S)
    return h["bus"][0].mean(dim=0).numpy()  # [n_bus, hidden]


def build_gnn_zonemap(cluster_labels: np.ndarray, bus_ids: list[int]) -> ZoneMap:
    """A ZoneMap built from KMeans cluster membership instead of the
    dominance-share sensitivity rule. `.bus` (each zone's single anchor
    bus) is never read by classify() -- confirmed by direct code read --
    so an arbitrary deterministic representative (min bus id) is fine.
    Every bus is assigned (KMeans has no concept of "unassigned"), unlike
    ZoneMap's own dominance-threshold construction -- see module docstring."""
    zones = []
    for cluster_id in sorted(set(cluster_labels.tolist())):
        buses = frozenset(b for b, c in zip(bus_ids, cluster_labels) if c == cluster_id)
        zones.append(DerZone(der_id=f"gnn_cluster_{cluster_id}", bus=min(buses), buses=buses))
    return ZoneMap(
        zones=tuple(zones), unassigned_buses=frozenset(),
        sensitivity_dv_dp={}, dominance_tau=float("nan"), delta_p_mw=float("nan"),
        rule=f"gnn_kmeans_k{K_CLUSTERS}_bus_embedding_pre_attack_mean",
    )


def build_engine(ag) -> DBNInference:
    interface = _interface_nodes(ag)
    return DBNInference(
        ag, InferenceConfig(clustering=fully_factorized_clustering(interface), m=M, p_pos=P_POS, p_neg=P_NEG,
                             delta_t_override=DELTA_T_OVERRIDE),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text())
    twin_cfg = yaml.safe_load(TWIN_CONFIG_PATH.read_text())
    perception_cfg = yaml.safe_load(PERCEPTION_CONFIG_PATH.read_text())
    seed = config["seed"]
    set_all_seeds(seed)

    RESULTS_DIR.mkdir(exist_ok=True)
    SUMMARIES_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = git_sha(REPO_ROOT)
    tag = "smoke_" if args.smoke else ""

    splits_cfg = perception_cfg["smoke"] if args.smoke else perception_cfg["splits"]
    n_train = int(splits_cfg["n_train"])
    n_val = int(splits_cfg["n_val"])
    n_calib = int(splits_cfg["n_calib"])
    n_test = int(splits_cfg["n_test"])
    train_cfg = dict(perception_cfg["training"])
    if args.smoke:
        train_cfg["n_epochs"] = int(perception_cfg["smoke"]["n_epochs"])
    if not args.smoke:
        assert n_test >= 30, "task requires >= 30 test scenarios"

    grid_cfg = GridConfig(
        network=twin_cfg["grid"]["network"], n_der=int(twin_cfg["grid"]["n_der"]),
        p_mw_levels=list(twin_cfg["grid"]["p_mw_levels"]),
        nominal_level_index=int(twin_cfg["grid"]["nominal_level_index"]),
        nonconvergence_is_unstable=bool(twin_cfg["grid"]["nonconvergence_is_unstable"]),
    )
    reaction_mode = twin_cfg["experiment"]["reaction_mode"]
    ag_physical = exp05.build_attack_graph(reaction_mode=reaction_mode, physical_evidence=True)

    # === stage 0: asset graph + heuristic zones ==============================
    print("stage 0: asset graph + heuristic zones ...", flush=True)
    grid = GridModel(grid_cfg)
    overlay = exp05.build_overlay(grid)
    zones_heuristic = exp05.build_zones(
        grid_cfg, float(twin_cfg["physical"]["delta_p_mw"]), float(twin_cfg["physical"]["dominance_tau"])
    )
    print(f"  heuristic zones: {[(z.der_id, sorted(z.buses)) for z in zones_heuristic.zones]}")
    print(f"  heuristic unassigned_buses: {sorted(zones_heuristic.unassigned_buses)}")

    probe_config = TwinConfig(grid=grid_cfg, horizon_time_units=float(T_TIME_UNITS))
    probe_trace = TwinRunner(ag_physical, probe_config, np.random.SeedSequence(seed)).run()
    obs = observed_endpoints(probe_trace.messages)
    asset_graph = build_asset_graph(grid, overlay, observed_endpoints_set=obs)
    nt, et = nonempty_metadata(asset_graph.data)
    edge_index_dict = {e: asset_graph.data[e].edge_index for e in et}
    bus_ids = [None] * asset_graph.counts["bus"]
    for bus_str, row in asset_graph.node_index["bus"].items():
        bus_ids[row] = int(bus_str)
    assert all(b is not None for b in bus_ids), "asset_graph.node_index['bus'] did not cover every row"

    # === stage 1: data + GNN training (GNN arm only) =========================
    print("\nstage 1: data generation + GNN training ...", flush=True)
    root = np.random.SeedSequence(seed)
    train_root, val_root, calib_root, test_root, model_root = root.spawn(5)
    train_seeds = train_root.spawn(n_train)
    val_seeds = val_root.spawn(n_val)
    calib_seeds = calib_root.spawn(n_calib)
    test_seeds = test_root.spawn(n_test)
    torch_seed = int(model_root.generate_state(1)[0])

    common_kwargs = dict(
        ag_physical=ag_physical, grid_cfg=grid_cfg, twin_cfg=twin_cfg, zones=zones_heuristic,
        asset_graph=asset_graph, overlay=overlay, nonempty_types=nt,
        dyn_config=DynamicFeatureConfig(se_noise_sigma=0.0, observability="voltage_only"),
    )
    train_scenarios = generate_split("train", train_seeds, **common_kwargs)
    val_scenarios = generate_split("val", val_seeds, **common_kwargs)
    test_scenarios = generate_split("test", test_seeds, **common_kwargs)

    torch.manual_seed(torch_seed)
    encoder_cfg = EncoderConfig(
        node_types=tuple(nt), edge_types=tuple(et), encoder_type="gnn",
        hidden=int(perception_cfg["architecture"]["hidden_dim"]),
        heads=int(perception_cfg["architecture"]["gnn_heads"]),
        n_gnn_layers=int(perception_cfg["architecture"]["n_gnn_layers"]),
        tcn_kernel_size=int(perception_cfg["architecture"]["tcn_kernel_size"]),
        tcn_dilations=tuple(perception_cfg["architecture"]["tcn_dilations"]),
        dropout=float(perception_cfg["architecture"]["tcn_dropout"]),
    )
    model = PerceptionEncoder(encoder_cfg)
    model, training_rows = exp05.train_model(
        model, train_scenarios, val_scenarios, edge_index_dict, train_cfg, seed=torch_seed
    )
    model.eval()
    training_df = pd.DataFrame(training_rows)
    training_df["git_sha"] = sha
    training_path = RESULTS_DIR / f"exp12_{tag}training_{timestamp}.csv"
    training_df.to_csv(training_path, index=False)
    print(f"  wrote {training_path}")

    # === stage 2: extract + cluster bus embeddings ============================
    print("\nstage 2: bus-embedding extraction + clustering ...", flush=True)
    embeddings = []
    for s in test_scenarios:
        n_pre = pre_attack_slice_count(s.discrete)
        embeddings.append(bus_embedding(model, s, edge_index_dict, n_pre))
    bus_matrix = np.mean(np.stack(embeddings, axis=0), axis=0)  # [n_bus, hidden]
    assert bus_matrix.shape == (len(bus_ids), int(perception_cfg["architecture"]["hidden_dim"]))
    assert np.isfinite(bus_matrix).all(), "non-finite bus embedding"

    kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=seed, n_init=10)
    labels_1 = kmeans.fit_predict(bus_matrix)
    labels_2 = KMeans(n_clusters=K_CLUSTERS, random_state=seed, n_init=10).fit_predict(bus_matrix)
    kmeans_deterministic = np.array_equal(labels_1, labels_2)

    zones_gnn = build_gnn_zonemap(labels_1, bus_ids)
    print(f"  GNN clusters: {[(z.der_id, sorted(z.buses)) for z in zones_gnn.zones]}")

    heuristic_bus_label = {}
    for z in zones_heuristic.zones:
        for b in z.buses:
            heuristic_bus_label[b] = z.der_id
    for b in zones_heuristic.unassigned_buses:
        heuristic_bus_label[b] = "unassigned"
    heuristic_labels_ordered = [heuristic_bus_label.get(b, "unassigned") for b in bus_ids]
    ari = adjusted_rand_score(heuristic_labels_ordered, labels_1.tolist())
    print(f"  adjusted_rand_score(heuristic, gnn) = {ari:.4f}")
    print(f"  heuristic coverage: {33 - len(zones_heuristic.unassigned_buses)}/33 buses; GNN coverage: {len(bus_ids)}/33 buses")

    cluster_df = pd.DataFrame({"bus": bus_ids, "heuristic_label": heuristic_labels_ordered, "gnn_cluster": labels_1})
    cluster_df["git_sha"] = sha
    cluster_path = RESULTS_DIR / f"exp12_{tag}cluster_assignment_{timestamp}.csv"
    cluster_df.to_csv(cluster_path, index=False)
    print(f"  wrote {cluster_path}")

    # === stage 3: observable-level KL (PHYS_LOCAL_DER / PHYS_WIDE_AREA) ======
    print("\nstage 3: observable-level KL (heuristic as P, GNN as Q) ...", flush=True)
    per_slice_heuristic = {t: [[] for _ in range(N_SLICES)] for t in (PHYS_LOCAL_DER, PHYS_WIDE_AREA)}
    per_slice_gnn = {t: [[] for _ in range(N_SLICES)] for t in (PHYS_LOCAL_DER, PHYS_WIDE_AREA)}
    trace_identity_ok = True
    for s in test_scenarios:
        discrete_gnn = discretize(
            s.trace, ag_physical, DELTA_T_OVERRIDE, N_SLICES,
            np.random.default_rng(s.discretize_seed), P_POS, P_NEG, zones=zones_gnn,
        )
        if discrete_gnn.records[0].grid is not s.discrete.records[0].grid:
            trace_identity_ok = False  # same GridState objects => same ContinuousTrace, never re-simulated
        for i in range(N_SLICES):
            rh = s.discrete.records[i]
            rg = discrete_gnn.records[i]
            for target, obs_fn in ((PHYS_LOCAL_DER, observe_local), (PHYS_WIDE_AREA, observe_wide)):
                vh = obs_fn(rh.physical) if rh.physical is not None else None
                vg = obs_fn(rg.physical) if rg.physical is not None else None
                if vh is not None:
                    per_slice_heuristic[target][i].append(float(vh))
                if vg is not None:
                    per_slice_gnn[target][i].append(float(vg))
        print(f"  scenario {s.run_id + 1}/{len(test_scenarios)} done", flush=True)

    observable_kl_rows = []
    for target in (PHYS_LOCAL_DER, PHYS_WIDE_AREA):
        kl_by_slice = {}
        for i in range(N_SLICES):
            h_vals, g_vals = per_slice_heuristic[target][i], per_slice_gnn[target][i]
            if h_vals and g_vals:
                kl_by_slice[i + 1] = binary_kl(float(np.mean(h_vals)), float(np.mean(g_vals)))
        if kl_by_slice:
            value, argmax_slice = m_kl(kl_by_slice)
            observable_kl_rows.append({"target": target, "m_kl": value, "argmax_slice": argmax_slice})
            print(f"  {target:<18} M_KL(heuristic||gnn) = {value:.6f} @ slice {argmax_slice}")

    observable_kl_df = pd.DataFrame(observable_kl_rows)
    observable_kl_df["git_sha"] = sha
    observable_kl_path = RESULTS_DIR / f"exp12_{tag}observable_kl_{timestamp}.csv"
    observable_kl_df.to_csv(observable_kl_path, index=False)
    print(f"  wrote {observable_kl_path}")

    # === stage 4: downstream posterior KL (UnstablePS) =======================
    print("\nstage 4: downstream DBN posterior KL (heuristic as P, GNN as Q) ...", flush=True)
    engine = build_engine(ag_physical)
    posterior_kl_rows = []
    for s in test_scenarios:
        discrete_gnn = discretize(
            s.trace, ag_physical, DELTA_T_OVERRIDE, N_SLICES,
            np.random.default_rng(s.discretize_seed), P_POS, P_NEG, zones=zones_gnn,
        )
        observed = list(CYBER_ANALYTICS) + [PHYS_LOCAL_DER, PHYS_WIDE_AREA]
        traj_h = engine.run(s.discrete.evidence_stream(observed), N_SLICES)
        traj_g = engine.run(discrete_gnn.evidence_stream(observed), N_SLICES)
        kl_by_slice = {
            i + 1: binary_kl(traj_h.marginals[i]["UnstablePS"], traj_g.marginals[i]["UnstablePS"])
            for i in range(N_SLICES)
        }
        value, argmax_slice = m_kl(kl_by_slice)
        posterior_kl_rows.append({"scenario": s.run_id, "m_kl": value, "argmax_slice": argmax_slice})
        print(f"  scenario {s.run_id + 1}/{len(test_scenarios)}: M_KL(UnstablePS) = {value:.6f}", flush=True)

    posterior_kl_df = pd.DataFrame(posterior_kl_rows)
    posterior_kl_df["git_sha"] = sha
    posterior_kl_path = RESULTS_DIR / f"exp12_{tag}posterior_kl_{timestamp}.csv"
    posterior_kl_df.to_csv(posterior_kl_path, index=False)
    mean_posterior_kl = float(posterior_kl_df["m_kl"].mean())
    print(f"  wrote {posterior_kl_path}")
    print(f"  mean M_KL(UnstablePS) across {len(test_scenarios)} scenarios = {mean_posterior_kl:.6f}")

    summary_rows = observable_kl_rows + [{"target": "UnstablePS_posterior", "m_kl": mean_posterior_kl, "argmax_slice": None}]
    summary_df = pd.DataFrame(summary_rows)
    summary_df["adjusted_rand_score"] = ari
    summary_df["heuristic_coverage_buses"] = 33 - len(zones_heuristic.unassigned_buses)
    summary_df["gnn_coverage_buses"] = len(bus_ids)
    summary_df["git_sha"] = sha
    summary_path = SUMMARIES_DIR / f"exp12_summary_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  wrote {summary_path}")

    # === validation gate =======================================================
    print("\n=== VALIDATION GATE ===")
    failures: list[str] = []

    print(f"(a) bus embedding shape [{len(bus_ids)},{bus_matrix.shape[1]}], finite ... PASS")
    print(f"(b) KMeans deterministic given fixed seed ... {'PASS' if kmeans_deterministic else 'FAIL'}")
    if not kmeans_deterministic:
        failures.append("KMeans not deterministic across repeated calls with the same seed")
    print(f"(c) GNN partition covers all {len(bus_ids)}/33 buses; heuristic unassigned={len(zones_heuristic.unassigned_buses)} (reported, not asserted)")
    print(f"(d) both zonings discretized the IDENTICAL ContinuousTrace (never re-simulated) ... {'PASS' if trace_identity_ok else 'FAIL'}")
    if not trace_identity_ok:
        failures.append("gnn-zoned discretize() did not reuse the identical ContinuousTrace")
    all_finite = np.isfinite(observable_kl_df["m_kl"]).all() and np.isfinite(posterior_kl_df["m_kl"]).all()
    print(f"(e) every KL value finite ... {'PASS' if all_finite else 'FAIL'}")
    if not all_finite:
        failures.append("non-finite KL value found")
    n_test_ok = args.smoke or n_test >= 30
    print(f"(f) n_test >= 30 ({n_test}) {'[--smoke, not gated]' if args.smoke else ''} {'PASS' if n_test_ok else 'FAIL'}")
    if not n_test_ok:
        failures.append(f"n_test={n_test} < 30")

    print(
        "\nNOTE: the KL/ARI numbers above (how much the GNN-derived clustering "
        "diverges from the heuristic ZoneMap) are REPORTED, not gated (CLAUDE.md "
        "rule 3). A large divergence is a valid, reportable finding, not a "
        "failure. See LAB_NOTEBOOK.md H1/H2/H3."
    )

    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
