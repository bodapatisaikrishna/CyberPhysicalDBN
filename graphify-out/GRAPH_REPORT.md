# Graph Report - .  (2026-08-10)

## Corpus Check
- 105 files · ~155,024 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1772 nodes · 5170 edges · 86 communities (67 shown, 19 thin omitted)
- Extraction: 83% EXTRACTED · 17% INFERRED · 0% AMBIGUOUS · INFERRED: 860 edges (avg confidence: 0.52)
- Token cost: 1,458,976 input · 0 output

## Community Hubs (Navigation)
- Twin Message Sanitization & Attacker Events
- Twin-to-DBN Scenario Bundling
- Physical Zone Classification (Consequence)
- Sherlock Feature/Label Extraction
- DBN Belief State & Cluster Decoding
- Amortized TTC Parameterization Model
- LSTM Autoencoder Baseline
- Attacker Config & TTC Timing
- Lead-Time Detection Metrics
- DBN CPT Parameterization (Table 1-3)
- exp04: Closed-Loop C1 Experiment
- exp05: Perception Training Pipeline
- Perception Encoder Architecture (GNN/TCN)
- Attack Graph Construction (graph.py)
- Attack-Graph Family Generation
- exp03: Twin Open-Loop Experiment
- Baseline Common Utilities
- FF Clustering & Soft-Evidence Wiring
- GNN Classifier Baseline Architecture
- GBM Baseline
- exp01: Paper-Reproduction Experiment
- Stack Verification Script
- RL Attacker Scenario Generation
- Rule-Based Baseline
- exp10: m-Sweep Experiment
- Perception Asset Graph Builder
- Temperature Calibration Tests
- AG-to-2TBN Compiler
- Forward-Sampling Tests
- GNN Classifier Forward Pass
- Forward (Generative) Sampling
- exp07: Sherlock Experiment
- Validation-Gate Summary Script
- exp02: Latched-Reaction KL Experiment
- RL Attacker Env Tests
- exp09: Adversarial RL (Claim C3)
- exp06: Baselines Comparison Experiment
- Causal TCN Building Blocks
- exp05 Reliability Diagrams
- ECE Calibration Metric
- KL Divergence Metric (Eq. 2)
- DBN Parameterization Tests (Fixtures)
- IED Dynamic Features
- Twin Attacker Process (SimPy)
- Perception Encoder Causality Tests
- Architecture Overview: DBN Core & Clustering Strategies
- Architecture Overview: Perception & Physical Loop
- exp08: Transfer C2 Experiment
- Brier Score & Skill Score
- RL Attacker Action Decoding Tests
- Project Thesis & Absolute Rules
- Claim C3 & Baselines Config
- Reproducibility Verification Script
- Slice-Boundary Timing Tests
- LAB_NOTEBOOK: exp01-03 Decisions
- Summary-Table Consolidation Script
- Run-Level Bootstrap Calibration Report
- Asset-Graph Electrical/Cyber Coupling Tests
- Latched-Reaction CPT Tests
- Claim C2 & Uniformization Equation
- Sherlock Download Script
- Attack-Graph Family Builder Internals
- Reliability Diagram Plotting Tests
- LAB_NOTEBOOK: Session 10 Findings
- Sherlock Dataset & IPAL Format
- Reaction-Node Memorylessness Tests
- Scenario 1 KL Figure & DBN Nodes
- AND/OR Gate CPT Tests
- exp01/exp03 Timebase-Pinning Tests
- FF/EX Degenerate-Step Sanity Check
- Clustering Validation Helper
- GNN-vs-MLP Figure & Source Run
- attack_graph Package Init
- baselines Package Init
- dbn Package Init
- eval Package Init
- Project Package Init
- parameterization Package Init
- perception Package Init
- twin Package Init
- DBN Evidence-Stream Integration Test
- Repo Root Marker
- Scenario 2 KL Figure (Fig. 8)
- Scenario 2 EX Probs Figure (Fig. 7b)
- Twin-vs-Scripted Comparison Figure
- Open-vs-Closed Posterior Figure (exp04)

## God Nodes (most connected - your core abstractions)
1. `GridConfig` - 103 edges
2. `GridModel` - 94 edges
3. `TwinRunner` - 83 edges
4. `DBNInference` - 81 edges
5. `TwinConfig` - 77 edges
6. `InferenceConfig` - 74 edges
7. `ContinuousTrace` - 59 edges
8. `build_attack_graph()` - 57 edges
9. `AttackerConfig` - 54 edges
10. `main()` - 51 edges

## Surprising Connections (you probably didn't know these)
- `ScenarioData` --uses--> `DBNInference`  [INFERRED]
  experiments/exp05_perception.py → src/dbn/inference.py
- `ScenarioData` --uses--> `InferenceConfig`  [INFERRED]
  experiments/exp05_perception.py → src/dbn/inference.py
- `ScenarioData` --uses--> `SoftEvidenceConfig`  [INFERRED]
  experiments/exp05_perception.py → src/dbn/soft_evidence.py
- `ScenarioData` --uses--> `DetectionResult`  [INFERRED]
  experiments/exp05_perception.py → src/eval/lead_time.py
- `ScenarioData` --uses--> `LeadTimeSummary`  [INFERRED]
  experiments/exp05_perception.py → src/eval/lead_time.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Three Falsifiable Claims of the Project Thesis** — claude_project, claude_c1_closed_loop, claude_c2_learned_parameters, claude_c3_adversarial_robustness [EXTRACTED 1.00]
- **C1 Closed-Loop Physical-Evidence Investigation** — lab_notebook_exp04_closed_loop_c1, lab_notebook_exp05_perception, lab_notebook_physical_evidence_nodes, lab_notebook_m1_corrreact_gating, lab_notebook_m2_zero_precursor_window [INFERRED 0.85]
- **Session 10 Ablation Sweep, Reproducibility Check, and Audit** — lab_notebook_exp10_m_sweep, lab_notebook_exp11_perception_ablation, lab_notebook_reproducibility_check, lab_notebook_maintenance_debugging_pass [EXTRACTED 1.00]

## Communities (86 total, 19 thin omitted)

### Community 0 - "Twin Message Sanitization & Attacker Events"
Cohesion: 0.07
Nodes (57): Manipulation, One end of each message, from `viewpoint`'s perspective, never both. - Message…, sanitize_bus_log(), AttackEvent, EventKind, Enum, Scripted attacker executing the Figure-2 attack processes (CLAUDE.md layer…, One SimPy process per attack-graph node, gated on its preconditions. (+49 more)

### Community 1 - "Twin-to-DBN Scenario Bundling"
Cohesion: 0.07
Nodes (52): build_overlay(), run_scenario(), build_twin_transfer_scenarios(), Fresh twin scenarios reduced to the shared bus-voltage subspace. Label = any…, build_overlay(), bundle_from_trace(), Builds a `ScenarioBundle` from an ALREADY-EXISTING `ContinuousTrace` -- never…, Observability (+44 more)

### Community 2 - "Physical Zone Classification (Consequence)"
Cohesion: 0.05
Nodes (44): build_zones(), build_zones(), build_zone_map(), DerZone, DeviationClass, Enum, Physical deviation as a measured outcome (CLAUDE.md layer [0]/[4], claim C1).…, Derive DER-to-bus zones from sensitivity shares. share_d(b) = S[d][b] / sum_d'… (+36 more)

### Community 3 - "Sherlock Feature/Label Extraction"
Cohesion: 0.06
Nodes (45): `[S, n_bus, 4]` (`BUS_DYNAMIC_COLUMNS`) -> `[S, 2]`…, twin_bus_voltage_shared_subspace(), _aggregate(), build_global_features(), build_labels(), build_shared_subspace(), chronological_chunks(), component_attribute_values() (+37 more)

### Community 4 - "DBN Belief State & Cluster Decoding"
Cohesion: 0.05
Nodes (41): BeliefState, LikelihoodMode, _cluster_prior_node(), DiGraph, TabularCPD, Stable auxiliary-node name for a multi-member cluster's joint prior., Point mass at 'every interface node inactive'. Evidenced directly by the…, `soft`: target name -> calibrated q = P(target=1|telemetry) for this slice.… (+33 more)

### Community 5 - "Amortized TTC Parameterization Model"
Cohesion: 0.07
Nodes (40): `{mitre_technique: table3_ttc}`, derived from `_NODE_DEFS` (never hand-copied)…, technique_table3_ttc(), AmortizedTrainConfig, apply_ttc_predictions(), ContextNormalizer, fit_context_normalizer(), fit_ttc_amortized_model(), known_techniques() (+32 more)

### Community 6 - "LSTM Autoencoder Baseline"
Cohesion: 0.06
Nodes (37): build_causal_windows(), error_to_probability(), fit_recon_error_scaler(), LSTMAETrialConfig, LSTMAutoencoder, ndarray, Tensor, LSTM autoencoder, reconstruction-error anomaly detection (CLAUDE.md's… (+29 more)

### Community 7 - "Attacker Config & TTC Timing"
Cohesion: 0.07
Nodes (27): AttackerConfig, parametrize, With delay = TTC exactly, completion times are hand-checkable. UnsecCred = 3 x…, A step never un-completes -- the self-loop persistence assumption., Session 8 (claim C2): AttackerConfig.speed_multiplier /…, Regression: every prior experiment constructs AttackerConfig() bare -- both new…, defense_slowdown_multiplier == speed_multiplier -> effective_ttc == ttc…, Session 9 (claim C3): AttackerConfig.excluded_nodes, the mechanism… (+19 more)

### Community 8 - "Lead-Time Detection Metrics"
Cohesion: 0.08
Nodes (24): count_upward_crossings(), DetectionOutcome, DetectionResult, evaluate_run(), first_crossing(), first_instability(), Enum, Detection lead time (claim C1's headline metric). Written and tested BEFORE… (+16 more)

### Community 9 - "DBN CPT Parameterization (Table 1-3)"
Cohesion: 0.10
Nodes (34): analytic_error_rates(), attach_cpds(), build_analytic_cpt(), build_attack_step_cpt(), build_gate_cpt(), build_latch_cpt(), build_latched_reaction_cpt(), build_reaction_cpt() (+26 more)

### Community 10 - "exp04: Closed-Loop C1 Experiment"
Cohesion: 0.11
Nodes (28): build_engine(), characterize_sensors(), invariant_band(), lead_time_table(), main(), physical_only_series(), posterior_series(), DataFrame (+20 more)

### Community 11 - "exp05: Perception Training Pipeline"
Cohesion: 0.10
Nodes (33): batch_forward(), compute_base_rates(), compute_pos_weight(), generate_split(), main(), masked_bce_loss(), pool_target(), predict_scenarios() (+25 more)

### Community 12 - "Perception Encoder Architecture (GNN/TCN)"
Cohesion: 0.14
Nodes (24): CausalTCN, EncoderConfig, PerceptionEncoder, Heterogeneous GNN + causal temporal encoder (CLAUDE.md layer [1]). Pipeline…, Ablation encoder (Session 10, GNN-vs-MLP axis): per-node-type MLP with NO…, host + mean(IED) + mean(DER) + mean(bus) + max(bus) + globals -> one shared…, The full pipeline: `x_dict [B,S,N_t,F_t]` + static edges + globals -> `{target:…, Readout (+16 more)

### Community 13 - "Attack Graph Construction (graph.py)"
Cohesion: 0.09
Nodes (25): Any, Fraction, ObservableKind, ReactionMode, _add_physical_evidence_nodes(), _analytic(), _apply_latched_reactions(), _attack_step() (+17 more)

### Community 14 - "Attack-Graph Family Generation"
Cohesion: 0.13
Nodes (24): family_graph_rows(), FamilyGeneratorConfig, FamilyGraph, FamilyGraphSpec, generate_family(), DataFrame, SeedSequence, `config.n_graphs` graphs, split `n_train`/`n_val`/`n_test` in that fixed order… (+16 more)

### Community 15 - "exp03: Twin Open-Loop Experiment"
Cohesion: 0.08
Nodes (26): first_firing_slice(), grid_calibration_sweep(), main(), DataFrame, Twin-driven evidence into the Session-2 DBN, vs the paper's scripted Scenario…, exp01's Scenario 2 evidence: dense, 0 before the raising time, 1 at and after., Stage 0: re-measure the dispatch ladder so its values have logged provenance., run_dbn() (+18 more)

### Community 16 - "Baseline Common Utilities"
Cohesion: 0.11
Nodes (19): aggregate_node_type(), BaselineResult, flatten_engineered_features(), hyperparameter_search(), Tensor, Shared scaffolding for external ML baselines (CLAUDE.md minimum-viable-…, Runs `train_and_score_fn(config, rng)` for EVERY entry in `configs` and returns…, One baseline's per-slice score trajectory for one scenario. `score` is aligned… (+11 more)

### Community 17 - "FF Clustering & Soft-Evidence Wiring"
Cohesion: 0.13
Nodes (14): fully_factorized_clustering(), FF: every interface node in its own cluster (Cerotti et al. Sec. IV)., Which analytic nodes receive learned virtual evidence, and how their q is…, SoftEvidenceConfig, Session 5: DBNInference wired to src/dbn/soft_evidence.py. These integration…, A stored reference trajectory computed with soft_evidence=None must be…, A degenerate q (near 0 or 1) fused via soft evidence must reproduce what hard…, exact_clustering forces the joint=True multi-member-cluster query path… (+6 more)

### Community 18 - "GNN Classifier Baseline Architecture"
Cohesion: 0.15
Nodes (18): GNNBaselineConfig, GNNClassifier, HeteroSpatialLayer, spatial (`HeteroSpatialLayer` x n_layers, with residual from layer 1 on) ->…, One `HeteroConv`-wrapped layer (GAT or SAGE per edge type) + GELU + per-type…, _edge_index_dict(), _model(), parametrize (+10 more)

### Community 19 - "GBM Baseline"
Cohesion: 0.13
Nodes (20): HistGradientBoostingClassifier, Protocol, build_flat_table(), GBMTrialConfig, ndarray, Gradient-boosted trees on engineered features (CLAUDE.md's `src/baselines/`:…, Duck-typed so this module has no import-time dependency on…, One row per `(scenario, slice)`. `y` is `grid_unstable` (measured, never… (+12 more)

### Community 20 - "exp01: Paper-Reproduction Experiment"
Cohesion: 0.11
Nodes (28): compute_kl_trajectories(), main(), plot_kl(), plot_probabilities(), DataFrame, Path, Reproduce Cerotti et al. Sec. IV, Scenarios 1 and 2. Runs FF and EX clustered…, No evidence at all (Cerotti et al. Figs. 5, 6, 9). (+20 more)

### Community 21 - "Stack Verification Script"
Cohesion: 0.09
Nodes (26): Graph, HeteroData, check_pandapower(), check_pgmpy(), check_simpy(), check_torch_geometric(), check_versions(), main() (+18 more)

### Community 22 - "RL Attacker Scenario Generation"
Cohesion: 0.17
Nodes (25): generate_scenario(), generate_split(), SeedSequence, ScenarioBundle, train_policy(), KnowledgeLevel, PPO, DBNInference (+17 more)

### Community 23 - "Rule-Based Baseline"
Cohesion: 0.15
Nodes (12): ndarray, Signature-based IDS proxy (CLAUDE.md's `src/baselines/`: "rule-based").…, `score[s] = |{name in observable_names : name fired (==1) at least once in the…, RuleConfig, score_trajectory(), Tests for src/baselines/rule_based.py., Non-vacuous: at least one all-zero and one all-firing case both stay within…, A cyber analytic that is DENSE (1 at every slice from its trigger onward) must… (+4 more)

### Community 24 - "exp10: m-Sweep Experiment"
Cohesion: 0.13
Nodes (18): run_config(), main(), Discretization m sweep (Cerotti et al. Sec. IV-B; CLAUDE.md's own "Reference…, run_config(), set_all_seeds(), exact_clustering(), EX: a single cluster containing every interface node (Sec. IV)., Trajectory (+10 more)

### Community 25 - "Perception Asset Graph Builder"
Cohesion: 0.11
Nodes (13): build_asset_graph(), nonempty_metadata(), Build the static asset graph for the twin's case33bw feeder. Thin wrapper…, Metadata restricted to node types with >= 1 row. HGTConv raises `IndexError` on…, obs(), _overlay(), fixture, The positive case: ControlCentre/DER_17/DER_32 all genuinely appear in a real… (+5 more)

### Community 26 - "Temperature Calibration Tests"
Cohesion: 0.14
Nodes (11): fit_temperature_for_target(), `(temperature, nll_before, nll_after)`. `T = exp(log_T)` for positivity by…, Tests for src/perception/calibration.py. The pass-through test is the one that…, A tiny, perfectly-separable dataset (n=4, extreme confidence) is exactly the…, Extremely overconfident-in-the-WRONG-direction data should push T toward T_MAX…, A monotone rescaling cannot change AP -- if it moves at all, that is a bug, not…, overconfidence=1.0 means the logits ARE the true generating logits -- the…, _synthetic() (+3 more)

### Community 27 - "AG-to-2TBN Compiler"
Cohesion: 0.11
Nodes (12): compile_to_2tbn(), DiGraph, DiscreteBayesianNetwork, Attack graph -> 2TBN compiler. Implements the AG-to-DBN translation of Cerotti…, Compile the attack graph into a 2TBN structure. No CPDs are attached. Edge…, ag(), Cross-checks of DBNInference against pgmpy VariableElimination run directly.…, No self-loop and no anterior-layer copy for analytic nodes. (+4 more)

### Community 28 - "Forward-Sampling Tests"
Cohesion: 0.14
Nodes (17): forward_sample_evidence_stream(), Analytic 0/1 emissions from ground truth, the discrete analog of…, chain_ag(), _finalize(), gated_ag(), _node(), DiGraph, fixture (+9 more)

### Community 29 - "GNN Classifier Forward Pass"
Cohesion: 0.15
Nodes (12): Tensor, GAT/GraphSAGE end-to-end binary classifier over the asset graph (CLAUDE.md's…, flatten_for_spatial(), `[B,S,N_t,F_t]` -> `[B*S*N_t, F_t]`, folding batch+slice into the node axis so…, Block-diagonal replication of a topology identical across every (batch, slice)…, replicate_static_graph(), unflatten_from_spatial(), _edge_index_dict() (+4 more)

### Community 30 - "Forward (Generative) Sampling"
Cohesion: 0.15
Nodes (14): forward_sample_trajectory(), _parent_value(), DiGraph, Forward (generative) sampling of a 2TBN attack-graph trajectory (CLAUDE.md…, Discrete analog of `src.twin.runner.validate_trace`: every violation found, as…, Topological order over precondition edges, self-loops excluded (a self-loop is…, The value a parent contributes to `node`'s activation rule this slice: ANTERIOR…, Per-slice 0/1 ground truth for every non-analytic node (`attack_step`,… (+6 more)

### Community 31 - "exp07: Sherlock Experiment"
Cohesion: 0.20
Nodes (18): check_uniform_cadence(), Chunk, eval_transfer_model(), locate_sherlock_files(), main(), pos_weight_for(), Path, Tensor (+10 more)

### Community 32 - "Validation-Gate Summary Script"
Cohesion: 0.15
Nodes (13): main(), Validation-gate summary for the attack graph, 2TBN compile and CPT generation.…, Node attributes left None because the source paper does not supply them.…, undetermined_fields(), compute_delta_t(), compute_ps(), Discretization step size (Cerotti et al. Eq. 3, Sec. III-E). delta_t = 1 / ( m…, Per-step attack success probability (Cerotti et al. Eq. 3, Sec. III-E). p_s =… (+5 more)

### Community 33 - "exp02: Latched-Reaction KL Experiment"
Cohesion: 0.15
Nodes (13): main(), Scenario 1 KL(EX||FF) under one-shot latched reactions. Tests the hypothesis…, run(), Scenario 1 KL(EX||FF) Divergence Plot, Scenario 2 KL(EX||FF) Divergence Figure (cf. Fig. 8), m_kl(), measure_step(), Accuracy and resource-usage metrics (Cerotti et al. Eq. 2, Eq. 4, Eq. 5, Sec.… (+5 more)

### Community 34 - "RL Attacker Env Tests"
Cohesion: 0.15
Nodes (5): _make_env(), Each episode's detect_term must equal a fresh, independent DBNInference.run()…, Selecting zero roots can never reach the goal -- reward should never contain a…, TestKnowledgeLevels, TestSpaces

### Community 35 - "exp09: Adversarial RL (Claim C3)"
Cohesion: 0.18
Nodes (17): main(), ndarray, Path, Adversarial RL attacker vs. detectors: testing claim C3 (CLAUDE.md layer…, Groups Monitor's per-episode reward log into n_steps-sized rollout- batch…, reward_curve_rows(), sample_configs(), score_dbn() (+9 more)

### Community 36 - "exp06: Baselines Comparison Experiment"
Cohesion: 0.23
Nodes (16): generate_scenario(), generate_split(), _load_exp05(), main(), ndarray, SeedSequence, External ML baselines vs. the proposed system (CLAUDE.md's minimum-viable-…, Import experiments/exp05_perception.py by path -- the pattern already… (+8 more)

### Community 37 - "Causal TCN Building Blocks"
Cohesion: 0.14
Nodes (4): CausalConv1d, Tensor, Left-pad-only causal 1D convolution: output[..., t] depends only on input[...,…, TCNBlock

### Community 38 - "exp05 Reliability Diagrams"
Cohesion: 0.19
Nodes (13): Reliability Diagram: CommandCoherence (exp05), Reliability Diagram: MeasureCoherence (exp05), Reliability Diagram: PhysLocalDER (exp05), Reliability Diagram: PhysWideArea Scenario (exp05), CalibrationReport, _mce(), Calibration of P(UnstablePS) against MEASURED grid instability (claim C1).…, ReliabilityBin (+5 more)

### Community 39 - "ECE Calibration Metric"
Cohesion: 0.20
Nodes (8): _bin_edges(), expected_calibration_error(), ndarray, ECE = sum_bins (count_b / n) * |empirical_rate_b - mean_predicted_b|. Empty…, Strategy, 4 samples, 2 uniform bins [0,0.5) and [0.5,1.0]. bin0: y_prob in [0,0.5) ->…, A probability of exactly 1.0 must land in the top bin, not be lost., TestExpectedCalibrationError

### Community 40 - "KL Divergence Metric (Eq. 2)"
Cohesion: 0.20
Nodes (7): binary_kl(), D_KL(Bernoulli(p1) || Bernoulli(q1)) (Cerotti et al. Eq. 2, specialized).…, Eq. 2 specialized to a Bernoulli pair. The whole KL half of this project's…, KL is not a metric; Eq. 4 fixes P=EX and Q=FF, so order matters. Guards the…, 0 log(0/q) = 0 by convention, so p=0 needs no clipping., P>0 where Q=0 is formally +inf; clipped so a run can still report., TestBinaryKL

### Community 41 - "DBN Parameterization Tests (Fixtures)"
Cohesion: 0.15
Nodes (8): fixture, UnsecCred/ModAuthProc -> CredAccess (AND) -> MITM, plus FileAccess. This is the…, P(MITM=1) = P(CredAccess=1) * (m + (1-m) * p_s_MITM), where P(CredAccess=1) =…, Same structure, with FileAccess=1 evidence, checked against a directly-built…, UnsecCred1 -> UnsecCred2: two persistent attack steps, pure inter-slice. Both…, TestBaselineChain, TestCredAccessMitmStructure, _ve_marginal()

### Community 42 - "IED Dynamic Features"
Cohesion: 0.35
Nodes (7): ied_dynamic_features(), _nearest_ladder_index(), `[S, n_ied, 10]`, columns = `IED_DYNAMIC_COLUMNS`. Every message-derived…, One message, from exactly one viewpoint's side. NO `origin`, NO `tampered_by`…, SanitizedMessage, The genuine CommandCoherence signature: a legitimate climb visits every…, TestZohAndStaleness

### Community 43 - "Twin Attacker Process (SimPy)"
Cohesion: 0.23
Nodes (4): DiGraph, Environment, CredAccess (AND) and UnstablePS (OR): resolve within the slice, no TTC., CorrReact / WrongLogicExec: one Bernoulli draw at the fixed success prob. In…

### Community 44 - "Perception Encoder Causality Tests"
Cohesion: 0.25
Nodes (5): _model(), _random_inputs(), Every one of the 4 heads must independently respect causality -- a bug isolated…, Perturbing raw input features at slice s > t must leave output at t bit-…, TestCausality

### Community 45 - "Architecture Overview: DBN Core & Clustering Strategies"
Cohesion: 0.22
Nodes (13): Source Paper Attack Graph (Figure 2), CL (Heuristic Clustering) Strategy, Layer 3: DBN Causal Core (2TBN + FF/BK inference), Layer 0: Digital Twin, EX (Exact) Inference Strategy, FF (Fully Factorized) Inference Strategy, KL Divergence / M_KL Equation (Eq. 2, Eq. 5), Session 1: Attack Graph and DBN Compiler (+5 more)

### Community 46 - "Architecture Overview: Perception & Physical Loop"
Cohesion: 0.21
Nodes (13): Claim C1: Closed Physical Loop, Layer 1: Perception (Heterogeneous GNN), Layer 4: Physical Consequence (Closed Loop), Session 4: Close the Physical Loop (Claim C1), Session 5: Perception Layer and Calibration, perception.yaml (Perception layer config), exp04_closed_loop_c1.py Experiment, exp05_perception.py Experiment (soft evidence) (+5 more)

### Community 47 - "exp08: Transfer C2 Experiment"
Cohesion: 0.28
Nodes (12): _arm_ttc_predictions(), _constant_prior_ttc(), main(), _n_slices_for_graph(), DataFrame, SeedSequence, Learned TTC parameterization: testing claim C2 (CLAUDE.md layer [2]/[3]). The…, Grand mean of realized_ttc / table3_ttc[technique] across ALL pooled training… (+4 more)

### Community 48 - "Brier Score & Skill Score"
Cohesion: 0.23
Nodes (7): brier_score(), brier_skill_score(), 1 - brier(model) / brier(base-rate-constant reference). Defends against ECE's…, Tests for src/eval/calibration.py. Written before experiments/exp04 uses it,…, The pitfall, demonstrated: a predictor outputting the base rate at every sample…, TestBrierScore, TestBrierSkillScore

### Community 49 - "RL Attacker Action Decoding Tests"
Cohesion: 0.26
Nodes (4): decode_action(), ndarray, parametrize, TestActionDecoding

### Community 50 - "Project Thesis & Absolute Rules"
Cohesion: 0.24
Nodes (11): Absolute Rules (Research Integrity), Boyen-Koller Clustering Optimization (Rejected, FF≈CL), Deliberately Rejected Directions, pandapower, pgmpy, Cyber-Physical DBN with Learned Perception project, Session 0: Scaffold and Stack Verification, simpy (+3 more)

### Community 51 - "Claim C3 & Baselines Config"
Cohesion: 0.31
Nodes (10): Claim C3: Adversarial Robustness, Session 6: External Baselines, Session 9: Adversarial Robustness (Claim C3), adversarial_c3.yaml (Claim C3 RL config), baselines.yaml (External baseline config), exp06_baselines.py Experiment (external ML comparison), exp09_adversarial_c3.py Experiment (RL attacker), Float32 Sigmoid Overflow Bug (lstm_ae/calibration) (+2 more)

### Community 52 - "Reproducibility Verification Script"
Cohesion: 0.31
Nodes (9): Pattern, diff_frame(), _import_pandas_numpy(), main(), newest_matching(), Path, Reproducibility check (Session 10, task 2): re-run experiments/exp01_…, # NOTE: exp01/exp03's own exit code reflects THEIR OWN internal (+1 more)

### Community 53 - "Slice-Boundary Timing Tests"
Cohesion: 0.33
Nodes (4): 1-based slice a continuous-time instant belongs to, matching…, slice_of(), Matches discretize()'s slice-end convention exactly., TestSliceOf

### Community 54 - "LAB_NOTEBOOK: exp01-03 Decisions"
Cohesion: 0.32
Nodes (8): delta_t_override Calibration Decision (Table 5 m=1 value), exp01_reproduce_paper.py Experiment (FF/EX reproduction), exp02: Full Scenario 1 KL under latched reactions, exp03_twin_open_loop.py Experiment, Decision: Validation gate verdict (Phase 1/2), One-shot latched reactions hypothesis test, Reaction Semantics Bug (Persistent vs Memoryless CorrReact), Reproducibility Check (exp01 PASS bit-for-bit / exp03 FAIL, BLAS non-determinism)

### Community 55 - "Summary-Table Consolidation Script"
Cohesion: 0.43
Nodes (7): figure_gnn_vs_mlp(), figure_m_sweep(), main(), newest_nonsmoke(), Path, Cross-experiment consolidation (Session 10, task 3): read every relevant…, write_axis_tables()

### Community 56 - "Run-Level Bootstrap Calibration Report"
Cohesion: 0.36
Nodes (4): calibration_report(), Full calibration report: never a single ECE number in isolation. The 95% CI is…, Resampling by run, not slice: a report built from few highly autocorrelated…, TestCalibrationReport

### Community 57 - "Asset-Graph Electrical/Cyber Coupling Tests"
Cohesion: 0.25
Nodes (3): ag.graph is the module's own in-service adjacency; the bus<->bus…, Control for the above: proves the cyber shortcut is doing real work, not just…, TestEdgesAndMetadata

### Community 59 - "Claim C2 & Uniformization Equation"
Cohesion: 0.43
Nodes (7): Claim C2: Learned TTC Parameterization, Layer 2: Parameterization (GNN/hypernetwork TTC), Session 8: Learned Parameterization (Claim C2), Uniformization Equation (Eq. 3), base.yaml (Base experiment config), transfer_c2.yaml (Claim C2 transfer config), exp08_transfer_c2.py Experiment

### Community 60 - "Sherlock Download Script"
Cohesion: 0.38
Nodes (3): download_scenario(), fetch_and_verify(), download_sherlock.sh script

### Community 61 - "Attack-Graph Family Builder Internals"
Cohesion: 0.48
Nodes (6): _add_attack_step(), _add_gate_node(), _build_graph(), DiGraph, GateType, A family of synthetic attack-graph variants for testing claim C2's transfer…

### Community 62 - "Reliability Diagram Plotting Tests"
Cohesion: 0.38
Nodes (4): One or more overlaid reliability curves (typically `{"before": ..., "after":…, reliability_diagram(), All mass in the top decile -> most bins empty; the plotted curve must have…, TestReliabilityDiagram

### Community 63 - "LAB_NOTEBOOK: Session 10 Findings"
Cohesion: 0.40
Nodes (6): Lab Notebook Protocol, Session 10: Ablations and Results Consolidation, exp10 Discretization m Sweep, exp11 GNN vs MLP Perception Ablation, Finding: GNN vs MLP Readout Pooling Confound (H3 refuted), Finding: KL(EX||FF) increases with m (H1 refuted)

### Community 64 - "Sherlock Dataset & IPAL Format"
Cohesion: 0.40
Nodes (6): Session 7: Sherlock Grounding, sherlock.yaml (Sherlock grounding config), IPAL Format (Industrial Protocol Abstraction Layer), Sherlock Dataset (Wagner et al., ACM CODASPY'25), Zenodo Record 15168928 (Sherlock v1), exp07_sherlock.py Experiment

### Community 65 - "Reaction-Node Memorylessness Tests"
Cohesion: 0.33
Nodes (3): P(WrongLogicExec) == 0.8 * P(ModCtrlLogic), same slice. Hand-computed:…, ModCtrlLogic -> WrongLogicExec: reaction, intra-slice, no persistence.…, TestReactionIsMemoryless

### Community 66 - "Scenario 1 KL Figure & DBN Nodes"
Cohesion: 0.40
Nodes (5): CorrReact (DBN node), Scenario 1 KL(EX||FF) Divergence Plot, MITM (DBN node), Source Paper Figure 6 (Cerotti et al., IEEE Access 2025), UnstablePS (DBN node)

### Community 67 - "AND/OR Gate CPT Tests"
Cohesion: 0.40
Nodes (3): Gate is untimed (self_loop=False): the A->Gate/B->Gate edges are NOT…, Row-by-row equivalence: build_gate_cpt's CPT table, evaluated at every parent-…, TestGateResolution

## Knowledge Gaps
- **27 isolated node(s):** `cyberphysicaldbn`, `Results`, `stable-baselines3`, `Experiment: verify_stack (dependency + smoke-test gate)`, `Experiment: attack graph + AG->2TBN compiler + uniformization` (+22 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **19 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GridConfig` connect `Twin-to-DBN Scenario Bundling` to `Twin Message Sanitization & Attacker Events`, `Physical Zone Classification (Consequence)`, `exp09: Adversarial RL (Claim C3)`, `exp06: Baselines Comparison Experiment`, `RL Attacker Env Tests`, `exp01/exp03 Timebase-Pinning Tests`, `Attacker Config & TTC Timing`, `exp04: Closed-Loop C1 Experiment`, `exp05: Perception Training Pipeline`, `IED Dynamic Features`, `exp03: Twin Open-Loop Experiment`, `exp08: Transfer C2 Experiment`, `RL Attacker Action Decoding Tests`, `Slice-Boundary Timing Tests`, `RL Attacker Scenario Generation`, `Asset-Graph Electrical/Cyber Coupling Tests`, `Perception Asset Graph Builder`, `exp07: Sherlock Experiment`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Why does `build_attack_graph()` connect `Attack Graph Construction (graph.py)` to `Twin Message Sanitization & Attacker Events`, `Twin-to-DBN Scenario Bundling`, `Physical Zone Classification (Consequence)`, `Amortized TTC Parameterization Model`, `DBN CPT Parameterization (Table 1-3)`, `exp04: Closed-Loop C1 Experiment`, `exp05: Perception Training Pipeline`, `exp03: Twin Open-Loop Experiment`, `exp01: Paper-Reproduction Experiment`, `RL Attacker Scenario Generation`, `exp10: m-Sweep Experiment`, `AG-to-2TBN Compiler`, `Forward-Sampling Tests`, `Forward (Generative) Sampling`, `exp07: Sherlock Experiment`, `Validation-Gate Summary Script`, `exp02: Latched-Reaction KL Experiment`, `exp09: Adversarial RL (Claim C3)`, `exp06: Baselines Comparison Experiment`, `DBN Parameterization Tests (Fixtures)`, `exp08: Transfer C2 Experiment`, `Latched-Reaction CPT Tests`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Why does `DBNInference` connect `RL Attacker Scenario Generation` to `Physical Zone Classification (Consequence)`, `DBN Belief State & Cluster Decoding`, `Attacker Config & TTC Timing`, `exp04: Closed-Loop C1 Experiment`, `exp05: Perception Training Pipeline`, `Attack-Graph Family Generation`, `exp03: Twin Open-Loop Experiment`, `FF Clustering & Soft-Evidence Wiring`, `exp01: Paper-Reproduction Experiment`, `exp10: m-Sweep Experiment`, `AG-to-2TBN Compiler`, `exp02: Latched-Reaction KL Experiment`, `RL Attacker Env Tests`, `exp09: Adversarial RL (Claim C3)`, `exp06: Baselines Comparison Experiment`, `DBN Parameterization Tests (Fixtures)`, `exp08: Transfer C2 Experiment`, `RL Attacker Action Decoding Tests`, `Reaction-Node Memorylessness Tests`, `exp01/exp03 Timebase-Pinning Tests`, `FF/EX Degenerate-Step Sanity Check`, `DBN Evidence-Stream Integration Test`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **Are the 51 inferred relationships involving `GridConfig` (e.g. with `ScenarioData` and `ScenarioBundle`) actually correct?**
  _`GridConfig` has 51 INFERRED edges - model-reasoned connections that need verification._
- **Are the 40 inferred relationships involving `GridModel` (e.g. with `ScenarioData` and `ScenarioBundle`) actually correct?**
  _`GridModel` has 40 INFERRED edges - model-reasoned connections that need verification._
- **Are the 47 inferred relationships involving `TwinRunner` (e.g. with `ScenarioData` and `ScenarioBundle`) actually correct?**
  _`TwinRunner` has 47 INFERRED edges - model-reasoned connections that need verification._
- **Are the 36 inferred relationships involving `DBNInference` (e.g. with `ScenarioData` and `ScenarioBundle`) actually correct?**
  _`DBNInference` has 36 INFERRED edges - model-reasoned connections that need verification._