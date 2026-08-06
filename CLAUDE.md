# Claude Code Prompt Pack
## Cyber-Physical DBN with Learned Perception — Build Guide

**How to use this document:**

1. Create your project directory, then save **Part 1** as `CLAUDE.md` at the repo root. Claude Code reads this automatically every session — it is the single most important file in the pack.
2. Run the session prompts in **Part 2** in order. One session per prompt. Do not skip ahead.
3. Each phase has a **validation gate**. Do not proceed past a gate until it passes. This is not bureaucracy — it is the only thing standing between you and six weeks of results built on a silent bug.

---

# PART 1 — `CLAUDE.md` (save this at repo root)

```markdown
# Project: Cyber-Physical DBN with Learned Perception and Closed Physical Loop

## What this project is

An extension of Cerotti et al., "Dynamic Bayesian Networks for the Detection and
Analysis of Cyber Attacks to Power Systems," IEEE Access 13 (2025) 186289–186306.

**Thesis:** A cyber-physical Dynamic Bayesian Network in which perception and
parameters are LEARNED from a digital twin rather than hand-assigned, and in which
physical consequence is SIMULATED and fed back as evidence — closing a loop the
source paper leaves open.

The DBN formalism is preserved intact: 2TBN structure, uniformization-based CPT
parameterization, Boyen–Koller inference, causal explainability. What changes is
everything feeding it, plus a physical layer the source paper lacks.

## Three falsifiable claims (everything serves these)

- **C1 (closed loop):** Making grid physics bidirectional — attack drives simulated
  instability, physical deviation returns as evidence — improves detection lead time
  and posterior calibration vs. the open-loop DBN.
- **C2 (learned parameters):** A GNN mapping (MITRE technique, asset context,
  defensive posture) → attack-step TTC, trained on twin executions, matches or beats
  expert-elicited TTCs AND transfers to unseen attack graphs.
- **C3 (adversarial robustness — highest novelty):** Under an RL attacker optimizing
  against the detector, the causal DBN degrades more gracefully than deep-IDS
  baselines, because structural preconditions cannot be skipped.

## Architecture (layer → responsibility)

```
[0] Digital twin        pandapower grid + abstracted comms + attacker/defender agents
                        → labeled telemetry, ground-truth attack state, physical outcome
[1] Perception          Heterogeneous GNN over cyber-physical asset graph
                        + temporal encoder → calibrated per-analytic likelihoods
                        → SOFT EVIDENCE into the DBN
[2] Parameterization    GNN/hypernetwork: (technique, asset ctx, defenses) → TTC
                        → p_s via uniformization (source paper Eq. 3)
[3] DBN causal core     2TBN + FF/BK inference. Consumes soft evidence + learned CPTs.
                        Emits posteriors, causal paths, counterfactuals.
[4] Physical            Compromised control actions execute in the twin → voltage/
    consequence         frequency evolve → instability MEASURED not asserted
                        → physical deviation returns as evidence (THE CLOSED LOOP)
[5] Decision (stretch)  DBN posterior as POMDP belief state → RL defense policy
[6] Explanation (opt.)  Max-posterior causal path → LLM analyst narrative
```

**Division of labor, never violated:** neural networks do perception (noisy,
high-dimensional, statistical). The DBN does causal fusion (structured, temporal,
explainable). Neither is asked to do the other's job.

## Key equations (from the source paper — reuse exactly, do not invent variants)

Uniformization, for concurrent attack steps with mean completion times T̄ᵢ:

    Δt  = 1 / ( m · Σᵢ (1 / T̄ᵢ) )
    p_s = Δt / T̄_s

where m controls discretization accuracy. Phase 4's learned model predicts T̄_s;
this equation converts it to a CPT entry.

KL divergence:   D_KL(P‖Q) = Σₓ P(x) · log( P(x) / Q(x) )
Max over time:   M_KL = max_{t ∈ [0,T]} ψ(t)

## Reference numbers to reproduce in Phase 1 (validation gate)

| Config | KL (target variable) | Per-slice time | Memory |
|---|---|---|---|
| EX (exact 1.5JT) | 0 (reference) | ~0.22 s | 1686 MB (m=1/3) to 5054 MB (m=1); fails above m>1 on 32 GB |
| CL (heuristic) | below ~1.25 × 10⁻⁴ | ~0.03 s | tens of MB |
| FF (fully factorized) | within ~2 × 10⁻² | ~0.03 s | tens of MB |

The source paper's own finding is that **FF ≈ CL**. Therefore a clean FF
implementation may be all that is ever needed. Do NOT sink weeks into full
Boyen–Koller clustering optimization — that direction was explicitly evaluated
and rejected for this project.

## Attack graph (from source paper Figure 2)

Three attack processes, ~20 nodes total.

- **Centre (MITM chain):** UnsecCred1 → UnsecCred2 → UnsecCred → (with
  ModAuthProc1 → ModAuthProc) → CredAccess(AND) → MITM →
  {SpoofRepMsg → CorrReact, UnauthCommand} → UnstablePS(OR)
- **Left (Stuxnet-style):** ModCtrlLogic → WrongLogicExec → UnstablePS
- **Right (rogue ICS service):** ModifyProgram → Masquerade → UnauthCommand

Analytic (evidence) nodes: FileAccess, FileIntegrity, MeasureCoherence,
CommandCoherence, NewServiceStarted, SWIntegrityDER, SWIntegritySCADA, SuspArg.

Analytics are UNTIMED nodes (no self-loops, no temporal arcs). Attack nodes have
self-loops (persistence — once achieved, an attack step does not revert).

CorrReact and WrongLogicExec are NOT attack steps — they model control-center
reactions. Source paper sets CorrReact success 0.7, WrongLogicExec 0.8.

## Tech stack (fixed — do not substitute without asking)

- `pandapower` — steady-state power flow. Use `runpp()`. NOT dynamic simulation.
- `pgmpy` — DBN structure and CPD representation
- `torch` + `torch_geometric` — GNN. Use `HeteroData` and `HGTConv`/`RGCNConv`
  for the heterogeneous asset graph.
- `simpy` — discrete-event attacker/comms simulation
- `networkx` — attack graph before DBN compilation
- `scikit-learn` — calibration (temperature scaling), PR curves, ECE
- `stable-baselines3` — PPO, for the Phase 5 RL attacker only
- `pytest` — every numerical component gets a test

## Scope discipline — CUT ORDER under time pressure

Cut from the top of this list first:
federated learning → defender RL → LLM narrative → LLM attack-graph construction
→ adaptive attacker → learned parameterization

**Minimum viable publication:** Phase 1 + Phase 2 + Phase 3 + external baselines
+ lead-time evaluation. Everything past that is upside.

## Deliberately rejected directions (do not propose these)

- Learning the BK clustering with GNN+RL. The source paper shows FF ≈ CL, so
  headroom is ~10⁻² KL. Per-slice inference (0.03 s) already fits inside Δt
  (8–500 s). This optimizes a solved problem.
- Putting the GNN on the ~20-node attack graph. Too small; no representational
  signal. The GNN belongs on the large cyber-physical ASSET graph.
- Dynamic/transient power simulation. Steady-state `runpp()` is sufficient for
  the instability-detection claim.
- Real IEC 61850/MMS protocol stack. Abstract communication as SimPy events.
  No claim in this project is about protocol-level evasion.

## ABSOLUTE RULES — violating these invalidates the research

1. **NEVER fabricate, estimate, hardcode, or placeholder a numerical result.**
   If an experiment has not been run, the value does not exist. Write `NotImplemented`
   or raise, never a plausible-looking number. This includes docstrings, README
   tables, and example outputs.
2. **NEVER write a results table, plot, or summary from expected values.**
   Every reported number traces to a logged experiment run with a seed.
3. **If a validation gate fails, STOP and report the discrepancy.** Do not adjust
   the target to match your output. Do not "approximately" pass a gate.
4. **Every experiment logs:** git commit SHA, random seed, full config, timestamp,
   and raw per-run outputs to a CSV under `results/`. No exceptions.
5. **State uncertainty explicitly.** If you are unsure whether an implementation
   matches the paper, say so and cite the specific equation or figure you are
   unsure about. Do not guess silently.
6. **Ask before scope expansion.** If a task seems to require a component not in
   the current phase, stop and ask.

## Code standards

- Type hints on all public functions.
- Docstrings cite the source-paper section/equation/figure being implemented,
  e.g. `"""Implements uniformization (Cerotti et al. Eq. 3, Sec. III-E)."""`
- Numerical code gets a `pytest` test with a hand-computed expected value.
- Config in YAML under `configs/`. No magic numbers in source.
- Seeds set explicitly and logged. `numpy`, `torch`, `random`, and env seeds.
- Never commit dataset files. `data/` is gitignored; document download steps.

## Directory layout

```
.
├── CLAUDE.md
├── configs/              # YAML experiment configs
├── src/
│   ├── attack_graph/     # NetworkX AG + MITRE technique mapping
│   ├── dbn/              # AG→2TBN compiler, CPT parameterization, FF/BK inference
│   ├── twin/             # pandapower grid + SimPy comms + agents
│   ├── perception/       # heterogeneous GNN, temporal encoder, calibration
│   ├── parameterization/ # technique → TTC amortized model
│   ├── baselines/        # LSTM-AE, GAT classifier, GBM, rule-based
│   └── eval/             # metrics: lead time, ECE, KL, latency, memory
├── experiments/          # one runnable script per experiment
├── results/              # CSV outputs, gitignored except summaries
├── notebooks/            # exploration only, never load-bearing
├── tests/
└── LAB_NOTEBOOK.md       # hypothesis before, finding after — every experiment
```

## Lab notebook protocol (non-negotiable)

Before each experiment, append to `LAB_NOTEBOOK.md`:
`## [date] Experiment: <name>` / `**Hypothesis:** I expect X because Y.`

After: `**Result:** Z.` / `**Interpretation:** W.` / `**Surprised?** yes/no —
if yes, what I checked.`

A surprising result means either a bug or a finding. Both demand investigation
before moving on.
```

---

# PART 2 — Session prompts, in order

Run one per Claude Code session. Each ends at a validation gate.

---

## Session 0 — Scaffold and stack verification

```
Read CLAUDE.md fully before doing anything.

This session does ONLY project scaffolding and dependency verification. Write no
research logic yet.

Tasks:
1. Create the directory layout specified in CLAUDE.md, with __init__.py files and
   a .gitignore that excludes data/, results/*.csv, __pycache__, .venv, and
   model checkpoints.
2. Create pyproject.toml (or requirements.txt) pinning: pandapower, pgmpy, torch,
   torch-geometric, simpy, networkx, scikit-learn, stable-baselines3, pytest,
   pyyaml, pandas, matplotlib.
3. Write scripts/verify_stack.py that imports every dependency, prints its version,
   and runs one minimal smoke test each:
   - pandapower: load networks.case14(), runpp(), assert convergence, print a
     bus voltage
   - pgmpy: build a 3-node BayesianNetwork with CPDs, run one query
   - torch_geometric: construct a small HeteroData object with two node types and
     one edge type, run one HGTConv forward pass
   - simpy: run a 10-step trivial process
   Each check prints PASS/FAIL independently and the script exits non-zero if any
   fail.
4. Create LAB_NOTEBOOK.md with the protocol header from CLAUDE.md.
5. Create configs/base.yaml holding: random seed, discretization parameter m,
   simulation horizon T, and paths.

Run verify_stack.py and report the actual output. Do not summarize it as working
if any check failed — paste what happened.

VALIDATION GATE: all four smoke tests PASS. If torch-geometric or pandapower fails
to install, stop and report the exact error rather than working around it.
```

---

## Session 1 — Attack graph and DBN compiler

```
Read CLAUDE.md. This session builds the attack graph and the AG→2TBN compiler.
No twin, no GNN, no inference yet.

Tasks:
1. src/attack_graph/graph.py — represent the source paper's Figure 2 attack graph
   as a NetworkX DiGraph. Nodes carry: name, node_type (attack_step | analytic |
   reaction | goal), mitre_technique_id, mitre_tactic, ttc (mean time to
   completion). Edges carry: edge_type (precondition | triggers_analytic) and
   whether the dependency is inter-slice.

   Encode all three attack processes and all eight analytic nodes exactly as
   specified in CLAUDE.md. Include the AND semantics at CredAccess and OR at
   UnstablePS. Use the source paper's Table 3 TTC values — if you are unsure of a
   specific value, mark it None and list which ones you could not determine
   rather than inventing a number.

2. src/dbn/compiler.py — compile the NetworkX AG into a 2TBN. Requirements:
   - attack-step nodes get self-loops (temporal persistence)
   - analytic nodes are UNTIMED (no self-loop, no temporal arc)
   - canonical form (no intra-slice arcs in the anterior layer)
   - output a pgmpy-compatible structure

3. src/dbn/parameterization.py — implement uniformization exactly as Eq. 3 in
   CLAUDE.md. Signature roughly:
       compute_delta_t(ttcs: dict[str, float], m: float) -> float
       compute_ps(ttc: float, delta_t: float) -> float
       build_attack_step_cpt(node, parents, p_s) -> TabularCPD
       build_analytic_cpt(node, p_pos, p_neg) -> TabularCPD
   The attack-step CPT must reproduce the structure of the paper's Table 1
   (8 rows for a node with one parent plus self): rows where the precondition has
   not occurred give probability 0 of activation; rows where the node is already
   active preserve state.

4. tests/test_parameterization.py — hand-compute Δt and p_s for a two-step example
   with known TTCs and m=1, and assert the code matches. Assert every generated
   CPT's columns sum to 1.0. Assert the Table 1 structural properties above.

VALIDATION GATE: all tests pass, and print a summary showing node count, edge
count, and how many CPTs were generated. Report the numbers — do not assert they
look correct.
```

---

## Session 2 — FF inference and paper reproduction

```
Read CLAUDE.md, especially the reference-numbers table.

This session implements inference and reproduces the source paper. This is THE
critical validation gate of the whole project — everything downstream assumes it
passed.

Tasks:
1. src/dbn/inference.py — implement forward filtering over the 2TBN with a
   pluggable clustering strategy. Implement two strategies first:
   - FF (fully factorized): each interface node in its own cluster
   - EX (exact): single cluster containing all interface nodes
   Interface nodes = nodes with inter-slice connections.
   Expose: step(evidence) -> posteriors, and run(evidence_stream, T) -> trajectory.

   Before writing code, state in one short paragraph what you understand an
   interface node to be and why the joint over interface nodes is the
   computational bottleneck. If your understanding is uncertain, say so.

2. src/eval/metrics.py — KL divergence (Eq. 2), M_KL (Eq. 5), per-slice wall-clock
   latency, and peak memory (use tracemalloc or psutil; document which and note
   that it is a lower bound).

3. experiments/exp01_reproduce_paper.py — reproduce the source paper's two
   scenarios:
   - Scenario 1: no evidence at all; forward-predict attack progression to T=200
   - Scenario 2: partial monitoring. Analytics fire at the paper's Table 4 times:
     FileAccess t=8, FileIntegrity t=9, MeasureCoherence t=31,
     CommandCoherence t=52, NewServiceStarted never.
     Set p_pos = p_neg = 1e-4 for all analytics (the paper's assumption).
   For each scenario compute KL(EX‖FF) over time for nodes UnstablePS, CorrReact,
   MITM, plus latency and memory. Log everything to results/ with seed and config.

4. Produce plots matching the paper's Figures 5–8 format.

VALIDATION GATE — report a comparison table of your numbers against the CLAUDE.md
reference table. Specifically check:
  - Scenario 2: does UnsecCred posterior go to ~1.0 right after t=8?
  - Scenario 2: does MITM reach ~1.0 by t≈20?
  - Scenario 2: at t=31, do SpoofRepMsg / CorrReact / UnstablePS reach
    approximately 1.0 / 0.7 / 0.85?
  - Is FF's KL on UnstablePS within order 10⁻²?
  - Is per-slice latency for FF around 0.03 s (order of magnitude)?

If any of these do not match, STOP. Do not proceed to Session 3. Report the
discrepancy and your best hypothesis about the cause. Do NOT adjust targets or
describe a near-miss as a match.
```

---

## Session 3 — Digital twin, open loop first

```
Read CLAUDE.md. Session 2's gate must have passed.

Build the digital twin. Scope discipline is critical here — this is where the
project most easily balloons. Steady-state power flow only. Abstracted comms only.

Tasks:
1. src/twin/grid.py — pandapower model of a distribution feeder with DERs,
   matching the source paper's control-center / DER / SCADA setting. Include: a
   control-center-controlled setpoint, at least two DERs, and enough buses that
   the asset graph is non-trivial (target 20–40 buses). Expose:
       apply_control_action(action) -> None
       solve() -> GridState        # runpp under the hood
       is_unstable() -> bool       # threshold on voltage/frequency deviation
   Document the instability threshold choice in a comment and in LAB_NOTEBOOK.md.

2. src/twin/comms.py — SimPy model of control-center ↔ DER message exchange.
   Abstract, NOT a protocol stack: messages carry (type, sender, receiver,
   payload_category, timestamp). Support normal exchange plus the manipulations
   the attack graph requires: measurement spoofing, command injection, and
   logic modification.

3. src/twin/attacker.py — scripted agent executing the three attack processes.
   Each attack step completes after a stochastic delay sampled from a distribution
   parameterized by that step's TTC. On completion, trigger the connected
   analytics with (a) a stochastic firing delay and (b) configurable
   p_pos / p_neg — replacing the paper's hand-picked Table 4 firing times.

4. src/twin/runner.py — orchestrate: attacker acts → comms carries manipulated
   messages → grid applies control actions → grid state evolves → analytics fire →
   emit an evidence stream. Output a structured trace: per time step, the
   ground-truth attack state, the analytic firings, and the full grid state.

5. experiments/exp02_twin_open_loop.py — run the twin, feed its evidence stream
   into the Session 2 DBN, and compare posterior trajectories against the paper's
   hand-scripted Scenario 2.

VALIDATION GATE: the twin runs end to end and produces a trace where (a) attack
steps complete in an order consistent with the graph's preconditions — never a
child before its parent, (b) analytics fire only after their triggering step
except for false positives, and (c) grid state changes measurably when a
compromised control action is applied. Assert (a) and (b) as tests. Report
whether the twin-driven posteriors resemble the scripted scenario, and where they
diverge.

Do NOT implement physical feedback into the DBN this session. That is Session 4.
```

---

## Session 4 — Close the physical loop (CLAIM C1)

```
Read CLAUDE.md. This session tests Claim C1. This is a real experiment with a
real possibility of a null result — treat a null honestly, it is publishable.

Tasks:
1. Add physical-deviation evidence nodes to the DBN. Design decision to reason
   about explicitly before coding: different attack paths should produce different
   physical signatures (spoofed measurements causing a wrong control response looks
   different from malicious logic misdirecting actuators). Write out your proposed
   CPT structure for these nodes and the justification in LAB_NOTEBOOK.md BEFORE
   implementing. If you cannot justify distinguishable signatures from the grid
   model, say so — that itself predicts a C1 null.

2. src/twin/consequence.py — make UnstablePS a measured outcome, not an asserted
   one: compromised control actions execute in the grid, and instability is
   detected from the solved power flow.

3. Wire physical deviation back as evidence into the DBN. The loop:
   attack → control action → grid state → deviation → evidence → posterior.

4. src/eval/lead_time.py — the headline metric. Define precisely:
       t_detect      = first t where P(UnstablePS) crosses threshold θ
       t_instability = first t where twin reports is_unstable()
       lead_time     = t_instability - t_detect
   Handle explicitly: posterior never crosses (missed detection), crosses with no
   instability (false alarm), multiple crossings. Sweep θ rather than fixing one
   value. Write this function and its tests BEFORE running the experiment — a
   post-hoc metric definition is a reviewable flaw.

5. experiments/exp03_closed_loop_c1.py — open-loop vs. closed-loop DBN on identical
   twin scenarios and identical seeds. Report lead time (across θ sweep),
   calibration (ECE), and posterior trajectories. Minimum 30 scenario runs with
   distinct seeds; report mean and spread, not a single run.

VALIDATION GATE: report the C1 result with its uncertainty. If closed-loop does
not beat open-loop, say so plainly and analyze why — do not re-tune until it wins.
Log the hypothesis and the finding in LAB_NOTEBOOK.md.
```

---

## Session 5 — Perception layer and calibration

```
Read CLAUDE.md. Replace the fictional p_pos = p_neg = 1e-4 with learned,
calibrated likelihoods.

Tasks:
1. src/perception/asset_graph.py — build the heterogeneous cyber-physical asset
   graph as a PyG HeteroData. Node types: bus, line, transformer, IED, RTU, DER,
   relay, host. Edge types: electrical_coupling, network_reachability,
   control_authority. Derive electrical topology from the pandapower net; define
   the cyber overlay explicitly in config.

   State clearly in a comment why this graph, and not the ~20-node attack graph,
   is where a GNN earns its place.

2. src/perception/encoder.py — heterogeneous GNN (HGTConv or RGCNConv, 2–3 layers)
   plus a temporal encoder (start with a small TCN) over telemetry windows.
   Output: per-analytic-node logit.

3. src/perception/calibration.py — temperature scaling fitted on a held-out split.
   Implement ECE and reliability diagrams. This is a core contribution, not an
   afterthought: report ECE before and after calibration.

4. src/dbn/soft_evidence.py — accept likelihoods rather than hard observations
   and enter them as VIRTUAL / LIKELIHOOD evidence on analytic nodes.

   Before implementing: state your understanding of how virtual evidence differs
   from observed evidence in a Bayesian network, and confirm how pgmpy handles it.
   If pgmpy does not support it directly, say so and propose the correct
   likelihood-ratio formulation rather than approximating with hard evidence.

5. experiments/exp04_perception.py — train on twin telemetry; evaluate AUC-PR and
   ECE. Then the key ablation: DBN posteriors under (a) hard binary evidence,
   (b) uncalibrated soft evidence, (c) calibrated soft evidence. The expected
   story is that uncalibrated neural evidence corrupts posteriors while calibrated
   evidence does not — verify or refute it.

VALIDATION GATE: report AUC-PR, ECE before/after calibration, and the three-way
ablation. Include the reliability diagram. If calibration does not improve
posterior quality, report that.
```

---

## Session 6 — External baselines

```
Read CLAUDE.md. The source paper compares only against its own inference variants.
Reviewers will demand external baselines. An undertrained baseline is the fastest
route to rejection — tune these honestly.

Tasks:
1. src/baselines/ — implement and tune:
   - LSTM autoencoder, reconstruction-error anomaly detection on telemetry
   - GAT or GraphSAGE end-to-end binary classifier over the asset graph
   - Gradient-boosted trees on engineered features
   - Rule/signature-based IDS proxy
   Use audited implementations where available (PyOD for AE/GBM) rather than
   writing from scratch.

2. For each baseline, run a documented hyperparameter search. Log the search space
   and the selected config. State how much tuning budget each received and confirm
   it is comparable to what the proposed system received.

3. experiments/exp05_baselines.py — evaluate all baselines and the proposed system
   on identical twin scenarios and seeds. Report AUC-PR, detection lead time, and
   ECE.

VALIDATION GATE: report the full comparison table. If a baseline beats the proposed
system on raw AUC-PR, report it prominently — the project's argument is lead time,
calibration, explainability, and robustness under adaptation, not raw AUC on
scripted attacks. Do not bury an unfavorable number.
```

---

## Session 7 — Sherlock grounding

```
Read CLAUDE.md. Twin-only evaluation invites the criticism the source paper already
absorbs. Ground the perception layer on real data.

Context: Sherlock (2025) is a power-grid IDS dataset from the Wattson co-simulator,
using pandapower for power flow and IEC 60870-5-104 for control-center-to-substation
communication. Three scenarios (01-Basic, 02-Semiurban, 03-Rural) over 35 days;
two networks have both attack-free and attack data, one is attack-only. It includes
network captures, host logs, and process ground truth. Available at
https://sherlock.wattson.it/

Tasks:
1. Write scripts/download_sherlock.sh and document the manual steps. Do not commit
   data. If the download or format differs from the description above, report the
   actual structure rather than adapting silently.
2. src/perception/sherlock_loader.py — parse Sherlock into the same feature and
   asset-graph representation the twin produces. Document every mismatch between
   Sherlock's schema and the twin's, and how you reconciled it.
3. experiments/exp06_sherlock.py — train/evaluate the perception layer on Sherlock.
   Report AUC-PR and ECE on held-out scenarios. Then test transfer: Sherlock-trained
   → twin-evaluated, and twin-trained → Sherlock-evaluated.

VALIDATION GATE: report perception-layer performance on real data honestly. A large
twin→Sherlock gap is an important finding about twin realism, not a failure to hide.
```

---

## Session 8 — Learned parameterization (CLAIM C2)

```
Read CLAUDE.md. This session tests Claim C2 and attacks the source paper's
admitted weakness that all TTCs are expert guesses.

Tasks:
1. Instrument the twin to measure actual attack-step completion times across
   randomized runs, varying asset configuration, defensive posture, and attacker
   capability.
2. src/parameterization/amortized.py — model mapping (MITRE technique embedding,
   asset context, defensive posture, attacker capability) → T̄_s, then p_s via
   the uniformization equation from CLAUDE.md. Reuse the existing
   parameterization module for the conversion — do not reimplement Eq. 3.
3. src/attack_graph/family.py — generate a family of attack-graph variants varying
   depth, branching factor, technique mix, and analytic coverage. Target 40–60
   graphs. Split 30 train / 5 validation / 25 test.
4. experiments/exp07_transfer_c2.py — parameterize held-out graphs with ZERO expert
   input. Compare downstream detection quality against (a) expert-elicited TTCs
   from the source paper's Table 3 and (b) a constant-prior control.

VALIDATION GATE: transfer is the entire claim — same-graph fitting is not a
contribution. Report performance on the 25 test graphs with no retraining. If
transfer fails, diagnose WHICH features fail to transfer; that diagnostic is
itself a finding.
```

---

## Session 9 — Adversarial robustness (CLAIM C3)

```
Read CLAUDE.md. Highest-novelty experiment. Comes directly from the source paper's
own unaddressed concern that an attacker knowing the DBN could choose low-detection
paths.

Tasks:
1. src/twin/rl_attacker.py — Gym-compatible environment wrapping the twin. Action
   space: attack-graph path selection plus stealth/timing parameters. Reward: reach
   UnstablePS while minimizing cumulative detection probability. Train with PPO.
2. Implement three attacker knowledge levels:
   (a) blind, (b) knows which analytics are deployed, (c) knows the full DBN
   structure and parameters.
3. experiments/exp08_adversarial_c3.py — run each knowledge level against the
   proposed system AND every Session 6 baseline. Produce the robustness curve:
   detection lead time vs. attacker knowledge.

The hypothesis to test: the DBN's hard structural preconditions bound how much
evasion is possible — you cannot inject a spoofed reporting message without first
establishing MITM, and MITM requires credential access. Black-box detectors have
no such floor.

VALIDATION GATE: report the robustness curve for all systems. If the DBN degrades
as fast as the baselines, C3 is refuted — report that clearly. Confirm the RL
attacker actually learned something (show reward curves) rather than failing to
train, which would produce a spuriously favorable result.
```

---

## Session 10 — Ablations and results consolidation

```
Read CLAUDE.md and LAB_NOTEBOOK.md in full.

Tasks:
1. Run the full ablation set: open vs. closed loop; expert vs. learned TTCs;
   calibrated vs. uncalibrated vs. hard evidence; GNN perception vs. per-asset MLP
   (isolates whether graph structure helps); cluster/config sweeps; the
   discretization m sweep from the source paper's Section IV-B.
2. Verify reproducibility: re-run three previously logged experiments from their
   configs and confirm identical numbers. Report any drift.
3. Consolidate every logged result into results/summary/ tables and paper-format
   figures matching the source paper's conventions.
4. Audit: grep the entire repo for hardcoded numerics in reporting paths, README
   tables, and docstrings. Confirm every reported number traces to a logged run
   with a seed. Report anything that does not.

VALIDATION GATE: the reproducibility check passes, and the audit finds zero
untraceable numbers. If the audit finds any, list them.
```

---

# PART 3 — Operating rules for these sessions

**One session per phase.** Long sessions degrade; Claude Code loses track of
constraints as context fills. When a session ends, commit, then start fresh — the
`CLAUDE.md` carries the context forward.

**Never let a gate slide.** The most likely failure mode of this entire project is
a Session 2 near-miss waved through as "close enough," discovered in Session 8.

**Push back on confident code.** When Claude Code produces a large module without
flagging any uncertainty, ask directly: *"Which parts of this are you least sure
match the paper? What did you have to guess?"* Research code that never expresses
doubt usually contains silent assumptions.

**Watch for these specific failure signatures:**
- A results table appearing before the experiment ran
- `p_pos = 1e-4` style constants leaking into places that should use learned values
- The GNN quietly migrating onto the attack graph instead of the asset graph
- Dynamic power simulation creeping in where `runpp()` was specified
- "Approximately matches the paper" without a side-by-side number comparison

**Update `CLAUDE.md` as decisions get made.** When you resolve the BK-vs-FF
question, or fix the instability threshold, or settle the asset-graph schema —
write it into `CLAUDE.md`. It is a living contract, and future sessions depend
on it being current.