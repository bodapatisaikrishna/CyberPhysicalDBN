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

[0] Digital twin pandapower grid + abstracted comms + attacker/defender agents → labeled telemetry, ground-truth attack state, physical outcome [1] Perception Heterogeneous GNN over cyber-physical asset graph + temporal encoder → calibrated per-analytic likelihoods → SOFT EVIDENCE into the DBN [2] Parameterization GNN/hypernetwork: (technique, asset ctx, defenses) → TTC → p_s via uniformization (source paper Eq. 3) [3] DBN causal core 2TBN + FF/BK inference. Consumes soft evidence + learned CPTs. Emits posteriors, causal paths, counterfactuals. [4] Physical Compromised control actions execute in the twin → voltage/ consequence frequency evolve → instability MEASURED not asserted → physical deviation returns as evidence (THE CLOSED LOOP) [5] Decision (stretch) DBN posterior as POMDP belief state → RL defense policy [6] Explanation (opt.) Max-posterior causal path → LLM analyst narrative


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

. ├── CLAUDE.md ├── configs/ # YAML experiment configs ├── src/ │ ├── attack_graph/ # NetworkX AG + MITRE technique mapping │ ├── dbn/ # AG→2TBN compiler, CPT parameterization, FF/BK inference │ ├── twin/ # pandapower grid + SimPy comms + agents │ ├── perception/ # heterogeneous GNN, temporal encoder, calibration │ ├── parameterization/ # technique → TTC amortized model │ ├── baselines/ # LSTM-AE, GAT classifier, GBM, rule-based │ └── eval/ # metrics: lead time, ECE, KL, latency, memory ├── experiments/ # one runnable script per experiment ├── results/ # CSV outputs, gitignored except summaries ├── notebooks/ # exploration only, never load-bearing ├── tests/ └── LAB_NOTEBOOK.md # hypothesis before, finding after — every experiment


## Lab notebook protocol (non-negotiable)

Before each experiment, append to `LAB_NOTEBOOK.md`:
`## [date] Experiment: <name>` / `**Hypothesis:** I expect X because Y.`

After: `**Result:** Z.` / `**Interpretation:** W.` / `**Surprised?** yes/no —
if yes, what I checked.`

A surprising result means either a bug or a finding. Both demand investigation
before moving on.