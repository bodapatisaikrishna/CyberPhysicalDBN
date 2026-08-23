# Cyber-Physical DBN with Learned Perception and Closed Physical Loop

An extension of Cerotti et al., *"Dynamic Bayesian Networks for the
Detection and Analysis of Cyber Attacks to Power Systems,"* IEEE Access
13 (2025) 186289–186306 — a Dynamic Bayesian Network intrusion detector
for power grids in which perception and parameters are **learned from a
digital twin** rather than hand-assigned, and physical consequence is
**simulated and fed back as evidence**, closing a loop the source paper
leaves open.

The DBN formalism is preserved intact (2TBN structure, uniformization CPT
parameterization, Boyen–Koller inference, causal explainability). What
changes is everything feeding it, plus a physical layer the source paper
lacks.

![System architecture](results/figures/architecture_diagram.png)

## Three falsifiable claims

- **C1 (closed loop):** making grid physics bidirectional — attack drives
  simulated instability, physical deviation returns as evidence — improves
  detection lead time and posterior calibration vs. the open-loop DBN.
- **C2 (learned parameters):** a GNN mapping (MITRE technique, asset
  context, defensive posture) → attack-step time-to-compromise, trained on
  digital-twin executions, matches or beats expert-elicited TTCs *and*
  transfers to unseen attack graphs with zero expert input.
- **C3 (adversarial robustness):** under an RL attacker optimizing against
  the detector, the causal DBN degrades more gracefully than deep-IDS
  baselines, because structural preconditions can't be skipped.

All three are reported honestly, wins and nulls alike — see
[Key results](#key-results) below.

## Architecture

```
[0] Digital twin        pandapower grid + abstracted comms + attacker/defender agents
[1] Perception          Heterogeneous GNN + temporal encoder → calibrated soft evidence
[2] Parameterization    GNN/hypernetwork: (technique, context) → TTC → p_s (uniformization)
[3] DBN causal core     2TBN + FF/BK inference — posteriors, causal paths, counterfactuals
[4] Physical            Compromised control actions execute in the twin → instability
    consequence         MEASURED (not asserted) → returns as evidence (the closed loop)
[5] Decision (stretch)  DBN posterior as POMDP belief → RL defense policy
[6] Explanation (opt.)  Max-posterior causal path → LLM analyst narrative
```

## Key results

| Claim | Finding | Evidence |
|---|---|---|
| Paper reproduction | Measured FF KL orders of magnitude below the paper's own `2×10⁻²` target; EX/FF latency close to the paper's reference numbers | [`exp01_reproduction_gate.png`](results/figures/exp01_reproduction_gate.png) |
| C1 — closed loop | Closed-loop wins at high detection thresholds (θ≥0.7), gap grows with θ; open-loop has longer raw lead time at low θ — not a clean sweep, reported both ways | [`claims_c1_c2_c3_summary.png`](results/figures/claims_c1_c2_c3_summary.png) |
| C2 — learned TTC | `amortized` model, zero expert input, matches/beats expert-elicited TTCs on 25 held-out test graphs (detection rate 1.0 vs. 0.8 at θ=0.5) | [`exp08_ttc_fit_scatter.png`](results/figures/exp08_ttc_fit_scatter.png) |
| C3 — adversarial robustness | DBN stays in a narrow ±20-slice lead-time band across attacker-knowledge levels; `lstm_ae`/`rule_based` baselines swing to −100+ slices | [`exp09_robustness_full_sweep.png`](results/figures/exp09_robustness_full_sweep.png) |
| External baselines | Several baselines (GBM, rule-based) match or beat the DBN on raw AUC-PR — the DBN's edge is lead time and calibration, not detection accuracy, stated plainly not buried | [`exp06_pr_curve.png`](results/figures/exp06_pr_curve.png) |
| Real-data grounding | Perception layer trained/evaluated on real [Sherlock](https://sherlock.wattson.it/) grid-IDS data; twin↔Sherlock transfer gap (~0.17 AUC-PR one direction) reported as a twin-realism finding | [`exp07_pr_curve.png`](results/figures/exp07_pr_curve.png) |
| GNN cluster vs. heuristic zoning (faculty-requested) | Unsupervised GNN clustering barely agrees with the hand-built heuristic (ARI=0.09, degenerate 31-vs-2 split). A zone-supervised auxiliary loss fixes the partition (ARI→0.22, balanced) but has *zero* effect on downstream detection KL | [`exp12_spatial_zone_map.png`](results/figures/exp12_spatial_zone_map.png) |

Every number above traces to a logged experiment run with a git SHA and
seed — see [`LAB_NOTEBOOK.md`](LAB_NOTEBOOK.md) for the full
hypothesis/result/interpretation record, and
[`results/figures/`](results/figures/) for all 45 generated figures.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python scripts/verify_stack.py          # smoke-tests every dependency
pytest tests/ -q                        # 530 tests
```

Run any experiment (each writes seeded, git-SHA-stamped CSVs to `results/`):

```bash
python experiments/exp01_reproduce_paper.py     # paper reproduction gate
python experiments/exp04_closed_loop_c1.py      # closed-loop lead time (C1)
python experiments/exp09_adversarial_c3.py      # adversarial robustness (C3)
```

Rebuild every figure from already-logged results (no recomputation):

```bash
python scripts/build_summary_tables.py
python scripts/generate_journal_plots.py
python scripts/generate_publication_plots.py
```

Interactive demo dashboard (Streamlit — see `.claude/launch.json`):

```bash
pip install -e ".[demo]"
streamlit run webapp/app.py
```

## Project structure

```
src/                  DBN compiler/inference, digital twin, perception GNN,
                       parameterization, baselines, eval metrics
experiments/           exp01-exp12, one runnable script per experiment
results/               CSV outputs (git-SHA + seed logged), figures/, summary/
configs/                YAML experiment configs, no magic numbers in source
tests/                 530 tests, one per numerical component
webapp/                 Streamlit demo dashboard
docs/                  Sherlock dataset download notes, literature review
LAB_NOTEBOOK.md        hypothesis before / finding after, every experiment
CLAUDE.md              the project's living research-integrity contract
```

## Research integrity rules

This project is governed by [`CLAUDE.md`](CLAUDE.md)'s absolute rules —
worth stating here because they shape everything above:

1. Never fabricate, estimate, or hardcode a numerical result.
2. Never write a results table, plot, or summary from expected values —
   every number traces to a logged experiment run with a seed.
3. A failed validation gate is reported, not adjusted to pass.
4. Every experiment logs git SHA, seed, config, timestamp, and raw output.
5. Uncertainty is stated explicitly, not glossed over.

## Data

Twin-generated data is synthetic and gitignored. Real-world grounding
uses [Sherlock](https://sherlock.wattson.it/) (Wagner et al., ACM CODASPY
2025) — see [`docs/sherlock_download.md`](docs/sherlock_download.md) for
download steps; the dataset itself is never committed.

## Tech stack

`pandapower` (power flow) · `pgmpy` (DBN/CPDs) · `torch` + `torch-geometric`
(heterogeneous GNN) · `simpy` (attacker/comms simulation) · `networkx`
(attack graph) · `scikit-learn` (calibration, PR curves) ·
`stable-baselines3` (PPO adversarial attacker) · `pytest`

## Background reading

[`docs/literature_review.md`](docs/literature_review.md) ties every design
decision in this codebase back to a specific source — journal papers,
ACM/IEEE conference papers, arXiv preprints, GitHub repositories, and the
MITRE ATT&CK for ICS knowledge base.
