# Literature Review

Sources drawn from journal papers, arXiv preprints, ACM/IEEE conference
proceedings, open-source software repositories, and standards/knowledge
bases — any digitally verifiable source relevant to a specific design
decision in this codebase, not journal papers alone. Organized by the
architecture layer each source informs (see `CLAUDE.md`'s Layer 0-6
description and `results/figures/architecture_diagram.png`). Every entry
states what this project reused unchanged, what it extended, and what gap
it fills relative to that source.

---

## 1. Base formalism — the source paper (Layer 3: DBN causal core)

**Cerotti, D., Raiteri, D.C., Franceschinis, G., et al. "Dynamic Bayesian
Networks for the Detection and Analysis of Cyber Attacks to Power
Systems."** *IEEE Access*, vol. 13, pp. 186289-186306, 2025.
DOI/IEEE Xplore: https://ieeexplore.ieee.org/document/11214202/
*(journal paper — this project's direct extension target)*

The 2TBN structure, uniformization-based CPT parameterization (Eq. 3),
Boyen–Koller (BK) approximate-inference comparison against exact (EX) and
fully-factorized (FF) clustering, and the attack graph (their Figure 2,
reproduced in `src/attack_graph/graph.py`) are all reused UNCHANGED —
`experiments/exp01_reproduce_paper.py` exists specifically to reproduce
this paper's Phase-1 reference numbers before any extension work began
(see `results/figures/exp01_reproduction_gate.png`). What this project
adds beyond it: a closed physical loop (the source paper asserts
instability, never simulates it), learned TTC parameterization (the
source paper hand-elicits every TTC from its own Table 3), external
non-DBN baselines, real-data grounding, and an adversarial-robustness test
— none of which the source paper attempts. Its own related-work section
also motivated the choice to compare against Molloy et al. (below) on the
BK-inference question specifically.

## 2. Attack-graph approximate inference (Layer 3)

**Muñoz-González, L., Sgandurra, D., Barrère, M., Lupu, E.C. "Efficient
Attack Graph Analysis through Approximate Inference."** *ACM Transactions
on Privacy and Security (TOPS)*, 20(3), 2017.
https://dl.acm.org/doi/10.1145/3105760 — preprint: https://arxiv.org/pdf/1606.07025
*(journal/ACM paper)*

Confirms that exact Bayesian inference does not scale for attack-graph
analysis and that Loopy Belief Propagation-style approximations scale
linearly in node count — the same scaling argument CLAUDE.md's "Deliberately
rejected directions" section uses to justify *not* pursuing a learned BK
clustering (`src/dbn/inference.py`'s FF/EX clustering strategies already
sit inside the acceptable latency/memory envelope this line of work
establishes as achievable).

## 3. Digital-twin / co-simulation grounding (Layer 0)

**Wagner, M., Bader, L., Wolsing, K., Serror, M., et al. "Sherlock: A
Dataset for Process-aware Intrusion Detection Research on Power Grid
Networks."** *Proceedings of the 15th ACM Conference on Data and
Application Security and Privacy (CODASPY 2025)*.
https://dl.acm.org/doi/10.1145/3714393.3726006 — preprint:
https://arxiv.org/pdf/2504.06102 — dataset: https://sherlock.wattson.it/
*(ACM conference paper + real dataset, digitally downloaded and used
directly in `experiments/exp07_sherlock.py`)*

**Wattson co-simulation framework.** https://github.com/fkie-cad/wattson
*(GitHub software repository)*

Sherlock is the real, non-synthetic data this project's perception layer
is grounded against (Session 7 / exp07) — the ONLY source in this review
whose artifact (not just its methodology) is directly consumed by this
codebase. Wattson's own architecture (PowerOwl on top of `pandapower` for
steady-state power flow, IEC 60870-5-104 for control-center↔substation
comms, Docker/namespace-based network emulation) independently validates
two of this project's own scope decisions, stated in CLAUDE.md's
"Deliberately rejected directions": steady-state `runpp()` is sufficient
(a peer system built the same way), and abstracting comms as SimPy events
rather than a full protocol stack is a defensible simplification (Wattson
itself is the heavier, protocol-accurate alternative this project
deliberately did not build, opting to ground against Wattson's *output*
data instead of reimplementing Wattson's *simulation*).

## 4. Power-flow engine (Layer 0)

**`pandapower`** — Thurner, L., Scheidler, A., Schäfer, F., et al.
"pandapower — An Open-Source Python Tool for Convenient Modeling,
Analysis, and Optimization of Electric Power Systems." *IEEE Transactions
on Power Systems*, 33(6), 2018. GitHub: https://github.com/e2nIEE/pandapower
— docs: https://www.pandapower.org/
*(journal paper + actively maintained GitHub software dependency, pinned
in `pyproject.toml`)*

Directly used, unmodified, as `src/twin/grid.py`'s power-flow backend
(`case33bw()` feeder, `runpp()`). Chosen specifically because it is the
same tool Wattson (source 3) builds on, keeping this project's twin
methodologically consistent with the one real dataset it evaluates
against, rather than picking an unrelated simulator that would introduce
an unstated confound between the twin's physics and Sherlock's physics.

## 5. Heterogeneous graph perception (Layer 1)

**Hu, Z., Dong, Y., Wang, K., Sun, Y. "Heterogeneous Graph Transformer."**
*Proceedings of The Web Conference (WWW) 2020*, ACM.
https://dl.acm.org/doi/fullHtml/10.1145/3366423.3380027
*(ACM conference paper)*

**Schlichtkrull, M., Kipf, T.N., Bloem, P., et al. "Modeling Relational
Data with Graph Convolutional Networks (R-GCN)."** *European Semantic Web
Conference (ESWC) 2018*. arXiv: https://arxiv.org/abs/1703.06103
*(conference paper)*

`src/perception/encoder.py`'s `SpatialEncoder` offers both `HGTConv`- and
`RGCNConv`-style convolution as configurable options (`EncoderConfig`)
directly reusing these two architectures via `torch_geometric`'s
implementations — HGT's node/edge-type-dependent attention is the reason a
SINGLE heterogeneous graph (bus/line/transformer/IED/RTU/DER/relay/host —
`src/perception/asset_graph.py`) is tractable at all, versus needing a
separate model per node type. This also anchors a specific, checked design
claim in the codebase: `test_hops_from_der_bus_to_host_equals_n_gnn_layers`
verifies the chosen `n_gnn_layers=3` is exactly the graph-distance from a
DER bus to a host over the network-reachability edge type, which is the
kind of structural argument HGT's own node/edge-type-aware receptive field
makes meaningful (a homogeneous GCN would not have a principled way to
reason about "hops" across heterogeneous edge semantics the same way).

## 6. GNN-based grid intrusion/anomaly detection (motivating Layer 1's placement)

**Boyaci, O., Umunnakwe, A., Sahu, A., et al. "Joint Detection and
Localization of Stealth False Data Injection Attacks in Smart Grids using
Graph Neural Networks."** arXiv:2104.11846, 2021.
https://arxiv.org/pdf/2104.11846
*(arXiv preprint)*

**Confirms this project's own stated novelty claim rather than merely
inspiring it.** Boyaci et al.'s search summary (and the broader body of
GNN-for-grid-security work surveyed alongside it) notes that "traditional
GNN-based methods... primarily rely on network data... failing to
adequately integrate physical data from power grid devices." This is
exactly the gap `src/perception/encoder.py`'s comment about the GNN
belonging on the "large cyber-physical ASSET graph" (not the ~20-node
attack graph, and not a network-topology-only graph) is written against —
CLAUDE.md's "Deliberately rejected directions" independently arrives at
the same conclusion this literature flags as a known limitation of prior
work: a GNN needs BOTH cyber and physical node types in one graph to be
worth its structural cost.

## 7. Calibration (Layer 1, `src/perception/calibration.py`)

**Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q. "On Calibration of Modern
Neural Networks."** *Proceedings of the 34th International Conference on
Machine Learning (ICML) 2017*, PMLR 70. arXiv: https://arxiv.org/abs/1706.04599
*(conference paper)*

Temperature scaling (a single scalar dividing pre-softmax logits, fit on
a held-out split) is implemented in `src/perception/calibration.py`
exactly as this paper specifies — chosen over more complex calibration
methods (Platt scaling with more parameters, isotonic regression)
precisely because Guo et al.'s own finding is that the single-parameter
variant is "surprisingly effective" and does not risk overfitting the
small held-out calibration split this project's twin can generate.
Expected Calibration Error (ECE), used throughout `results/figures/*calibration*.png`
and `src/eval/calibration.py`, is the same metric this paper introduces.

## 8. MITRE ATT&CK for ICS (Layer 0/2 — attack-step technique mapping)

**MITRE ATT&CK for Industrial Control Systems.**
https://attack.mitre.org/matrices/ics/ — maintained by MITRE Corporation,
public release 2020, continuously updated.
*(standards / public knowledge base, not a paper — explicitly the kind of
source the user asked to include)*

`src/attack_graph/graph.py`'s `mitre_technique_id`/`mitre_tactic` fields
and `technique_table3_ttc()` (added for exp08's amortized TTC model) map
this project's attack-graph nodes onto real ATT&CK-for-ICS technique IDs
rather than inventing an ad hoc taxonomy — the same choice the source
paper (source 1) itself makes, and the reason CLAUDE.md's Session 8
(learned parameterization) can meaningfully claim its GNN embeds "MITRE
technique" as one of its context features: the technique IDs are drawn
from a real, externally maintained ontology, not project-internal labels
that would make the "transfers to unseen attack graphs" claim (C2)
untestable outside this codebase.

## 9. Adversarial RL against intrusion detectors (Layer 4/C3)

**Vitorino, J., Andrade, R., Praça, I., Sousa, O., Maia, E. "Evading Deep
Reinforcement Learning-based Network Intrusion Detection with Adversarial
Attacks."** *Proceedings of the 17th International Conference on
Availability, Reliability and Security (ARES 2022)*, ACM.
https://dl.acm.org/doi/fullHtml/10.1145/3538969.3539006
*(ACM conference paper)*

Directly motivates `experiments/exp09_adversarial_c3.py`'s central design
question — whether a detector's structural constraints bound how much an
RL-trained adversary can evade it. This body of work demonstrates the
opposite finding for black-box ML detectors (RL agents CAN find evasive
paths against them, generalizing the source paper's own unaddressed
concern from a stated risk into a demonstrated one for other detector
families). C3's own honest, structural-precondition-based counter-claim —
that a causal DBN's hard preconditions (e.g. MITM requires CredAccess
first) put a floor under evasion that a purely statistical detector lacks
— is this project's answer to exactly the vulnerability this ARES 2022
paper demonstrates in non-causal detectors; `src/twin/rl_attacker.py`'s
three knowledge-level design (blind / analytics-known / full-DBN-known)
mirrors this line of work's own escalating-adversary-knowledge structure.

## 10. RL training infrastructure (Layer 4)

**`stable-baselines3`** — Raffin, A., Hill, A., Gleave, A., et al.
"Stable-Baselines3: Reliable Reinforcement Learning Implementations."
*Journal of Machine Learning Research*, 22(268), 2021. GitHub:
https://github.com/DLR-RM/stable-baselines3 — docs:
https://stable-baselines3.readthedocs.io/
*(JMLR paper + actively maintained GitHub software dependency, pinned in
`pyproject.toml`)*

PPO from this library is used unmodified as `src/twin/rl_attacker.py`'s
training algorithm — chosen (over a custom PPO implementation) specifically
because this project's own reproducibility rule (CLAUDE.md rule 4: every
experiment logs a seed and must reproduce) depends on an implementation
whose seeding/determinism behavior is independently tested and documented,
which a from-scratch PPO would not offer without redoing exactly that
verification work this library's own test suite already provides.

---

## Summary: source types actually used

| Type | Count | Examples |
|---|---|---|
| Journal paper | 3 | Cerotti et al. (IEEE Access), pandapower (IEEE Trans. Power Systems), TOPS (Muñoz-González) |
| ACM/IEEE conference paper | 4 | HGT (WWW), Sherlock (CODASPY), evasion (ARES), ESWC (R-GCN) |
| arXiv preprint | 2 | GNN FDIA detection (Boyaci et al.), calibration (Guo et al., also ICML) |
| GitHub software repository | 3 | pandapower, Wattson, stable-baselines3 |
| Standards / public knowledge base | 1 | MITRE ATT&CK for ICS |

Per-source detail on what was reused verbatim vs. extended is in each
section above, not repeated here — this table exists only to show the
source-type breadth explicitly requested, not as a substitute for the
per-source reasoning.
