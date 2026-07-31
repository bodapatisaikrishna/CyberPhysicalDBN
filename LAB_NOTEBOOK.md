# Lab Notebook

Protocol (CLAUDE.md, non-negotiable):

Before each experiment, append:
`## [date] Experiment: <name>` / `**Hypothesis:** I expect X because Y.`

After:
`**Result:** Z.` / `**Interpretation:** W.` / `**Surprised?** yes/no — if yes, what I checked.`

A surprising result means either a bug or a finding. Both demand investigation
before moving on.

---

## 2026-07-30 Experiment: verify_stack (dependency + smoke-test gate)

**Hypothesis:** I expect all four load-bearing libraries (pandapower, pgmpy,
torch_geometric, simpy) to install cleanly into a fresh `uv`-managed `.venv`
(Python 3.11) and pass one minimal smoke test each — pandapower via
`case14()` + `runpp()`, pgmpy via a 3-node Bayesian network query, torch_geometric
via a two-node-type `HeteroData` + one `HGTConv` forward pass, and simpy via a
10-step process — because these are all mature, widely-used libraries with
no unusual version constraints expected on a standard Python 3.11 environment.
Risk flagged before running: `pandapower` 3.5.x may require `numpy>=2`, which
could conflict with the `torch` build that resolves alongside it.

**Result:** All four checks PASS on first run, no workarounds needed. Fresh `uv`
venv (Python 3.11.7, macOS-15.7.2-arm64) resolved 102 packages with no conflicts.
Versions installed: pandapower==3.5.4, pgmpy==1.1.2, torch==2.13.0,
torch-geometric==2.8.0.post1, simpy==4.1.2, networkx==3.6.1, scikit-learn==1.9.0,
stable-baselines3==2.9.0, pytest==9.1.1, pyyaml==6.0.3, pandas==2.3.3,
matplotlib==3.11.1.

- pandapower: `case14()` + `runpp()` converged, bus 0 `vm_pu=1.060000`.
- pgmpy: pgmpy 1.1.2 exposes `DiscreteBayesianNetwork` (the new name for
  `BayesianNetwork`); 3-node chain query gave `P(C=1)=0.400000`.
- torch_geometric: `HGTConv` forward pass on a 2-node-type/1-edge-type
  `HeteroData` produced the expected `(3, 16)` output shape.
- simpy: 10-step process ran to `env.now=10` exactly.

**Interpretation:** The predicted risk (pandapower 3.5.x forcing `numpy>=2`
conflicting with an existing `torch` build) did not materialize, because the
project venv is isolated from the anaconda base environment — `uv` resolved
`torch==2.13.0` fresh against `numpy==2.4.6` with no other consumer to
conflict with. This confirms the earlier decision to use a project-local
`.venv` rather than installing into anaconda base (which pins torch==2.2.1
for other work) was the right call.

**Surprised?** no — clean install was the expected outcome; the isolated-venv
choice was specifically made to avoid the one conflict scenario that seemed
plausible.

---

## 2026-07-30 Experiment: attack graph + AG->2TBN compiler + uniformization

**Hypothesis:** I expect the Figure-2 attack graph (23 nodes) to compile into a
canonical-form 2TBN with zero anterior-layer intra-slice arcs, and the N-parent
generalization of Table 1 to reproduce the paper's 8-row SpoofRepMsg CPT exactly,
because Table 1's structure decomposes into three precedence-ordered rules
(precondition-false forces 0; else self-persistence forces 1; else Bernoulli(p_s))
that are parent-count-agnostic. I expect this to hold for the N=2 case
(UnauthCommand, parents MITM + Masquerade) without special-casing.

Two predictions I am less certain of:
- The single inter-slice rule ("inter-slice iff both endpoints self-loop") should
  classify all 36 edges correctly with no exceptions. If any edge needs a special
  case, my reading of Figure 2 is wrong somewhere.
- Delta_t computed from the 11 real Table-3 TTCs at m=1 should be ~0.0684
  (= 600/8767 by hand). If the code disagrees, either my hand arithmetic or the
  TTC transcription is wrong.

**Result:** 18/18 tests pass on first run. Gate summary
(`scripts/build_and_report.py`, m=1.0 from `configs/base.yaml`):

- Attack graph: 23 nodes (12 attack_step incl. CredAccess, 8 analytic, 2 reaction,
  1 goal); 13 self-looping. 36 edges = 15 precondition + 13 self-loop + 8
  triggers_analytic; 22 inter-slice / 14 intra-slice.
- Compiled 2TBN: 36 nodes (13 anterior, 23 ulterior), 36 edges, **0 anterior-layer
  intra-slice arcs** — canonical form holds.
- Eq. 3 at m=1: `delta_t = 0.06843846241587773`, matching the hand-computed
  600/8767 to full double precision. 11 TTCs entered the sum. Largest p_s is
  UnsecCred* at 0.2053; all p_s < 1, as required for a probability.
- 23 CPTs generated, 152 table entries, every column summing to 1.

**Interpretation:** Both uncertain predictions held. The single inter-slice rule
("inter-slice iff both endpoints self-loop") classified all 36 edges with no
special cases, which is evidence the Figure-2 reading is right — a misread edge
would most likely have surfaced as either a cycle (pgmpy rejects those) or an
anterior-layer intra-slice arc. The N-parent generalization of Table 1 reproduced
the published 8-row SpoofRepMsg CPT exactly and extended to UnauthCommand's two
parents without special-casing.

**Surprised?** no, with one caveat worth recording. Table 1 rows 3-4 say a node
whose precondition is inactive goes to 0 *with probability 1 even when it was
previously active* — precondition-false outranks self-persistence, which reads
oddly for a model whose whole point is that attack steps do not revert. Checked:
that region is unreachable from an inactive initial state, because a child can
only activate while its parent is active and parents themselves persist. So it is
a don't-care region the authors filled with the forced-0 convention, not a
modeling claim about reversion. Implemented as published rather than "corrected",
and asserted explicitly in `test_two_parent_cpt_structure` so a future refactor
cannot silently flip it.

**Not implemented (stated, not silently omitted):** anterior-layer priors. Tables
1-3 specify the transition model only; the paper publishes no initial
distribution, so the model is not yet `check_model()`-valid. Deferred to the
inference phase where the prior becomes an explicit logged config choice.
