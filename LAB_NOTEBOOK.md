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
