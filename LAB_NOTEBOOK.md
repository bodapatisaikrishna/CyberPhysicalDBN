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

---

## 2026-07-31 Experiment: forward filtering (FF/EX) + paper reproduction

**Hypothesis:** I expect the compiled model, run through pgmpy VariableElimination
with a per-step anterior-prior swap (FF: 13 independent CPDs; EX: one 8192-state
AntJoint auxiliary node), to reproduce the paper's Scenario 1 and Scenario 2
posterior trajectories and KL(EX||FF) curves within the target shapes/orders of
magnitude below. Initial belief at t=0 is deterministic all-13-interface-nodes-
inactive, evidenced by every curve in Figs 5 and 7 starting at Pr=0.

Specific, falsifiable predictions (paper's own numbers, read from the PDF):
- Scenario 2: UnsecCred jumps to ~1.0 immediately after t=8; MITM reaches ~1.0
  by t=20; at t=31, SpoofRepMsg/CorrReact/UnstablePS are ~1.0/0.7/0.85; at t=52,
  UnauthCommand and UnstablePS reach ~1.0.
- Scenario 1 KL(EX||FF): UnstablePS peaks ~2e-2 around t=40-50 then decays;
  CorrReact rises and stabilizes ~0.11-0.12 (does NOT decay to 0); MITM peaks
  ~1.8e-2 early (t~10) then decays to ~0 by t~50.
- Scenario 2 KL(EX||FF) for UnstablePS: two humps, ~1.15e-2 near t=31 and
  ~0.6e-2 near t=52, then to 0.
- FF latency ~0.03s/slice, order of magnitude (paper's own MATLAB/BNT figure;
  my Python/pgmpy implementation is expected to differ in absolute terms).

Pre-registered expected discrepancy (CLAUDE.md rule 3 — stated before running,
not excused after): EX's memory footprint here will be far below the paper's
~5054MB, because a correctly implemented EX cluster is one 8192-entry float64
array (~64KB) plus one CPD of the same size — the paper's MATLAB/BNT figure
reflects that toolbox's junction-tree overhead, not an inherent cost of exact
inference on 13 binary interface nodes. This is expected and not a correctness
signal.

Design note carried into this experiment: a Plan-agent review of my first draft
caught that MITM is NOT conditionally independent of UnsecCred/ModAuthProc given
only the anterior layer, because MITM's parent CredAccess is intra-slice (doesn't
self-loop). The filtering engine is built on pgmpy VariableElimination reusing
attach_cpds()'s tested CPTs directly, specifically to avoid a hand-rolled
independence assumption that would have silently zeroed out MITM's KL(EX||FF)
curve. tests/test_inference.py's second toy case is a hand-derived check against
exactly this structure (UnsecCred/ModAuthProc/CredAccess/MITM).

**Result:** GATE FAILED on first run (m=1.0, the value carried over from
`configs/base.yaml` and used unchanged since the parameterization session).
`experiments/exp01_reproduce_paper.py`, Scenario 2 / EX:

| check | got | target | result |
|---|---|---|---|
| UnsecCred @t=8 | 0.9986 | ~1.0 | PASS |
| MITM @t=20 | 0.3333 | ~1.0 | FAIL |
| SpoofRepMsg @t=31 | 0.9594 | ~1.0 | FAIL (close) |
| CorrReact @t=31 | 0.0001 | ~0.7 | FAIL |
| UnstablePS @t=31 | 0.0400 | ~0.85 | FAIL |
| UnauthCommand @t=52 | 0.9449 | ~1.0 | FAIL (close) |
| UnstablePS @t=52 | 1.0000 | ~1.0 | PASS |

Scenario 1 KL(EX‖FF): UnstablePS peaked 0.248 at t=137 (target ~2e-2 at
t≈40-50); CorrReact plateaued at 0.436 (target ~0.11-0.12); MITM peaked 0.455
at t=35 (target ~1.8e-2 at t≈10). All off by roughly an order of magnitude or
more, all peaking/plateauing later and higher than the paper's curves.

**Interpretation — investigated, root cause identified, not silently patched.**
`tests/test_inference.py` passed a hand-derived closed-form check before this
run (independent verification against pgmpy `VariableElimination`, not just
internal self-consistency), so the *filtering engine* was already trusted
going in. The failure pattern here — everything correlated with MITM's own
transition converging far too slowly — pointed at parameterization, not
inference, so I investigated `compute_delta_t`'s input rather than
`inference.py`'s logic.

`collect_uniformization_ttcs` (written last session) sums `1/T_bar` over all
11 timed attack-step nodes, giving `Σ=14.6117` and `delta_t=0.06844` at m=1.
Cerotti et al. Table 5 independently publishes `delta_t` in seconds for six
different values of m (1/3, 1/2, 1, 5, 10, 20). Converting each to the paper's
own time unit (600s) and solving `Σ = 1/(m·delta_t)` gives **the same Σ≈3.6117
at every single m** (variation only at the 4th significant digit, consistent
with Table 5's 2-decimal rounding) — this is strong, direct evidence for the
Σ Table 5 actually used, independent of any curve-reading. My Σ=14.6117 is
about 4x too large, so my delta_t is about 4x too small, so every p_s in the
model is about 4x too small, which is exactly the "everything converges too
slowly" pattern observed.

I could not uniquely determine *which* subset of TTCs produces Σ=3.6117 from
the paper's Sec. III-E text alone — a brute-force search over all 2^11 subsets
found a large tied family (several structurally unrelated node-sets landing on
the identical sum, an artifact of the round-number TTCs in Table 3, e.g.
1/2+1/2+1/2 coincidentally equalling 1/(1/3)+1/2). The paper's own words —
"such as the initial concurrent attack techniques 'ModCtrlLogic', 'UnsecCred'
and 'ModifyProgram'" — read most naturally as an illustrative example of why
Eq. 3 generalizes to a sum, not a closed definition of Σ's scope, so I can't
resolve this from text alone either.

**Diagnostic (not a fix — a single ad hoc run to test the hypothesis in
isolation, not committed to any file):** re-ran Scenario 2 / EX with
`collect_uniformization_ttcs` unchanged but `m` chosen so `delta_t` exactly
matches Table 5's published value (`m≈0.2472` instead of `1.0` — since delta_t
depends only on the *product* `m·Σ`, matching delta_t this way is equivalent
to matching Σ directly, and every node's `p_s=delta_t/T_bar` comes out
identical regardless of which subset "really" produced that delta_t). Result:
MITM @t=20 rose from 0.333 to 0.806 (much closer to ~1.0, order of magnitude
right); SpoofRepMsg @t=31 rose from 0.959 to 0.994. Most strikingly,
**CorrReact jumped to exactly 0.7000 at t=32** — the precise signature of a
single first-shot `Bernoulli(0.7)` draw (Table 1's rule: precondition newly
met, self not previously active → `P(active)=p_s`, and CorrReact's `p_s` is
the paper's own fixed 0.7) — landing one time slice after the paper's stated
t=31, with UnstablePS similarly following one slice behind its ~0.85 target
(0.747 at t=32, 0.925 at t=33 — 0.85 falls almost exactly between them). This
is consistent with a plain off-by-one indexing convention between my `t` (my
`step()` call number, 1-indexed) and the paper's `t`, on top of the ~4x delta_t
error being the dominant effect.

This is a real, well-evidenced bug, but it is in `collect_uniformization_ttcs`
(and by extension last session's `parameterization.py`, already committed and
tested), not in this session's `inference.py`/`metrics.py`, which check out
independently against hand-derived math. Per the task's explicit instruction
and CLAUDE.md rule 3, I am **stopping here** rather than silently changing
`collect_uniformization_ttcs`'s scope and re-running until the gate passes —
that would be adjusting the implementation to match a target I can't yet fully
justify from the paper text, which is exactly what rule 3 forbids. Reported to
the user with the full diagnostic; the exact correct Σ scope needs sign-off
before `parameterization.py` changes.

**Surprised?** yes. I expected either a clean pass or a clearly-broken curve;
instead the diagnostic showed the SAME model, engine, and code, off by a
close-to-exactly-explicable magnitude factor (Σ scope) plus a close-to-exactly-
explicable one-slice offset (indexing convention) — strong evidence the model
and engine are fundamentally sound and this is a parameterization/calibration
bug rather than a structural one, but I don't yet have enough to be certain
which TTC subset (or exact indexing fix) is correct, so I'm not calling it
resolved.

**Addendum, same day — calibrated rerun (delta_t_override, user-confirmed):**
Added `InferenceConfig.delta_t_override` / `attach_cpds(..., delta_t_override=)`
(additive, default `None`, existing 22 tests unaffected) and reran the full gate
with `delta_t_override=166.13/600` (Table 5's m=1 value). Result: **4/7 checks
now PASS** (UnsecCred@8, SpoofRepMsg@31, UnauthCommand@52, UnstablePS@52) vs 2/7
before. **3 still FAIL**: MITM@20 (0.8061), CorrReact@31 (0.0001, but exactly
0.7000 one slice later at t=32), UnstablePS@31 (0.1523, 0.7471 at t=32, 0.9245
at t=33 — 0.85 falls *between* slices, not at either).

Per the user's direction, investigated whether the 3 remaining failures are a
single uniform one-slice offset. **They are not.** CorrReact's failure is
cleanly a one-slice lag with an exact landing (0.7000 at t=32) — well
explained (Table 1's rule: a fresh precondition, self not previously active,
gives P(active)=p_s exactly on the first opportunity, and CorrReact's p_s
*is* the paper's fixed 0.7). UnstablePS partially fits the same story but
doesn't land exactly on 0.85 at any single t. MITM's gap does not fit a
one-slice shift at all: it climbs gradually via its own p_s from CredAccess's
resolution (~t=9) through t=30 (0.9452), then jumps sharply to 0.9997 at
**t=31 itself** — coincident with SpoofRepMsg's own evidence-forced jump, not
one slice after it. This jump is not obviously a bug: SpoofRepMsg's
precondition is MITM at t-1, so MeasureCoherence=1 evidence at t=31 (tiny
p_neg) makes it near-certain MITM was already active at t=30, and MITM's own
persistence then keeps it there — an "explaining away" correlation exact
inference is *supposed* to capture. Whether the paper's own curve shows this
same t=31-not-t=20 jump, or genuinely reaches ~1.0 by t=20 through some
mechanism this model doesn't have, I cannot determine from the published
figures alone at this resolution.

**Status: gate still not passing.** The dominant bug (Δt magnitude) is fixed
and well-evidenced. Two smaller, distinct, only-partially-understood
discrepancies remain (CorrReact/UnstablePS's slice-level timing, MITM's
convergence rate near t=20). Reported to the user rather than continuing to
iterate further without checking in.

---

## 2026-07-31 Experiment: exp01 run 3 -- reaction semantics + t-axis fixes

**Hypothesis:** I expect the two bugs found by investigating run 2's failures
to fix the remaining probability checks. (a) Reactions were modelled as
persistent self-looping interface nodes; Table 3 gives them TTC=0 and Fig. 5a
shows CorrReact flat at exactly 0.7, which a persistent node with p_s=0.7
cannot do (it ratchets to 1.0). (b) The paper's t axis is in TIME UNITS
("t (x 10 min)"), not DBN slices; at delta_t=166.13 s that is 3.6117 slices
per unit, so T=200 is 722 slices. I expect (b) alone to fix MITM@20, and (a)
to fix CorrReact@31.

**Result:** **6/7 probability checks PASS** (was 2/7, then 4/7).

| check | got | target | result |
|---|---|---|---|
| S2 @t=8 UnsecCred | 1.0000 | ~1.0 | PASS |
| S2 @t=20 MITM | 0.9941 | ~1.0 | PASS |
| S2 @t=31 SpoofRepMsg | 0.9947 | ~1.0 | PASS |
| S2 @t=31 CorrReact | 0.6963 | ~0.7 | PASS |
| S2 @t=31 UnstablePS | 0.8088 | ~0.85 | FAIL |
| S2 @t=52 UnauthCommand | 0.9859 | ~1.0 | PASS |
| S2 @t=52 UnstablePS | 0.9980 | ~1.0 | PASS |

Latency now 0.0158 s/slice (paper ~0.03 s) -- right order, gate satisfied.

Two independent confirmations that the core math is correct:
- ModCtrlLogic matches the analytic exponential CDF 1-exp(-t/50) to 4 decimals
  at t=10/25/50 (0.1812/0.3933/0.6340 vs 0.1813/0.3935/0.6321). The
  discretized geometric converges to the continuous exponential exactly as
  Sec. III-E requires.
- Scenario 1 FF-vs-EX differs by up to 0.13, so the clustering approximation is
  genuinely doing work; this is not a degenerate "both configs identical" run.

**Interpretation:** Both bugs were real and both are fixed. Run 2's data was
never wrong -- its *gate* was reading slice indices as time units. Two
discrepancies remain, and they are different in kind:

1. **UnstablePS@31 = 0.8088 vs the text's 0.85 -- the paper contradicts
   itself here, and my number matches its figure.** Fig. 7b (read at 4x zoom)
   plots the t=31 jump at ~0.81, then a gradual climb to ~0.855 just before
   t=52. Closed form: UnstablePS = 1-(1-0.7)(1-0.8*(1-exp(-t/50))) gives
   0.8109 at t=31 and 0.8499 at t=49.0. So the text's "0.85" is the value at
   the END of the post-t=31 plateau, not at its start. I have left the gate
   checking the text's 0.85 (so it reports FAIL) rather than switch to the
   reading that makes me pass -- CLAUDE.md rule 3. Which is authoritative is
   the user's call, not mine.

2. **The KL checks moved the WRONG way, and this traces to the same reaction
   decision -- the paper's own figures are in sharp conflict.**
   - Fig. 5a (CorrReact flat at exactly 0.7) requires a MEMORYLESS reaction.
   - Fig. 6c (CorrReact KL stabilises just below 0.12 and explicitly "is not
     able to converge to the exact solution") and Fig. 8a (S2 UnstablePS KL
     peak ~1.15e-2) require a reaction that CARRIES STATE, since a memoryless
     one makes FF and EX converge to the same 0.7 and the KL decay to 0.

   Measured: S1 CorrReact KL plateau 0.0353 (target 0.11-0.12); S2 UnstablePS
   KL 1.6e-16 (target 1.15e-2). The S2 value is machine epsilon, and I can
   explain it structurally: ModCtrlLogic's branch is disjoint from the
   centre/right branches, and continuous Table-4 evidence pins everything else,
   so with memoryless reactions there is no cross-branch correlation left for
   FF to discard. The paper's model must retain correlation mine does not.

   **Candidate reconciliation, NOT implemented and NOT verified:** a "one-shot
   latch" reaction -- the control centre gets exactly one chance to react when
   the spoof occurs, succeeding w.p. 0.7, and that outcome then persists. This
   plateaus at exactly 0.7 (satisfying Fig. 5a) while carrying state across
   slices (potentially satisfying Figs. 6c/8a). It cannot be expressed by a
   binary Table-1-style CPT, because such a CPT cannot distinguish "precondition
   just became true" from "precondition has been true for a while"; it needs a
   3-state node or an auxiliary latch variable. That is a structural change
   beyond this session's scope, so per CLAUDE.md rule 6 I am stopping to ask
   rather than implementing it unilaterally.

**Surprised?** yes, twice. First that the paper's text and its own Fig. 7b
disagree on UnstablePS@31 -- I checked by rendering the figure at 4x and
deriving the closed form independently, and both agree with each other and
against the text. Second that fixing the reaction semantics improved every
probability check while making the KL checks worse; I expected one consistent
direction. Checked that this is not a degenerate run (S1 FF/EX differ by 0.13)
and traced the S2 near-zero KL to a specific structural cause rather than
assuming a bug. The conflict appears to be in the source material, not in the
implementation -- but I cannot rule out a third reading of the reaction
semantics that satisfies all three figures at once, so I am not claiming the
paper is wrong, only that these two readings are mutually exclusive.

**Gate verdict: NOT PASSED.** 6/7 probability checks pass; 1 fails against the
text (matches the figure); KL checks do not reproduce. Not proceeding to
Session 3.

---

## 2026-07-31 Experiment: one-shot latched reactions (hypothesis test)

**Hypothesis:** The paper's figures are mutually inconsistent under any
memoryless reaction: Fig. 5a's flat-0.7 plateau requires no memory, but Fig. 6c
states the FF divergence for CorrReact "is not able to converge to the exact
solution (it stabilizes just below 0.12)", which requires memory. I predicted a
ONE-SHOT LATCHED reaction satisfies both -- the control centre gets exactly one
chance to react when its precondition first holds, succeeding w.p. 0.7, and
that outcome persists. Marginal is 0.7 x P(precondition ever held), so it
plateaus at exactly 0.7 (Fig. 5a) while carrying state, so FF's independence
assumption should incur a PERMANENT error (Fig. 6c).

First: is the latch even necessary? "Precondition holds now AND did not hold
last slice" is a t-2 dependency, and a 2TBN is Markov order 1, so it cannot be
expressed in the reaction node alone. An explicit auxiliary latch node is
required. That is a structural addition NOT drawn in Fig. 2, so it is
implemented behind `build_attack_graph(reaction_mode=...)` with "memoryless"
remaining the default; nothing about the previous runs changes.

**Result:** Implemented (`build_latch_cpt`, `build_latched_reaction_cpt`, +6
tests, 31/31 pass). Scenario 1, delta_t = 166.13/600:

| model | EX CorrReact | FF CorrReact | KL(EX\|\|FF) | converges? |
|---|---|---|---|---|
| memoryless | 0.700 | 0.700 | -> 0 (max 0.0353) | yes -- CONTRADICTS Fig. 6c |
| latched | 0.700 | **0.5079** | **0.0761, stable** | **no -- MATCHES Fig. 6c** |
| paper Fig. 6c | -- | -- | "just below 0.12" | no |

FF-latched CorrReact is flat at 0.5079 from t=20 through t=60 while
SpoofRepMsg keeps climbing (0.599 -> 0.973), i.e. the error is genuinely
permanent, not a transient.

Correction to a claim made when first reporting this: I stated the EX-latched
plateau was 0.7 "by construction" and, sharpening it, that CorrReact /
SpoofRepMsg should be exactly 0.7 at every t. Measured, it is not -- the ratio
converges to 0.7 FROM BELOW (0.4498 at slice 8, then 0.6173, 0.6571, 0.6734,
0.6818 at slice 40). Cause: the one-shot fires the slice AFTER its precondition
turns on, so CorrReact(t) = 0.7 x P(SpoofRepMsg active by t-1) while
SpoofRepMsg(t) = P(active by t); with P still rising the ratio lags below 0.7
and only closes as P saturates. The asymptotic 0.7 plateau -- which is what
Fig. 5a shows and what this finding rests on -- does hold. The exact-at-every-t
version did not, and was asserted from derivation rather than measurement.

**Interpretation:** The latched reading reproduces the qualitative behaviour the
paper explicitly describes and the memoryless reading cannot: a divergence that
stabilises instead of decaying to zero. That is a real structural finding -- it
says the paper's reactions must carry state, which in turn means Fig. 2's
self-loops on CorrReact/WrongLogicExec are meaningful and Table 3's TTC=0 does
NOT mean "memoryless". The two readings are now distinguished by evidence
rather than by preference.

The magnitude is still short: 0.076 vs ~0.115, about 66%. So the latched model
as I have specified it is closer to the paper's but is not identical to it. I
did not tune anything to close that gap, and will not -- the remaining
difference is a real, reported discrepancy, not something to fit away.

**Not run:** the full 4-config gate in latched mode. Latched inference costs
~1.03 s/slice for FF (vs 0.015 s memoryless) because the extra coupling worsens
VE's elimination order, and EX is far slower again at 2^15 = 32768 joint states;
the full sweep is hours. Since the latch is a HYPOTHESIS about undocumented
paper internals rather than something the paper states, spending that compute
to chase a number was not obviously justified without checking in first.

**Surprised?** no on direction, yes on cleanliness. I expected the latched model
to produce some permanent gap; I did not expect FF to lock to a single value
(0.5079) that stably from t=20 onward. Checked it was not a stuck computation
by confirming SpoofRepMsg continues to evolve over the same slices.

**Gate verdict unchanged: NOT PASSED.** The latched result explains WHY the KL
checks failed and rules out the memoryless reading as the paper's model, but
does not by itself reproduce the published magnitudes.

---

## 2026-07-31 Experiment: exp02, full Scenario 1 KL under latched reactions

**Hypothesis:** Following the CorrReact result above, I expected the latched
reading to move all three Scenario 1 KL checks toward the paper's Fig. 6
values, since it is the reading that reproduces the paper's own stated
qualitative behaviour (a divergence that stabilises rather than decaying).

Scope: Scenario 1, FF and EX, 250 slices (t~69) rather than the paper's 722.
The paper's Scenario 1 KL features all occur early (MITM t~10, UnstablePS
t~40-50), so this window contains every feature under test at ~1/3 the compute
of the full horizon (EX-latched costs ~7.5 s/slice at 2^15 states). Behaviour
past t~69 is NOT measured by this run; that is a budget choice, stated rather
than implied.

**Result:** the hypothesis is WRONG. The latched reading does not uniformly
improve the KL checks -- it fixes one and badly breaks another.

| node | paper Fig. 6 | memoryless | latched |
|---|---|---|---|
| UnstablePS | peak ~2e-2 @ t 40-50 | 0.0273 @ t 6.6 | **0.292 @ t 8.3** |
| CorrReact | ~0.12, does NOT converge | 0.0353, converges to 0 | **0.1256 pk -> 0.069, does not converge** |
| MITM | ~1.8e-2 @ t 10 | 0.0075 @ t 1.9 | 0.0075 @ t 1.9 |

- CorrReact: latched now MATCHES the paper on both magnitude (0.1256 vs "just
  below 0.12") and the qualitative non-convergence. Memoryless cannot.
- UnstablePS: latched is **15x TOO HIGH** (0.292 vs ~2e-2). Memoryless was
  close in magnitude (0.0273). This is the reverse of the CorrReact result.
- MITM: unchanged between readings, as expected -- MITM sits upstream of both
  reactions, so the reaction semantics cannot affect it. Its ~2.4x shortfall
  (0.0075 vs 1.8e-2) is therefore a separate, still-unexplained discrepancy,
  not attributable to this modelling choice either way.

**Interpretation:** Neither reaction reading reproduces Cerotti et al. Fig. 6.
Each satisfies a different subset of the published curves and contradicts the
rest. The UnstablePS blow-up under latching has a clear mechanism: UnstablePS
is an OR over WrongLogicExec, CorrReact and UnauthCommand, and latching gives
BOTH reactions a permanent FF error, which compounds through the OR rather than
cancelling. So the same property that makes CorrReact match makes UnstablePS
diverge.

A separate signal worth recording: under BOTH readings, every KL peak lands
early (t~2-9) against the paper's t~10-50. This is NOT a global time-scaling
error, because the probability timeline is independently correct -- exp01's
UnsecCred@8, MITM@20, SpoofRepMsg@31 and UnauthCommand@52 all match. So the
posteriors are correctly timed while their FF/EX disagreement peaks too early,
which points at the clustering/approximation dynamics rather than at delta_t.
I do not have an explanation for this and am not going to invent one.

**Surprised?** yes. I expected latching to move all three checks the same
direction, and pre-registered that expectation above. It did not: it fixed
CorrReact and broke UnstablePS by an order of magnitude. Checked the UnstablePS
number is not a bug by confirming the mechanism (compounding permanent errors
through the OR gate) and that MITM -- structurally upstream of the reactions --
is bit-identical between the two readings, which is what it should be if the
change is doing only what it claims to.

**Conclusion: the published Scenario 1 KL curves are not reproducible from the
paper's description under either reaction reading I have tested.** I am not
going to keep enumerating readings until one fits; that would be fitting to the
target rather than deriving from the source. Recorded as an open discrepancy.

**CORRECTION, same day.** The line above ("latched is 15x too high on
UnstablePS") was drawn from M_KL, a max-over-t summary, and that statistic hid
the actual structure. Two follow-up checks, both from data already on disk:

1. `binary_kl` verified against hand arithmetic and scipy to 1e-12, including
   the Eq. 4 argument order (EX is P, FF is Q; the metric is asymmetric). The
   KL discrepancy is NOT a metric bug. Regression tests added
   (tests/test_metrics.py).

2. Decomposing UnstablePS's KL over its three OR-parents under the MEMORYLESS
   model gives KL_WrongLogicExec identically 0 at every t. Reason: memoryless
   computes WrongLogicExec intra-slice from ModCtrlLogic, and that sub-process
   is structurally disjoint from the centre/right branches, so FF has no
   correlation to lose there. The whole UnstablePS divergence comes from
   CorrReact and UnauthCommand, which share MITM as a common ancestor; MITM
   saturates fast (TTC=2), so that correlation dies by t~10 and the KL decays.
   That is why the memoryless peak lands at t~7 and nothing appears near
   t~40-50.

   Fig. 6a's peak sits at t~40-50, which is precisely when ModCtrlLogic
   (TTC=50) is mid-range. For that branch to contribute at all,
   WrongLogicExec must be a persistent INTERFACE node so FF separates it from
   ModCtrlLogic into a different cluster. That is more evidence the paper's
   reactions carry state -- and it is a mechanism, not a curve-fit.

3. Re-reading the LATCHED UnstablePS KL trajectory (not just its max): it is
   TWO-HUMPED. Large early peak 0.292 at t=8.3, decaying to 0.00084 by t=40,
   then RISING again -- 0.0036 at t=50, 0.0053 at t=60, 0.0059 at t=69, still
   climbing when the 250-slice run stopped. The second hump is the
   ModCtrlLogic branch, in the right place for Fig. 6a.

So the accurate statement is NOT "latched breaks UnstablePS". It is: latched
ADDS the feature Fig. 6a describes (a ModCtrlLogic-driven hump at t~40+) which
memoryless cannot produce at all, while ALSO producing a large early spike that
Fig. 6a does not show. My t~69 truncation was too aggressive and cut off the
hump I was trying to measure. Re-running to t~120.

**Surprised?** yes, and it was a self-inflicted error: I reported a max-over-t
statistic as if it characterised a curve, and it concealed a two-hump shape
that changes the interpretation. Checked by plotting the trajectory rather than
re-reading the summary.

**Extended run (t~120, slice 433) -- second hump resolved.** It is a real
peak-and-decay, not a monotone rise:

    t=39.9 KL=0.00084   t=70.1 KL=0.00590   t=100.0 KL=0.00472
    t=50.1 KL=0.00362   t=80.0 KL=0.00581   t=119.9 KL=0.00341
    t=60.1 KL=0.00531   t=90.0 KL=0.00534
    local maxima: (t=8.3, 0.29203) and (t=73.1, 0.00593)

Against Fig. 6a's "peak ~2e-2 around t~40-50": the second hump is **3.4x too
low** (0.00593 vs ~0.02) and peaks **~25 time units late** (t=73.1 vs t~40-50).
For reference, ModCtrlLogic is mid-range (P=0.5) at t = 50*ln2 = 34.7, which is
about where the paper's peak sits; mine lags well past that.

**Final position on the reaction question.** Neither reading reproduces Fig. 6:

| | memoryless | latched |
|---|---|---|
| CorrReact, magnitude | 0.0353 vs ~0.12 | **0.1256 vs ~0.12 -- matches** |
| CorrReact, converges to 0? | yes -- contradicts paper | **no -- matches paper** |
| UnstablePS, ModCtrlLogic-driven hump | **absent entirely** (KL for that branch is identically 0) | **present**, but 3.4x low and 25 units late |
| UnstablePS, spurious early spike | none | **0.292 at t=8.3, not in Fig. 6a** |
| MITM | 0.0075 vs 1.8e-2 | identical (upstream of reactions) |

Latched is closer on mechanism -- it is the only reading that can produce the
ModCtrlLogic-driven hump Fig. 6a shows at all, and it matches Fig. 6c's
CorrReact magnitude and non-convergence. But it also produces a large early
spike the paper does not show, and its hump is off in both height and timing.

**Stopping the search here.** Every remaining move I can think of (tuning the
latch timing, altering which nodes enter delta_t, adjusting cluster membership)
would be selecting a model by how well its output matches a target curve, which
is what CLAUDE.md rule 3 exists to prevent. Recorded as an open discrepancy
against the source rather than resolved by fitting.

**Gate verdict: NOT PASSED.** 6/7 probability checks pass (exp01, memoryless,
and the 1 failure matches Fig. 7b while contradicting the paper's own text).
KL curves do not reproduce under either reaction reading.

---

## 2026-08-01 Plot inspection: what actually reproduced

Inspected the generated figures rather than only the scalar checks (they were
produced but never looked at, which is the same unchecked-deliverable failure
as reporting an unverified number).

**Both probability figures are curve-for-curve matches to the paper.**

`exp01_scenario2_probs_b.png` vs Fig. 7b: UnsecCred steps to 1 at t=8; MITM
saturates by t~20; UnstablePS climbs to ~0.37, jumps to ~0.81 at t=31, climbs
to ~0.855, then to 1 at t=52; UnauthCommand flat at 0 until t=52. Every
feature, in the right place. It also independently confirms the text-vs-figure
finding recorded above: the plotted t=31 jump is ~0.81, not the text's 0.85.

`exp01_scenario1_probs_a.png` vs Fig. 5a: ModAuthProc and ModifyProgram
saturate almost immediately; SpoofRepMsg reaches ~1 by t~75-100; **CorrReact
plateaus at exactly 0.7**; ModCtrlLogic climbs slowly to ~0.98 at t=200.

**Interpretation, and a scope observation I should have made earlier.** What
reproduces is the DBN's posterior behaviour -- structure, CPTs, uniformization,
evidence conditioning, forward filtering -- in both scenarios, essentially
exactly. What does not reproduce is the KL(EX||FF) analysis, which measures the
error of the BK *approximation*, not the model. Those are different claims: one
is about whether the model is right, the other about how badly a particular
clustering approximates it.

CLAUDE.md is explicit that the second is not load-bearing here: "The source
paper's own finding is that FF ~= CL. Therefore a clean FF implementation may be
all that is ever needed. Do NOT sink weeks into full Boyen-Koller clustering
optimization -- that direction was explicitly evaluated and rejected for this
project", and separately lists BK-clustering work under deliberately rejected
directions because "headroom is ~10^-2 KL ... this optimizes a solved problem".

So the unreproduced part is precisely the part this project has already decided
not to build on, and the reproduced part is the part every downstream phase
(closed loop, learned parameterization, adversarial robustness) actually
depends on. I am NOT claiming this passes the gate -- the gate's KL checks were
specified explicitly and they fail. I am recording that the failure is confined
to a component the project treats as out of scope, so the decision about
whether to proceed is better informed.

**Gate verdict: NOT PASSED.** 6/7 probability checks pass (exp01, memoryless).
KL checks do not reproduce under either reaction reading. Not proceeding to
Session 3 without a decision on how to treat this.

---

## 2026-08-01 Provenance gap found and fixed (no experiment re-run)

Not a new experiment -- a correctness fix to logging infrastructure, recorded
because it changes how every CSV produced this session should be read.

`git_sha()` in both experiment scripts shelled out to `git rev-parse HEAD`,
which reports the last COMMIT regardless of uncommitted changes. Every run
this session executed against a dirty working tree (all of this session's
fixes -- reaction latch, delta_t override, slice/time-unit fix -- were
uncommitted), so every summary CSV's `git_sha` column names the PARENT commit
(`1abbf37`), not the code that actually produced its numbers. That silently
violates CLAUDE.md rule 4's intent even though a SHA was, literally, logged.

Fixed: `src/eval/provenance.py::git_sha(repo_root)`, shared by both experiment
scripts (deduplicating the identical function each previously defined),
appends `-dirty` when `git diff-index --quiet HEAD --` reports changes.
Verified it correctly reports `1abbf37...-dirty` against the current tree.

**All summary CSVs already on disk from this session's experiments (exp01
runs, exp02 runs) should be read as: produced from a dirty tree based on
commit 1abbf37, not as exact commit-level provenance.** Not re-running the
experiments solely to regenerate them with corrected SHAs -- the numbers
themselves are unaffected by this fix, only their logged provenance string
would change, and exp01's EX runs cost real time. If a clean-provenance run is
needed later, the fix is already in place for the next run to pick up
automatically once the code is committed.

---

## 2026-08-01 Decision: validation gate verdict for this phase

Recorded as a decision, not an experiment: the user reviewed the investigation
above and made the call on how to treat it.

**Verdict: PASSED on the load-bearing dimension. KL(EX||FF) logged as an open,
documented discrepancy, not blocking.**

Reasoning (user-endorsed):
- Posterior reproduction -- structure, CPTs, uniformization, evidence
  conditioning, forward filtering -- matches the paper curve-for-curve in both
  scenarios (2026-08-01 plot inspection entry above). This is the DBN causal
  core, and it is what C1/C2/C3 (LAB_NOTEBOOK's actual research claims) depend
  on.
- What does NOT reproduce is KL(EX||FF), which measures the accuracy of the
  Boyen-Koller clustering APPROXIMATION, not the model. CLAUDE.md explicitly
  de-scopes this: "Do NOT sink weeks into full Boyen-Koller clustering
  optimization... FF ~= CL... this optimizes a solved problem" -- the exact
  category of work this discrepancy sits in.
- Two real implementation bugs were found and fixed in the process (reaction
  semantics, slice/time-unit indexing), a third calibration gap was patched
  and documented (delta_t via Table 5 override, exact TTC subset unresolved),
  and two of my own analysis errors were caught and corrected (a degenerate
  test, a max-over-t statistic misread as a curve). All of that stands
  regardless of the KL verdict.

This is a documented exception, not a silent pass -- CLAUDE.md rule 3 ("If a
validation gate fails, STOP and report the discrepancy... do not adjust the
target") was followed: the discrepancy was found, investigated to a stopping
point, and reported before this decision was made, not glossed over to reach
it.

Proceeding to commit this session's work and to Phase 2 planning.

---

## 2026-08-01 Experiment: digital twin, open loop (exp03)

**Hypothesis.** I expect the twin to run end to end producing traces that
satisfy the structural invariants by construction (precondition ordering,
analytic attribution, measurable physical effect), and I expect its
twin-driven posteriors to reach the same terminal state as the paper's
scripted Scenario 2 but on a SUBSTANTIALLY FASTER timeline.

Falsifiable timing prediction, computed from Table 3's TTCs by Monte Carlo
(200k draws) BEFORE running anything -- this is arithmetic from published
config values, not a result:

| analytic | twin mean | Table 4 | Table 4's percentile in the twin distribution |
|---|---|---|---|
| FileAccess | 1.00 | 8 | ~100th (P(X>8) ~ 4e-9) |
| FileIntegrity | 1.00 | 9 | ~100th |
| MeasureCoherence | 18.31 | 31 | 84th |
| CommandCoherence | 42.39 | 52 | 71st |

I expect this divergence and do NOT intend to close it. Fig. 7's own caption
reads "second scenario with a slow attack and randomized times", so Table 4 is
explicitly not a typical draw. The sharp, genuinely interesting part of the
prediction is the ASYMMETRY: the two late analytics (31, 52) are plausible
draws from the twin (84th/71st percentile), while the two early ones (8, 9)
are essentially impossible under the paper's own Table 3 TTCs. If that holds,
Scenario 2's credential-theft timing is not reconcilable with the TTCs that
parameterize the very same model -- a finding about the source, not about my
implementation. Tuning firing delays to hit Table 4 would be fitting to a
target (CLAUDE.md rule 3) and is explicitly not being done.

Second pre-registered prediction: a systematic +delta_t/2 ~ 0.14 time-unit
discretization offset per step (continuous completion time mapped to
ceil(t/delta_t)), accumulating along chains to ~0.4 for the 3-stage UnsecCred.
Small, one-directional, and expected -- logging it now so it is not later
mistaken for a bug. The underlying means are unbiased (Geometric with
p = delta_t/T_bar has mean T_bar time units); only slice-boundary rounding
shifts.

**Instability threshold.** 0.90 / 1.10 pu, read at runtime from
`net.bus.min_vm_pu` / `max_vm_pu`. This is NOT a value I chose: it is
case33bw's own declaration, and it coincides with EN 50160's +/-10% band for
European distribution networks (the paper's setting is Italian, CEI 0-16).
A test asserts the limits come from the network rather than from a literal.
Measured headroom at the nominal ladder level (0.8 MW/DER): vmin 0.9611,
vmax 1.0000 -- comfortable on both sides; the destabilising level (3.0 MW/DER)
gives vmax 1.1311, a real violation.

**Stated limitation, up front.** The twin samples attack delays from the
continuous-time law the DBN's CPTs discretize, and samples analytics from the
DBN's own Table-2 likelihood with the DBN's own p_pos/p_neg. Agreement between
twin-driven and scripted posteriors is therefore close to guaranteed by
construction. **exp03 is a plumbing validation and an envelope measurement, not
evidence that the DBN models reality.** The `deterministic` delay-law arm
exists precisely to make that honest: it measures degradation when the twin's
law is NOT the DBN's.

**Result: GATE PASSED.** 64/64 twin tests; 47/47 Sessions 1-2 tests still green.
All five invariants pass: (a) precondition ordering clean across 40 replicates
(2 arms x 20), (b) analytic attribution exact, (c) physical effect measurable
(nominal vmax 1.0000 -> compromised 1.2336 against a 1.10 limit), (d) posterior
rises over the horizon, (e) evidence-stream scope guard holds.

Stage-0 ladder sweep (now with logged provenance, `exp03_grid_sweep_*.csv`),
DERs derived at buses {17, 32}:

| p_mw/DER | vmin | vmax | n_violated | unstable |
|---|---|---|---|---|
| 0.0 | 0.9131 | 1.0000 | 0 | no |
| 0.8 | 0.9611 | 1.0000 | 0 | no |
| 2.0 | 0.9837 | 1.0699 | 0 | no |
| 3.0 | 0.9893 | 1.1311 | 3 | YES |
| 5.0 | 0.9961 | 1.2336 | 13 | YES |

**Prediction 1 (timing) held, and the ASYMMETRY held in the sharp form.**
Measured median raising times (exponential arm, 20 replicates) vs Table 4:

| analytic | twin median | twin p10-p90 | Table 4 | Table 4 inside p10-p90? |
|---|---|---|---|---|
| FileAccess | 0.97 | 0.55-1.69 | 8 | **NO** (far above) |
| FileIntegrity | 0.97 | 0.28-1.97 | 9 | **NO** (far above) |
| MeasureCoherence | 11.49 | 5.43-34.67 | 31 | **yes** |
| CommandCoherence | 33.64 | 6.65-131.69 | 52 | **yes** |

Predicted medians were 1.00 / 1.00 / 18.31 / 42.39; measured 0.97 / 0.97 /
11.49 / 33.64 (medians run below means, as expected for right-skewed sums of
exponentials). The two LATE analytics' Table 4 values are ordinary draws from
the twin; the two EARLY ones are not reachable under the paper's own Table 3
TTCs.

**Prediction 2 (discretization offset) held exactly.** Every deterministic-arm
raising time equals ceil(t/delta_t)*delta_t to within 0.01: 1.0 -> 1.11,
18.0 -> 18.27, 43.0 -> 43.19, 2.0 -> 2.22. Offsets 0.11-0.27, bounded by
delta_t = 0.277 as predicted. The deterministic arm also reproduced the
precondition arithmetic exactly (UnsecCred 3x1/3 = 1, MITM +2 = 3,
SpoofRepMsg +15 = 18, UnauthCommand min(3,4)+40 = 43).

**The most informative measurement, which max|diff| had obscured.** Comparing
the scripted Scenario 2 curve against the twin's p10-p90 envelope slice by
slice:

| node | scripted inside envelope | below p10 | above p90 |
|---|---|---|---|
| UnstablePS | 87.7% | 12.3% | **0.0%** |
| CorrReact | 89.0% | 11.0% | **0.0%** |
| MITM | 84.9% | 15.1% | **0.0%** |

The scripted scenario is inside the twin's distribution ~85-89% of the time,
and every single out-of-envelope slice is BELOW p10 -- never above. A
perfectly one-directional deviation. `max |diff|` (0.53-0.98) had made this
look like disagreement; it is not, it is a pure time shift with matching
shapes and identical plateau values (CorrReact plateaus at exactly 0.7 in both;
all three nodes reach identical terminal values).

**Open-loop baseline for Session 4** (`open_loop_lag_slices`, reported as a
descriptive statistic, NOT claimed as detection lead time and NOT claim C1):

| arm | median first-unstable slice | threshold | P(UnstablePS) crossing | lag |
|---|---|---|---|---|
| exponential | 44 | 0.50 | 38 | **-6** |
| exponential | 44 | 0.90 | 118 | +74 |
| exponential | 44 | 0.99 | 125 | +82 |
| deterministic | 66 | 0.50 | 66 | 0 |
| deterministic | 66 | 0.90 | 156 | +90 |

The threshold dominates the sign of the lag: at 0.5 the posterior crosses at or
before physical instability, at 0.9+ it trails by 74-90 slices. Sweeping rather
than picking a threshold was the right call -- a single choice would have
determined the headline number.

**Interpretation.** The twin reproduces the paper's Scenario-2 posterior SHAPES
and terminal values while running systematically faster, and the paper's own
Fig. 7 caption explains why: "second scenario with a slow attack and randomized
times". The one-directional envelope result quantifies that caption -- Table 4
is a slow draw, not a typical one. The genuinely new finding is the asymmetry:
Scenario 2's credential-theft times (t=8, 9) are not reachable under the Table 3
TTCs that parameterize the same model (twin p90 = 1.69 and 1.97), while its
later times (31, 52) are ordinary draws. So the early part of the paper's
scripted timeline is not reconcilable with its own TTCs; the later part is.
Nothing was tuned to close this.

**Surprised?** yes, on one point. I expected `max |diff|` to be the headline
comparison and it was actively misleading -- 0.53-0.98 reads as gross
disagreement, when the envelope analysis shows 85-89% containment with
zero one-sided violations. Checked by computing the envelope coverage directly
rather than trusting the summary scalar, which is the same lesson as the
2026-07-31 M_KL/two-hump correction: a max-over-t statistic does not
characterise a curve. Also mildly surprised the 0.5-threshold lag came out
NEGATIVE (posterior leads physics by 6 slices); that is a real open-loop
baseline worth beating in Session 4, not an artifact -- the posterior responds
to analytics that fire when the attack step completes, whereas instability
requires the control centre to then act on spoofed data.

**Stated limitation, restated.** Under the exponential arm the twin draws from
the DBN's own law and the DBN's own Table-2 likelihood, so agreement is close
to guaranteed. The deterministic arm is the honest control: it shifts every
raising time later (1.11/1.11/18.27/43.19 vs 0.97/0.97/11.49/33.64) and moves
the first-unstable slice from 44 to 66, yet the posterior still reaches the
same terminal values -- i.e. the DBN is robust to this particular
misspecification, which is a (weak) result rather than a tautology.
