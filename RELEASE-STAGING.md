<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# RELEASE-STAGING — terminals-worldmodel

An append-only ledger of release blocks. Each release is a finite, self-similar
block, not a free-form changelog. Every block carries the same three things and
is cut by the same three-part gate:

- **open** — what the world gets for free in this block.
- **moat** — what we keep, stated as a boundary in code, not a vibe.
- **gate** — `validate → converge → observe`. A block is only cut when all three
  hold: it is **validated** (tests / proofs green), it **converges** (the merge or
  rollout settles to a finite coherence-R), and it is **observed** (the win emits a
  replayable trace).

One idea runs through every block: **convergence is commodity, the constraint is
sold.** Anyone can run the open planner and watch candidates converge (that is the
`R` half — convergence *proposes*). The proprietary, sellable thing is the
reality-grounded constraint that *disposes* — which converged atoms actually cement
to truth. So the moat boundary is drawn explicitly in each block: the open half is
the runnable substrate, the kept half is the constraint that grounds it.

Newest block is at the top of the detail section; the ledger only grows downward.

## Ledger

| version | name | status | open | moat | gate |
|---|---|---|---|---|---|
| 0.4.0 | trained-predictor seam | planned | `predict(latent, action) -> latent` interface + loader contract + tiny runnable reference predictor — plug in your own model, run the full observe→simulate→rank→coherence-R loop with no planner/cost changes | trained predictor weights + the φ-curated curation/RSI loop (solve_history → LoRA SFT → measure → promote) that makes one predictor beat the no-model baseline | validate: a loaded predictor beats the coherence-only baseline on a held-out OS-trace delta (external verifier) · converge: receipt done=true / R=1 · observe: predict/score rows hit the causal chain |
| 0.3.0 | solver + demo + CI | planned | copy-pasteable trace→plan run on a real CEM/MPPI solver + green CI matrix (3.10–3.13) + parity test keeping the public package honest against the in-repo shim | none new — convergence (R, the coherence cost, swm plumbing) is the commodity; the kept thing is the trained OS-world latent predictor (real `predict_fn`) + the live OS trace corpus that grounds it | validate: CI matrix green + parity-guard passes (shim ≡ package) · converge: CEM/MPPI demo lands the goal-reaching change with R→1 on the settled rollout · observe: demo emits a reproducible trace→plan receipt |
| 0.2.0 | grokker constraint | built+validated | the whole runnable substrate, AGPL-open: ZeroPlanner (goal_progress × coherence_R) + proof_of_life + micro_eval + observability trace + full OS world/observation/action surface | coherence_R read on latent displacement (the searching→knowing collapse) as the proprietary SIGNAL discipline + the curated φ-training corpus + the live RSI loop | validate: 28 tests green + proof_of_life ok ∧ micro_eval beats baseline ~12× · converge: union-distinct merge idempotent, R_converging ≈ 0.833 ≫ R_wandering = 0 · observe: every transition emits a replayable trace |
| 0.1.0 | commodity substrate | shipped | the full convergence substrate free — OS-as-world swm env + trace format + coherence-R as the cost any planner minimizes; roll a config change forward and score it before applying | nothing withheld here, on purpose — convergence is commodity (R proposes), the giveaway drives adoption; the sellable thing is the reality-grounded constraint, shipped in later gated blocks | cut when both swm self-checks pass (env + format + Costable register and run clean), a candidate rollout converges to a finite coherence-R, and that R is observable on a live OS-state trace |

## Blocks (detail)

### 0.4.0 — trained-predictor seam · planned

The world model stops being a fixed analytic stand-in. This block draws the moat
boundary in code: the `predict(latent, action) -> latent` rollout step becomes a
public, loadable interface the host OS fills with its own model — while the trained
weights and the loop that produced them stay closed.

- **open** — The world gets the exact `predict(latent, action) -> latent` interface
  + a loader contract + a tiny runnable reference predictor — everything needed to
  plug in your own model and run the full observe → simulate → rank → coherence-R
  loop, with zero changes to the planner or cost.
- **moat** — We keep the trained predictor weights and the φ-curated curation/RSI
  loop that produces them (solve_history → LoRA SFT → measure → promote): the
  interface is commodity and convergence is commodity, but the reality-grounded
  constraint that makes one predictor actually beat the no-model baseline is
  proprietary — that constraint is the sellable thing.
- **gate** — Cut 0.4.0 only when a trained predictor loaded through the seam beats
  the coherence-only/no-model baseline on a held-out OS-trace delta validated by an
  external verifier (validate), the convergence receipt reads done=true / R=1
  (converge), and the seam emits its predict/score rows to the causal chain so the
  win is visible end-to-end (observe).

### 0.3.0 — solver + demo + CI · planned

Hardening + reproducibility release. No new moat: it makes what 0.2.0 shipped
runnable on a real solver, repeatable in CI, and provably mirrored by the in-repo
shim. Replaces the 0.2.0 stand-in ranking in the demo with an actual swm solver
minimizing `CoherenceCost`.

- **open** — The world gets a copy-pasteable trace→plan run on a real CEM/MPPI
  solver, a green CI matrix that proves it installs and runs everywhere, and a
  parity test that keeps the public package honest against the in-repo shim.
- **moat** — None new — convergence (R, the coherence cost, the swm-native
  plumbing) is the commodity we open; the proprietary constraint we keep is the
  trained OS-world latent predictor (the real `predict_fn` dynamics) plus the live
  OS trace corpus that grounds it — the dispose half that turns a ranked candidate
  into a correct one is the sellable part.
- **gate** — Cut when: validate = CI matrix green (install + register + demo +
  tests) and the parity-guard passes (shim ≡ package); converge = the CEM/MPPI demo
  lands the goal-reaching change with R→1 on the settled rollout; observe = the demo
  emits a reproducible trace→plan receipt anyone can re-run.

### 0.2.0 — grokker constraint · built+validated

The OS world-model substrate, merged union-distinct from the in-repo grokker
superset into one standalone package (one impl per concept, clean public names kept
as thin aliases). License: AGPL-3.0-or-later.

- **open** — The world gets the whole runnable substrate, AGPL-open: ZeroPlanner
  (goal_progress × coherence_R) + proof_of_life + micro_eval + observability trace +
  the full OS world/observation/action surface — simulate a config change, score it,
  ship only the ones predicted to help.
- **moat** — We keep the constraint, not the code: coherence_R read on latent
  displacement (the searching→knowing collapse) is the proprietary SIGNAL discipline
  — plus the curated φ-training corpus and the live RSI loop. Convergence is
  commodity (anyone can run the open planner); the reality-grounded constraint that
  decides what's real is the proprietary, sellable part.
- **gate** — Cut the block when all three hold: validated (28 tests green +
  proof_of_life ok ∧ micro_eval beats baseline ~12×), converged (union-distinct
  merge idempotent, R_converging ≈ 0.833 ≫ R_wandering = 0), and observed (every
  transition emits a replayable trace).

### 0.1.0 — commodity substrate · shipped

- **open** — The full convergence substrate ships free — the OS-as-world swm env,
  the trace format, and coherence-R as the cost any planner minimizes; anyone can
  roll a config change forward and score it before applying.
- **moat** — Nothing withheld here, on purpose — convergence is commodity (R
  proposes), so the giveaway drives adoption; the sellable thing is the proprietary
  reality-grounded constraint that disposes (which validated atoms cement to truth),
  shipped in later, gated blocks, not this one.
- **gate** — Cut when both swm self-checks pass (env + format + Costable register
  and run clean), a candidate config rollout converges to a finite coherence-R, and
  that R is observable on a live OS-state trace.
