<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Changelog

All notable changes to `terminals-worldmodel` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Newest release first.

## [0.4.0] - planned — trained-predictor seam

The world model stops being a fixed analytic stand-in. This block draws the moat
boundary in code: the `predict(latent, action) -> latent` rollout step becomes a
public, loadable interface the host OS fills with its own model — while the trained
weights and the loop that produced them stay closed.

### Added
- **`Predictor` protocol** — the open one-step interface `predict(latent: ndarray, action: ndarray) -> ndarray`. Formalizes the signature `DemoWorldModel.predict(z, a) -> z` already exercised end-to-end in `demo.py`. The planner (`ZeroPlanner`) and the swm `Costable` (`CoherenceCost`) call only this protocol, so any conforming model drops in with no change to the rollout, coherence-R scoring, or `1 - goal_progress*coherence_R` cost.
- **Loader contract** — `load_predictor(uri) -> Predictor`: a host-supplied resolver maps a weights URI to a `Predictor`. Ships the contract and a null/raising default; the OS registers the real resolver. The format of the weights blob is deliberately unspecified here (host's choice).
- **Reference predictor** — a tiny, dependency-free numpy predictor (a damped linear `z' = z + γ·M·a` settler) that conforms to the protocol and passes proof-of-life. Proves the seam wires correctly and gives downstream tests a deterministic stand-in; it is NOT the production model.

### Changed
- `ZeroPlanner` / `CoherenceCost` will type their world-model argument as the `Predictor` protocol instead of an ad-hoc duck-typed object — the interface is now the contract, not a convention.
- `DemoWorldModel` reclassified as a conforming reference implementation of `Predictor` (behavior unchanged); docs relabel it "reference, not production."
- README + spec gain a "moat boundary" note: interface open (AGPL-3.0), trained weights + curation/RSI loop closed.

## [0.3.0] - planned — "solver + demo + CI"

Hardening + reproducibility release. No new moat: it makes what 0.2.0 shipped runnable on a real solver, repeatable in CI, and provably mirrored by the in-repo shim. Replaces the 0.2.0 stand-in ranking in the demo with an actual swm solver minimizing `CoherenceCost`.

### Added
- End-to-end `trace → plan` demo on a real swm solver (CEM and MPPI), driving the OS-world from collected traces through `CoherenceCost` to a chosen config change — no hand-rolled ranking, the solver picks.
- CI matrix (Python 3.10–3.13 × `stable-worldmodel` pin) that installs, registers the env/format, and runs the demo + full test suite on every push.
- Parity-guard test asserting the in-repo MIT shim (`irc-strix-halo/src/worldmodel/`) stays isomorphic to this AGPL package — same coherence-R, same cost, same env/format contract; it fails loud if the two drift.

### Changed
- README `Status` section reflects the demo now running on a real CEM/MPPI solver (the transparent stand-in dynamics remains the harness default; production dynamics is still a trained latent predictor via `predict_fn`).
- Version `0.2.0 → 0.3.0`.

## [0.2.0] — 2026-05-30 — "grokker constraint"

The OS world-model substrate, merged union-distinct from the in-repo grokker
superset into one standalone package (one impl per concept, clean public names
kept as thin aliases). License: AGPL-3.0-or-later.

### Added
- **ZeroPlanner** — scores every candidate config change by `goal_progress × coherence_R`, so a stable do-nothing rollout never wins on stability alone. Simulate the change, score it, surface only what's predicted to actually help.
- **proof_of_life** — one-call liveness probe: confirms the stability detector reads `R_converging ≈ 0.833` vs `R_wandering = 0.0`, reports backend presence, and degrades closed when swm/GPU torch is absent.
- **micro_eval** (`run_micro_eval` / `MicroEvalResult`) — the learned predictor beats the no-model baseline ~12× (model MSE 0.00107 vs 0.01257).
- **Observability trace** (`trace_path` / `log_transition` / `read_transitions`) — every transition logged as a replayable bootstrap record.
- **Full OS world/observation/action surface** — `TerminalsWorld`, `PageObservation`, `OSAction`, `TerminalsOSEnv`, swm-native `register_*` (env `terminals-os/Config-v0`, format `terminals-os-trace`, `CoherenceCost` as the swm planning cost).

### Changed
- Merged the in-repo superset (the 15 grokker modules) into the published package via deterministic `UNION DISTINCT`; the in-repo path is now a re-export shim. Idempotent — re-running the build from source reproduces byte-for-byte.
- Relicensed every module to AGPL-3.0-or-later (the 0.1.0 standalone was Apache-2.0); grokker's collapse_detector made an optional import so the package stays standalone.
- Validated: 28 tests pass; proof_of_life + micro_eval green live.

## [0.1.0] - 2026-05-30 — "commodity substrate"

### Added
- swm environment `terminals-os/Config-v0`: the OS's own state as a `stable-worldmodel` env (page context → observation, a config change → action), so an OS configuration change can be rolled forward and scored before it is applied.
- Dataset format `terminals-os-trace`: the on-disk trace schema for recorded OS observation/action rollouts, registered as a first-class swm format (re-entry-guarded so a duplicate `register_format` is a no-op, not a raise).
- `CoherenceCost`: a swm `Costable` (`get_cost(info_dict, action_candidates)`) that any swm solver minimizes; the planner scores `goal_progress × coherence_R`, so doing nothing never wins on stability alone.
- `coherence-R` definition: `R = exp(-d_tail / scale)`, the rollout-stability measure that reads the searching→knowing collapse signal off latent displacement (the soft Čech gluing obstruction).

### Changed
- License pinned to AGPL-3.0 (copyleft-only); SPDX header on every authored file. In-repo files stay MIT to match the parent repo; this published extension is AGPL-3.0.

[0.4.0]: https://github.com/Intuition-Labs-LLC/terminals-worldmodel/releases/tag/v0.4.0
[0.3.0]: https://github.com/Intuition-Labs-LLC/terminals-worldmodel/releases/tag/v0.3.0
[0.2.0]: https://github.com/Intuition-Labs-LLC/terminals-worldmodel/releases/tag/v0.2.0
[0.1.0]: https://github.com/Intuition-Labs-LLC/terminals-worldmodel/releases/tag/v0.1.0
