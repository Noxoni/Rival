# Codex Start Prompt — Rival Milestone 07

You are continuing development of **Rival**, an offline/private RLBot v5 1v1 bot.

Milestone 06 correctly stopped at 20M because RocketSim/headless evaluation improved while the exported candidate severely regressed in real RLBot matches. Do **not** resume serious PPO training yet.

Work directly in `Noxoni/Rival`, isolate the sim-to-RLBot transfer failure, commit stable diagnostics and evidence, and push to `origin/main`.

## Starting boundary

Canonical completed M06 rollback commit:

`652395a9f512ce835830bfc5bc3a7cb078f6105e`

At start:

1. fetch `origin/main` and confirm the repository;
2. preserve all completed M06 work, local ignored checkpoints, raw telemetry, and the existing superseded stash;
3. read every file under `handoff/v7.0/`;
4. read `docs/MILESTONE_06_RESULTS.md`, `training/ENVIRONMENT.md`, `training/rival_training/observations.py`, `training/rival_training/actions.py`, `training/rival_training/policy.py`, production `bot/obs_builder.py`, `bot/action_parser.py`, `bot/config.py`, and candidate deployment/evaluation tooling;
5. do not modify or promote production Rival.

## Conceptual decomposition

Use the RLGym-style decomposition explicitly:

`state s -> observation O(s) -> policy pi(o) -> action function I -> action a -> transition T(s'|s,a)`

The goal is to identify which boundary or interaction explains the M06 transfer failure.

## Primary tasks

### 1. Fix diagnostic flexibility only

The current candidate runtime forces candidate policies to tick skip 4. Add a diagnostic-only way to run candidate/student actors at tick skip 8 or 4 and to hard-mask appended actions 90..157.

Production defaults remain frozen Wisp at tick 8. The diagnostic controls must be explicit and off by default.

### 2. Same-observation policy parity

Using a representative live RLBot observation corpus, compare:

- frozen production Wisp;
- zero-step reconstructed Wisp student;
- rejected 20M trained student.

Use the same first-90 legal mask.

The zero-step reconstruction must preserve frozen-Wisp first-90 logits/top-1 on live observations. If it does not, stop and fix that model/export path before match testing.

For the trained actor, quantify legacy-action drift, confidence/margin changes, and action-transition frequencies.

### 3. Transfer diagnostic match matrix

Execute `TRANSFER_DIAGNOSTIC_MATRIX.md`.

Required modes:

- P0: frozen Wisp, tick 8;
- Z8: zero-step reconstructed student, legacy-only, tick 8;
- Z4: zero-step reconstructed student, legacy-only, tick 4;
- T8: rejected 20M trained actor, legacy-only, tick 8;
- T4: rejected 20M trained actor, legacy-only, tick 4.

Use bounded full five-minute RLBot matches against installed Nexto and Wisp v2-75B, balanced sides. Start at four games/mode; only expand an ambiguous affected mode up to eight games.

Do not turn this into another long skill benchmark.

### 4. Observation-domain audit

Execute `OBSERVATION_DOMAIN_AUDIT.md`.

Compare live production 432 observations with the closest training/RocketSim 432 representations feature-group by feature-group. The training docs already acknowledge approximation in prediction, ETA, landing normal, score handling, and previous-action plumbing; quantify which differences materially move Wisp logits/actions.

Use live natural states rather than hand-authored trick scenarios.

### 5. Action-function parity

Prove legacy action index -> controller semantics across production, candidate deployment, and training parser, including X mirroring.

Separate spatial/controller parity from temporal repeat/delay behavior.

### 6. Short-horizon transition audit

Only after O, pi, and I are understood, compare short-horizon RocketSim vs RLBot physical evolution for natural-state samples under fixed controller sequences. This is a bounded diagnostic, not a demand for bit-identical physics.

## No serious training

Milestone 07 does not authorize resuming Stage C or spending another large PPO budget.

A tiny synthetic or optimizer-free smoke may be used to validate diagnostic code, but do not create a new trained candidate.

## Required result

Commit `docs/MILESTONE_07_RESULTS.md` and compact machine-readable reports that answer:

- Does zero-step reconstructed Wisp preserve RLBot gameplay at tick 8?
- What happens to the same zero-step policy at tick 4?
- How much did the 20M actor drift inside actions 0..89 on live observations?
- Is the 20M actor healthier at tick 8 than tick 4?
- Which observation feature groups differ most between training and live RLBot, and which differences actually affect Wisp decisions?
- Are action parser/mirroring semantics exact?
- Is there measurable short-horizon RocketSim/RLBot transition divergence large enough to matter?
- What is the ranked causal explanation for M06 failure?
- What specific architecture/training correction should Milestone 08 implement?

A mixed/interacting diagnosis is acceptable. Do not force a single-cause conclusion if evidence does not support one.

## Verification

Run the relevant existing production/training suites plus new diagnostics tests, Ruff/compile checks, JSON parse/hash checks, `git diff --check`, and remote readback.

Do the diagnostic work and stop at a coherent pushed boundary. Do not resume serious training in this milestone.
