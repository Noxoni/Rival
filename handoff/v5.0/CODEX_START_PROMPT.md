# Codex Start Prompt — Rival Milestone 05

You are continuing development of **Rival**, a high-end offline/private Rocket League 1v1 bot.

This milestone is an architectural pivot. Do **not** respond with another high-level plan. Work directly in `Noxoni/Rival`, build the training foundation described below, commit stable work, and push it to `origin/main`.

## First: synchronize safely

1. Confirm the repository is `Noxoni/Rival`.
2. Fetch `origin/main`.
3. Inspect every commit after `3c15ff55ba6005777c3ab6457dc3d14e8453a966`.
4. Preserve all legitimate newer v4.1 natural-play work/results. The v5.0 handoff was authored while a v4.1 run could still be active.
5. Read **every file under `handoff/v5.0/`** before implementation.
6. Read current `README.md`, `docs/MILESTONE_03_RESULTS.md`, the latest v4.1 results/reports, `bot/obs_builder.py`, `bot/action_parser.py`, `bot/backend/model.py`, and relevant runtime config.
7. Do not reset/squash history or overwrite unrelated local files/checkpoints/evidence.
8. Verify the frozen Wisp teacher hashes before using them.

If the local worktree still contains an unfinished v4.1 run, preserve it and reach a coherent pushed/stashed boundary before beginning v5.0. Do not mix two active implementations in the same dirty worktree.

---

# Central objective

**Build Rival's first complete RLGym/RocketSim training pipeline.**

RLGym/RocketSim becomes the training environment. RLBot/Rocket League remains the deployment and benchmark environment.

Do not add another tactical rule to Wisp. Do not build another exact scripted challenge suite. Do not start a giant uncontrolled training run.

Milestone 05 must leave us with a reproducible, resumable system capable of training a new Rival policy toward advanced mechanics and better 1v1 decision quality.

---

# Required implementation

Follow `ARCHITECTURE.md`, `TEACHER_BOOTSTRAP.md`, `REWARD_AND_CURRICULUM.md`, and `MILESTONE_05_SPEC.md`.

## 1. Isolated training environment

Create a `training/` subtree with its own dependency environment/configuration.

Use the Python ecosystem for this milestone:

- RLGym v2 / Rocket League simulation package
- RocketSim
- `rlgym-tools`
- `rlgym-ppo`
- PyTorch CUDA when available

Do not downgrade or destabilize the existing RLBot runtime to satisfy training dependencies.

Resolve and record exact installed versions after a working environment is established.

## 2. Headless natural 1v1 environment

Build a modular RLGym v2 1v1 environment using `RocketSimEngine`.

Natural self-play is the default training distribution. No renderer by default.

Implement reset/step stress tests and metrics.

## 3. Mechanics-capable action space

Create `RivalExpandedActionV1`.

Hard requirement: action indices `0..89` must exactly match the current Wisp action rows/order/controller semantics.

Append unique actions derived from a richer `AdvancedLookupTableAction` candidate, initially around:

- `torque_subdivisions=3`
- `flip_bins=16`
- `include_stalls=True`

Compute the actual final count and fingerprint from the installed version. Do not blindly hard-code an expected total.

Support:

- `legacy8` action repeat 8
- `mechanics4` action repeat 4

The trainable Rival policy must be capable of `mechanics4` so it can learn timing-sensitive mechanics/recoveries that Wisp's 8-tick cadence may suppress.

## 4. Observation path

Implement a Wisp-compatible 432-value teacher observation path in RLGym.

Use RocketSim/RLGym prediction/shared-info tooling as needed for Wisp's ball-prediction semantics.

Implement a clean student observation interface. Prefer a Wisp-compatible v1 layout unless there is a compelling low-cost reason to differ.

Do not spend the milestone redesigning observations.

## 5. Bootstrap from Wisp

Do not train from random first unless the bounded teacher bootstrap demonstrably fails.

Try the two bootstrap paths in order:

### A. Direct reconstruction

Inspect TorchScript modules/parameters. If a trainable PyTorch reproduction can be built **and numerically verified against Wisp**, use it and expand the action head.

Bound this attempt. If it becomes reverse-engineering work, stop.

### B. Behavior distillation

Use frozen Wisp as a teacher in natural headless RocketSim trajectories.

Generate a bounded dataset and train a PPO-compatible student actor.

For a 4-tick student, teacher actions chosen every 8 ticks may target the two corresponding 4-tick student steps.

Report held-out agreement/loss and checkpoint reload validity.

A bounded teacher-bootstrap failure is acceptable if documented honestly; the training scaffold must still be completed.

## 6. Reward v1

Winning must dominate.

Implement/log separate components for:

- goal/concede outcome;
- modest possession/useful-touch or offensive-progress shaping;
- small boost-efficiency shaping;
- small recovery-value shaping.

Optional low-weight curriculum aids include flip-reset, wavedash, and aerial-usefulness rewards.

Do not reward musties/breezis/zap dashes/etc. by name. Reward the outcomes that make those mechanics useful.

Specifically preserve incentives for:

- not throwing away controlled ceiling/aerial possession;
- acquiring/using resets productively;
- using flips/momentum to reduce aerial boost dependence;
- retaining boost for recovery;
- landing and regaining defensive speed quickly after lost possession or a failed shot.

## 7. Throughput benchmark

Benchmark a few worker counts rather than guessing. Suggested candidates:

`8, 12, 16, 24`

Choose the best stable headless rollout configuration and record steps/sec and relevant CPU/GPU resource observations.

Do not spend excessive time micro-optimizing.

## 8. PPO smoke

Run a bounded multi-iteration PPO smoke using the best available initialization.

Prove:

- rollout collection;
- PPO update;
- finite losses/rewards;
- reward-component metrics;
- checkpoint save;
- checkpoint reload;
- resume;
- action-distribution logging;
- no NaN/invalid-state failure.

Do not launch a multi-hour or billion-step training run in this milestone.

## 9. Deployment seam

Create an inference/export adapter sufficient to load the student actor and produce actions.

Do not replace the current Wisp-backed live Rival bot yet.

If evaluating `rlgym-rlbot`, keep it isolated; it currently has its own RLBot version requirements. Never downgrade the working production environment just to install the wrapper.

---

# Mechanical capability target

The architecture must leave Rival able to learn, through RL rather than hand-coded macros:

- flip resets and useful reset follow-ups;
- ceiling resets/control;
- better air-dribble control and opponent outplays;
- musty/breezi/Meeri-pop-like control sequences if they become useful;
- wavedash/zap-dash/wall-dash-style recovery and acceleration;
- sidewall recovery/skimming behavior;
- use of flips to maintain aerial momentum and conserve boost;
- fast defensive recovery after missed offense or possession loss.

Milestone 05 does **not** have to prove mastery of all of these. It must create an action/observation/training system that does not structurally prevent them.

---

# Verification

At minimum run:

- current Rival test suite;
- new training tests;
- exact Wisp first-90 action equivalence test;
- expanded-action uniqueness/fingerprint test;
- 432-observation shape/finiteness tests;
- RocketSim reset/step stress smoke;
- reward finiteness tests;
- Wisp teacher hash verification;
- teacher bootstrap validation if used;
- PPO save/reload/resume test;
- deployment inference smoke;
- lint/compile/type checks appropriate to changed code;
- `git diff --check`;
- scan staged files to ensure no huge datasets/checkpoints/secrets/machine-specific raw paths were accidentally added.

---

# Required committed outputs

Commit and push:

- `training/` implementation;
- reproducible training dependency/config files;
- action-table metadata/fingerprint;
- `docs/MILESTONE_05_RESULTS.md`;
- bootstrap report;
- throughput report;
- PPO smoke/checkpoint manifest;
- tests.

Large generated datasets and training checkpoints may remain Git-ignored; record hashes, sizes, format versions, and reproduction commands.

---

# End-of-run report

Return a concise result containing:

- commit SHA(s);
- final `origin/main` SHA;
- exact dependency/runtime versions;
- action-space count/fingerprint and first-90 parity result;
- observation shape/design;
- teacher-bootstrap method used and outcome;
- dataset size and held-out imitation metrics if distillation ran;
- selected worker count and measured throughput;
- PPO smoke timesteps/iterations and key metrics;
- checkpoint save/reload/resume result;
- test results;
- runtime warnings/errors;
- what remains before the first serious training run.

Do the work. Prefer a functional training system over additional prose or speculative architecture changes.
