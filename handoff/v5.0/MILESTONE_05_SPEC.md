# Rival Milestone 05 — RLGym Training Foundation

## Mission

Build the first complete, testable Rival training pipeline in RLGym/RocketSim without disturbing the current RLBot baseline.

The result of this milestone is **a working training system**, not a promise that one short run produces a better-than-Wisp final bot.

---

## Starting procedure

1. Fetch `origin/main`.
2. Inspect all commits after `3c15ff55ba6005777c3ab6457dc3d14e8453a966`.
3. Preserve legitimate v4.1 natural-play results and any coherent work already pushed by the active Codex run.
4. Do not reset/squash historical handoffs.
5. Read the complete `handoff/v5.0/` package.
6. Verify the Wisp teacher model hashes before using them.
7. Leave unrelated local files and ignored evidence/checkpoints untouched.

If v4.1 is still mid-run locally, stop it only at a coherent boundary or let it finish. Do not run two conflicting Codex implementations against the same worktree.

---

## Stage 1 — isolated training environment

Create a dedicated training environment that does not modify the working RLBot dependency set.

Requirements:

- Python version compatible with current RLGym v2 ecosystem and PyTorch CUDA.
- RLGym v2 / Rocket League sim package.
- `rlgym-tools`.
- `rlgym-ppo` from its maintained source if necessary.
- pinned/recorded resolved dependency versions after installation succeeds.
- reproducible setup command/script.
- training outputs/checkpoints ignored by Git by default.

Record environment versions in `training/ENVIRONMENT.md` or a generated manifest.

Do not install `rlgym-rlbot` into the existing Rival bot environment if its RLBot version pin conflicts.

---

## Stage 2 — 1v1 RocketSim environment

Implement `training/env` with modular RLGym v2 components.

Minimum environment:

- 1 blue vs 1 orange;
- `RocketSimEngine`;
- normal Soccar physics;
- no rendering by default;
- goal termination;
- sensible no-touch/game-time truncation;
- deterministic seed option for debugging, but natural randomized training by default;
- reset/step smoke test for at least several thousand decisions without NaN/exception.

Add metrics hooks from the beginning.

---

## Stage 3 — action space

Implement `RivalExpandedActionV1`.

Hard invariants:

- first 90 action rows exactly match `bot/action_parser.py` values/order;
- same teacher action index maps to equivalent controller input;
- append only unique mechanics-capable actions;
- generate additions from `rlgym-tools` `AdvancedLookupTableAction` or equivalent reviewed implementation;
- initial candidate: `torque_subdivisions=3`, `flip_bins=16`, `include_stalls=True`;
- compute actual union count at runtime/build time;
- serialize/fingerprint the final table;
- unit test exact first-90 equivalence and uniqueness.

Implement cadence modes:

- `legacy8` = repeat 8;
- `mechanics4` = repeat 4.

Primary student training should be capable of using `mechanics4`.

---

## Stage 4 — observations

Implement a Wisp-compatible teacher observation path.

Requirements:

- expected teacher input shape 432;
- finite values over randomized states;
- semantic reproduction of current normalization and feature ordering;
- ball prediction supplied via RocketSim/RLGym shared-info or an equivalent tested path;
- previous-action semantics represented correctly;
- documented differences, if any, between live RLBot Wisp observations and RLGym teacher observations.

Implement student observation v1 with a clean interface. Prefer Wisp-compatible layout initially unless a clearly better design can be added without breaking bootstrap simplicity.

Do not stall the milestone redesigning observations.

---

## Stage 5 — Wisp bootstrap

Follow `TEACHER_BOOTSTRAP.md`.

### 5A direct reconstruction attempt

Spend a bounded effort inspecting TorchScript parameters/graph.

If trainable reconstruction can be verified numerically, use it.

If not, stop and use 5B.

### 5B behavior distillation fallback

Build the Wisp teacher adapter and generate a bounded natural RocketSim dataset.

Train a student actor whose weights can seed PPO.

For `mechanics4`, repeat each native 8-tick Wisp target for the two corresponding 4-tick student decisions.

Required reporting:

- dataset trajectories/records;
- train/validation split method;
- teacher inference throughput;
- loss curve summary;
- top-1/top-k agreement;
- checkpoint hash/path convention;
- new-action selection rate before PPO.

Do not pursue perfect imitation indefinitely. Produce a healthy resumable checkpoint.

If both direct reconstruction and bounded distillation fail, document the exact blocker and still complete the rest of the training scaffold with a random smoke actor. Do not silently pretend Wisp initialization succeeded.

---

## Stage 6 — Reward v1

Implement a small, instrumented initial reward stack based on `REWARD_AND_CURRICULUM.md`.

Required categories:

- dominant goal/concede outcome;
- modest possession/useful-touch or offensive progress shaping;
- small boost-efficiency component;
- small recovery-value component.

Optional in the first smoke run:

- low-weight flip-reset bonus;
- low-weight wavedash bonus;
- aerial-distance/usefulness bonus.

Log each component independently.

Do not create rewards for named freestyle mechanics.

All rewards must stay finite in long random/teacher rollouts.

---

## Stage 7 — throughput benchmark

Before a long training smoke, benchmark headless rollout throughput for a bounded set of worker counts.

Suggested candidates:

`8, 12, 16, 24`

For each viable setting record:

- total environment steps/sec;
- simulated game seconds/sec if available;
- CPU utilization/pressure if readily measurable;
- GPU utilization/memory if readily measurable;
- inference batch behavior;
- errors/stalls.

Select the best stable value; do not assume more processes is better.

Do not spend hours microbenchmarking.

---

## Stage 8 — PPO smoke training

Run a bounded PPO smoke from the best available initialization:

1. Wisp-derived student if bootstrap succeeded;
2. otherwise a clearly labeled random student only to prove the pipeline.

Use natural 1v1 self-play.

Goals of the smoke:

- experience collection works;
- PPO updates occur;
- losses/rewards remain finite;
- checkpoint saving works;
- checkpoint reload works;
- training resumes from the checkpoint;
- expanded action distribution can be inspected;
- metrics are emitted.

Do not run an uncontrolled multi-hour/billion-step training job in this milestone. Choose a bounded timestep target sufficient to exercise multiple PPO iterations/checkpointing, then stop cleanly.

---

## Stage 9 — deployment/inference seam

Implement enough of `training/deploy` to load the student actor and generate an action for a single observation/state.

Prefer an adapter that can later slot into Rival's existing RLBot runtime.

Do not replace the current live Wisp baseline in this milestone.

If `rlgym-rlbot` is tested, do so in the isolated training environment and record its RLBot-version compatibility. Never downgrade the existing working RLBot setup to make the wrapper install.

---

## Stage 10 — tests and documentation

Required tests/checks:

- existing Rival test suite still passes;
- training unit tests;
- exact first-90 action equivalence;
- expanded action uniqueness/fingerprint;
- 432 teacher observation shape and finite values;
- environment reset/step stress smoke;
- reward finiteness;
- Wisp hashes unchanged;
- teacher/student bootstrap validation if used;
- PPO checkpoint save/reload/resume;
- inference/export smoke;
- `git diff --check`;
- no secrets or machine-specific generated training data accidentally committed.

Required committed docs/results:

- `docs/MILESTONE_05_RESULTS.md`;
- resolved training dependency manifest;
- training config(s);
- action-table metadata/fingerprint;
- bootstrap report;
- throughput report;
- PPO smoke report/checkpoint manifest.

Large datasets and checkpoints may stay Git-ignored with hashes and reproduction paths recorded.

---

## Explicit non-goals

Do **not** spend Milestone 05 on:

- more fake-challenge scripted-probe engineering;
- another hand-coded tactical override;
- fully solving every named mechanic;
- long final model training;
- porting the trainer to Rust;
- parallel RocketLeague.exe instance engineering;
- changing Wisp teacher weights;
- replacing the production Rival bot before a trained checkpoint earns promotion.

---

## Completion definition

Milestone 05 is complete when Rival has a reproducible, resumable RLGym/RocketSim 1v1 training pipeline with a mechanics-capable action space, verified Wisp bootstrap result (success or documented bounded failure), instrumented reward stack, measured throughput, successful PPO smoke training, and a model-inference seam suitable for the next milestone's real training run.
