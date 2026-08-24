# Codex start prompt — Rival Milestone 10.2 Ball Acquisition

You are implementing and executing Rival Milestone 10.2 in `Noxoni/Rival`.

This is a **prerequisite-learning milestone**, not a general Rocket League training campaign.

## First: establish repository authority

1. Stop if any Rival training process from M10/v10.1 is unexpectedly still running; do not kill unrelated Rocket League/RLBot processes.
2. Fetch remotes.
3. Verify `origin/main` contains the completed Milestone 10.1 closeout at or after authority commit:
   `cc2d971b4990121684920f87a0ee2b87b6dc801b`.
4. Fetch `origin/rival-v10.2-ball-acquisition`.
5. Read **every file** under `handoff/v10.2/` before changing code.
6. Bring the v10.2 handoff package onto your working `main` non-destructively. If `main` still equals the authority base, fast-forward is appropriate. If it advanced, preserve all newer work and reconcile; never rewind or force-reset history.
7. Confirm the worktree is clean except for the historical intentional untracked/stashed items already documented by the project.

## Governing objective

Milestone 10.2 teaches only:

> **locate/reach the ball by reducing separation, then make real physical contact with it.**

Do not expand scope into dribbling, aerial control, scoring, opponents, self-play, imitation, or named mechanics.

The skill ladder in `SKILL_PROGRESSION.md` is authoritative for what comes later.

## Frozen architecture

Do not change unless a direct implementation defect forces a separately documented stop:

- `RivalPolicyV1` actor topology;
- `RivalObsV1` 714-float observation;
- `RivalActionV1` native controller distribution;
- 120-Hz policy/physics cadence;
- one-tick action-delay semantics;
- canonical RocketSim/RLBot adapters;
- actor export/live inference contract.

No Wisp/Nexto actor/trunk may enter the scratch policy.

## Source skill transfer

Use actor weights from:

`training/checkpoints/milestone10_1/boundaries/plus-010h/032019870`

Expected actor SHA-256:

`e6b9fd1a38dbb5c6670711a8b47769a628d2f20caf7c88738f372d0839b7a3b6`

This actor is retained because v10.1 established the locomotion/speed primitive.

However, the reward regime changes materially. Therefore:

- load actor weights exactly;
- initialize a **fresh RivalCriticV1**;
- initialize fresh actor Adam state;
- initialize fresh critic Adam state;
- do not resume v10.1 critic values or optimizer moments;
- preserve the source checkpoint byte-for-byte.

## Implement the v10.2 stack

Create clearly versioned Rival-owned modules/scripts/tests for at least:

- `RivalBallAcquisitionRewardV1`;
- `RivalBallAcquisitionCurriculumV1`;
- `RivalBallAcquisitionEnvV1`;
- active-learner/dummy experience isolation;
- `RivalBallAcquisitionEvaluationV1`;
- recoverable v10.2 campaign runner;
- boundary finalizer/reporting;
- optional RLViser inspection path if useful, isolated from training workers.

Naming/files may follow existing repository conventions, but version identifiers above must appear in machine-readable evidence.

## Reward: implement exactly

Read `BALL_ACQUISITION_REWARD.md` as authority.

The only positive learning signals are:

1. small signed **car-caused progress toward the current ball position**;
2. `+1.0` for each genuine new physical active-learner ball touch.

Critical requirements:

- **zero speed reward**;
- zero direct action/controller reward;
- zero jump/dodge/boost reward;
- zero recovery reward;
- zero aerial-specific reward;
- zero ball-to-goal reward;
- **goal reward = 0**;
- **concede reward = 0**;
- dense distance absolute episode budget = `0.75`;
- touch rewards are not included in that dense budget;
- repeated genuine separated touches each earn another `+1.0`;
- sustained contact is not rewarded once per 120-Hz tick.

Do not blend facing, velocity, speed, or ball-progress terms back into the distance objective.

## Curriculum: implement exactly

Read `BALL_ACQUISITION_CURRICULUM.md`.

Phase A reset weights:

- stationary_close 30%;
- stationary_medium 25%;
- moving_chase 20%;
- awkward_heading 20%;
- natural_kickoff_holdout 5%.

One active learner per episode; one all-zero non-interfering dummy car exists only to preserve the unchanged opponent observation contract. Randomize active team. Dummy transitions must not enter PPO loss construction.

Phase B is harder acquisition/generalization with the frozen weights in the curriculum file. It is only authorized after one Phase A readiness pass.

No active opponent, Wisp, Nexto, or self-play in this milestone.

## Preflight

Execute every gate in `IMPLEMENTATION_GATES.md` before spending real v10.2 campaign experience.

Particularly prove:

- exact actor transfer + fresh critic/optimizers;
- reward truth table;
- true touch detector on RocketSim traces;
- dummy exclusion from PPO;
- 50k Phase-A reset audit;
- frozen 500-episode evaluation corpus and disjoint generalization corpus;
- source actor baseline on both corpora;
- real disposable CUDA PPO update with no dummy rows;
- exact reload after disposable smoke;
- reset to exact source actor + fresh critic/optimizers before campaign start.

Commit compact preflight evidence under:

`training/results/milestone10_2/`

Do not commit giant checkpoints/raw dumps.

## Training authority

Maximum added v10.2 physical simulated game-hours: **15**.

Because only one agent is trainable:

- 1 simulated game hour = `432,000` active-learner 120-Hz transitions;
- maximum trainable active-learner steps = `6,480,000`.

Do not mix this stage-step count arithmetically with the historical two-trainable-agent `cumulative_agent_steps` convention. Report both historical source metadata and v10.2 stage-active steps explicitly.

Starting PPO values are frozen in `M10_2_CAMPAIGN.json`. If the one-trainable-agent path materially changes throughput, perform the bounded worker check authorized there and select on **trainable learner steps/sec**, not dummy-inclusive raw rows.

## Evaluation boundaries

Evaluate at added v10.2 hours:

`+1, +2.5, +5, +7.5, +10, +12.5, +15`.

At every boundary:

- checkpoint atomically;
- independently reload before evaluation;
- run deterministic frozen gate corpus;
- apply `EVALUATION_AND_EXIT_GATES.md` exactly;
- preserve compact machine-readable evidence;
- push a coherent stable Git boundary.

When a checkpoint appears to pass, run the disjoint generalization corpus as required.

Phase A first pass only unlocks Phase B. Final skill completion requires **two consecutive Phase-B acquisition-ready passes**.

Stop immediately when the final gate passes. Do not spend the remainder just because budget remains.

Honor the +5h no-learning stop and +15h hard stop.

## Capability interpretation

This milestone is successful only if Rival reliably touches the ball from the specified reachable acquisition distributions.

Do not claim success because:

- reward rises;
- the car drives fast;
- PPO losses are finite;
- workers are healthy;
- goals happen accidentally;
- jump/dodge usage rises.

Primary evidence is episode-level first-touch acquisition success and time-to-first-touch.

## Completion outputs

Create:

`docs/MILESTONE_10_2_RESULTS.md`

and compact evidence under:

`training/results/milestone10_2/`

The result document must state one explicit authority decision, e.g.:

- `ball_acquisition_skill_passed_unlock_ground_control`, or
- one of the documented stop/failure decisions.

If passed, identify the exact preserved Stage-1 actor checkpoint that future Stage-2 ground-control/dribbling work must start from.

If failed, preserve the best/recoverable checkpoint and explain the measured failure without authorizing an unversioned reward change.

## Final verification

Before declaring completion:

- run production tests;
- run training tests;
- Ruff training code;
- compileall training code/scripts/tests;
- parse all new JSON;
- independently reload final checkpoint and held actor outputs;
- verify source v10.1 checkpoint unchanged;
- verify frozen Wisp production hashes/configuration unchanged;
- verify no v10.2 training worker remains;
- verify checkpoint binaries/RLViser remain ignored as appropriate;
- verify local `main` and `origin/main` match after final push;
- report final commit SHA.

No production promotion is authorized by Milestone 10.2.
