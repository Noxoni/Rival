# Codex start prompt — Rival v10.1 agency bootstrap

You are implementing and executing Rival Milestone 10.1 in the canonical repository `Noxoni/Rival`.

This is a steering change from the original M10 100-hour campaign because capability evidence at +10h showed the scratch policy was still largely non-interactive despite healthy PPO/runtime telemetry.

## 0. Preserve the active M10 boundary first

Do **not** alter the currently running M10 process mid-iteration or silently discard its progress.

1. Let the existing M10 run reach the nearest clean **+25 simulated-hour** boundary.
2. Run the already-authorized +25 M10 fixed evaluation and transfer checks.
3. Save the exact +25 checkpoint, optimizer states, trainer counters, metrics and reports.
4. Commit and push the completed +25 boundary to `origin/main`.
5. Verify local `HEAD`, `origin/main`, and remote ref agree.
6. Stop the original M10 continuation there. Do not spend the remaining +75h of the old campaign until this steering change has been reviewed.

If +25 already completed before this prompt is received, use that exact completed/pushed boundary and do not rerun it unnecessarily.

## 1. Import this package without rewinding history

The v10.1 package lives on branch:

`origin/rival-v10.1-agency-bootstrap`

That design branch was created from the pushed +10-era main boundary and may be behind your completed +25 history.

Bring **only the `handoff/v10.1/` package** onto the final +25 `main` history by a normal merge/rebase/cherry-pick strategy that preserves every newer M10 commit. Never reset `main` backward to the design branch.

Read every file under `handoff/v10.1/` before implementation.

## 2. Immutable architecture

Do not redesign or replace:

- `RivalPolicyV1`;
- `RivalCriticV1`;
- `RivalObsV1` / its schema;
- `RivalActionV1` / its schema;
- `RivalCanonicalStateV1`;
- native 120-Hz cadence;
- one-tick RLBot action-delay semantics;
- episode symmetry;
- the proven `rlgym-ppo` hybrid-distribution math;
- export/deployment contracts;
- production frozen Wisp.

Do not train from random again. Resume the exact completed M10 +25 checkpoint including actor, critic, both optimizers and trainer counters.

## 3. Implement new versioned bootstrap components

Do not mutate `RivalScratchRewardV1` or `RivalScratchResetCurriculumV1` in place.

Create new, explicitly versioned implementation modules, expected names:

- `training/rival_training/v10_bootstrap_reward.py`
- `training/rival_training/v10_bootstrap_curriculum.py`
- `training/rival_training/v10_bootstrap_environment.py` or an equally clean versioned environment seam
- `training/rival_training/v10_bootstrap_metrics.py` if needed
- a recoverable v10.1 campaign runner/config under `training/`

Implement exactly the behavioral contract in:

- `AGENCY_BOOTSTRAP_REWARD.md`
- `AGENCY_BOOTSTRAP_CURRICULUM.md`
- `EVALUATION_AND_EXIT_GATES.md`
- `M10_1_CAMPAIGN.json`

The bootstrap reward must include:

- +10 goal / -10 concede;
- cadence-safe useful-speed rate;
- stronger ball-approach potential;
- +0.30 base reward per distinct logical ball touch;
- +0.45 additional reward for a qualifying aerial touch;
- progressively larger same-possession touch-chain bonuses;
- small attacking ball-progress potential;
- total absolute non-outcome episode budget <=7.5;
- no recovery reward during this bootstrap;
- no boost-waste penalty during this bootstrap;
- no dodge-resource reward during this bootstrap;
- **no reward for raw control/button presses**.

Implement logical-touch debounce/auditing exactly enough that continuous car-ball contact cannot print reward every physics tick.

## 4. Implement the interaction-dense reset curriculum

Phase A starts at v10.1 activation:

- 30% ground acquisition;
- 20% moving-ball chase;
- 20% touch-chain/follow-up;
- 15% easy aerial contact;
- 10% easy finish;
- 5% natural kickoff.

Use the broad randomized distributions in `AGENCY_BOOTSTRAP_CURRICULUM.md`, not narrow repeated scenarios.

During v10.1:

- no-touch timeout = 10 seconds;
- episode timeout = 120 seconds;
- goal terminates normally.

Advance curriculum phases only at clean evaluation boundaries after their readiness gates pass. Do not silently weaken gates.

## 5. Pre-training verification for the new reward/curriculum only

The M09 architecture gates do not need to be repeated wholesale because the architecture is immutable.

Before applying another PPO update, add/run focused tests proving:

1. reward formulas and 120-Hz cadence scaling;
2. +10/-10 outcome precedence;
3. 7.5 non-outcome budget ceiling;
4. touch debounce and chain reset/increment behavior;
5. aerial-touch classification;
6. no direct action-press reward exists;
7. curriculum family distribution over >=10,000 resets/phase;
8. all reset physics finite/legal;
9. mirror/team symmetry;
10. dead-play timeout behavior;
11. exact +25 checkpoint fresh reload;
12. one real CUDA PPO smoke iteration from a disposable copy of the checkpoint with finite updates on both action branches;
13. production frozen Wisp files/config untouched.

The disposable smoke must not alter the canonical resume checkpoint.

## 6. Run the bounded bootstrap

Resume the exact +25 M10 policy and run v10.1.

Evaluation boundaries from bootstrap activation:

- +2.5h
- +5h
- +10h
- +15h
- +20h
- +25h maximum

At every boundary:

- checkpoint actor, critic, optimizers and trainer state;
- run the original fixed deterministic M09/M10 protocol for historical comparison;
- run bootstrap-specific probes;
- report motion, touches, aerial touches, chains, goals, no-touch truncations, action-use diagnostics and reward-component spend;
- compare directly with M09, M10 +10 and M10 +25.

If the exit gate passes on two consecutive boundaries, stop v10.1 early. Do not consume the remaining budget just because it exists.

If Phase A is still below its readiness gate at +10 bootstrap hours, stop and report rather than continuing blindly.

Absolute v10.1 maximum: **25 added simulated game-hours / 21.6M added agent-steps**, with clean-boundary overshoot handled the same way as M09/M10.

## 7. Interpretation rules

Do not claim success because:

- steps/sec is high;
- reward rises;
- PPO losses are finite;
- workers remain healthy;
- two inactive policies draw 0-0.

Success requires actual deterministic capability: motion, touches, jumps, dodges, aerial touches, repeated touches and goal activity.

If reward increases without those behaviors, treat it as likely reward exploitation and investigate.

## 8. Repository discipline

- meaningful stable implementation/evidence boundaries go to GitHub;
- generated large checkpoints/raw telemetry remain ignored;
- commit compact machine-readable summaries and hashes;
- preserve M08/M09/M10 history and the historical v4 stash;
- keep production frozen Wisp unless separately authorized;
- do not promote a bootstrap candidate to production.

At completion/stop, produce `docs/MILESTONE_10_1_RESULTS.md`, compact evidence under `training/results/milestone10_1/`, final verification, and exact remote SHA/readback.
