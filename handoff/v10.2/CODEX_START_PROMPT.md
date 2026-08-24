# Codex start prompt — Rival v10.2 Progressive Prerequisite Curriculum

You are implementing and executing Rival Milestone 10.2 in `Noxoni/Rival`.

This is an **unattended progressive skill curriculum** authorized through Stage 4 only. You may advance automatically from one stage to the next **only after the exact prerequisite success gate passes**. Any stage failure/stop condition ends the entire run. Never skip a prerequisite.

## 1. Establish repository authority

1. Stop if any prior Rival training process is unexpectedly still running; do not kill unrelated Rocket League/RLBot processes.
2. Fetch remotes.
3. Verify `origin/main` contains completed Milestone 10.1 at or after:
   `cc2d971b4990121684920f87a0ee2b87b6dc801b`.
4. Fetch `origin/rival-v10.2-ball-acquisition`.
5. Read **every file** under `handoff/v10.2/` before changing code.
6. Bring the entire v10.2 handoff package onto working `main` non-destructively. Preserve any newer main work; never rewind/force-reset history.
7. Confirm the worktree is clean except for historical intentionally preserved/untracked items already documented by the project.

## 2. Governing state machine

Execute:

```text
Stage 1 — Ball acquisition
   exact pass -> ball_acquisition_skill_passed_unlock_ground_control
        |
        v
Stage 2 — Ground control / dribbling
   exact pass -> ground_control_skill_passed_unlock_aerial_control
        |
        v
Stage 3 — Aerial acquisition / air-dribble control
   exact pass -> aerial_control_skill_passed_unlock_finishing
        |
        v
Stage 4 — Finishing / scoring
   exact pass -> finishing_skill_passed_unlock_opponent_pressure
        |
        v
STOP FOR HUMAN REVIEW
```

Stage 5 opponent pressure, self-play, opponent league, imitation, and production promotion are **not authorized**.

Primary authorities:

- `PROGRESSIVE_STAGE_PROTOCOL.md`
- `M10_2_PROGRESSIVE_CAMPAIGN.json`
- `OVERNIGHT_10H_BUDGET.md`
- Stage-specific documents listed below.

## 3. Overnight wall-clock authority

The entire unattended progressive campaign has a **10 real wall-clock hour** authority from the beginning of real Stage-1 progressive work.

Follow `OVERNIGHT_10H_BUDGET.md` exactly:

- reserve the final 20 minutes for clean finalization;
- record elapsed/remaining wall time at every boundary;
- use measured iteration/evaluation duration to decide whether another PPO iteration safely fits;
- unused time from a mastered stage carries forward;
- if a prerequisite needs longer than its nominal share, it may consume later-stage time while still inside its own stage/no-learning limits;
- never skip a prerequisite to preserve time;
- do not start a new stage with less than 30 minutes remaining before the finalization reserve;
- on wall-clock exhaustion preserve a clean checkpoint/evidence and stop with:
  `stop_progressive_overnight_wall_clock_budget_exhausted`.

Nominal overnight planning shares are roughly:

- Stage 1: ~1.5 real hours;
- Stage 2: ~2.0 real hours;
- Stage 3: ~3.0 real hours;
- Stage 4: ~3.17 real hours;
- finalization reserve: 20 minutes.

These are soft planning shares. Capability gates and the global 10-hour ceiling are authoritative.

## 4. Frozen architecture for all stages

Do not redesign:

- `RivalPolicyV1`;
- `RivalObsV1` (714 floats);
- `RivalActionV1` native controller distribution;
- 120-Hz policy and physics;
- one-tick action-delay semantics;
- canonical RocketSim/RLBot adapters;
- actor topology/export/live inference contract.

All isolated Stages 1–4 use one trainable learner and one non-interfering inert dummy solely to preserve the unchanged opponent-observation contract. Dummy transitions must never enter PPO loss/GAE.

No Wisp/Nexto actor/trunk enters the scratch policy.

## 5. Stage transition contract

At every successful stage transition:

1. complete the passing evaluation boundary;
2. independently reload the passing checkpoint and reproduce held actor outputs;
3. preserve the passing actor checkpoint immutably;
4. write compact stage evidence and push a stable Git boundary;
5. initialize the next stage from **actor weights only**;
6. initialize a fresh `RivalCriticV1`;
7. initialize fresh actor Adam state;
8. initialize fresh critic Adam state;
9. create/freeze the next stage's deterministic gate corpus and disjoint unseen corpus before its first real PPO update;
10. measure previous-skill retention baseline;
11. run the next stage's focused implementation/preflight, including one disposable real CUDA PPO update;
12. discard disposable training and restart the real next-stage campaign from the exact transferred actor plus fresh critic/optimizers.

Never carry critic values or optimizer moments across reward changes.

## 6. Stage 1 — Ball acquisition

Start actor from:

`training/checkpoints/milestone10_1/boundaries/plus-010h/032019870`

Expected actor SHA-256:

`e6b9fd1a38dbb5c6670711a8b47769a628d2f20caf7c88738f372d0839b7a3b6`

Authority:

- `BALL_ACQUISITION_REWARD.md`
- `BALL_ACQUISITION_CURRICULUM.md`
- `EVALUATION_AND_EXIT_GATES.md`
- `IMPLEMENTATION_GATES.md`
- `M10_2_CAMPAIGN.json`

Teach only:

> reduce car-to-ball separation and make genuine physical ball contact.

Reward:

- small signed car-caused distance progress;
- `+1.0` every genuine new learner ball touch;
- zero speed reward;
- zero goal/concede reward;
- zero future-skill reward.

Maximum Stage-1 experience: 15 learner-simulated hours / 6,480,000 active learner steps.

Only exact success decision authorizes Stage 2.

## 7. Stage 2 — Ground control / dribbling

Authority:

- `STAGE_2_GROUND_CONTROL.md`

Implement versioned:

- `RivalGroundControlRewardV1`;
- `RivalGroundControlCurriculumV1`;
- `RivalGroundControlEnvV1`;
- `RivalGroundControlEvaluationV1`.

Teach:

> after first contact, keep the ball reachable and produce repeated controlled ground contacts.

First touch becomes a small bridge. Genuine follow-up ground touches are the semantic maximum. Scoring remains reward-neutral.

Maximum: 20 learner-simulated hours / 8,640,000 active learner steps.

Only exact success decision authorizes Stage 3.

## 8. Stage 3 — Aerial acquisition / air-dribble control

Authority:

- `STAGE_3_AERIAL_CONTROL.md`

Implement versioned:

- `RivalAerialControlRewardV1`;
- `RivalAerialControlCurriculumV1`;
- `RivalAerialControlEnvV1`;
- `RivalAerialControlEvaluationV1`.

Teach:

> intentionally acquire an elevated ball and sustain repeated controlled aerial contacts.

Do not reward button presses. Physical aerial acquisition/follow-up contact is the lesson. Scoring remains reward-neutral.

Maximum: 30 learner-simulated hours / 12,960,000 active learner steps.

Only exact success decision authorizes Stage 4.

## 9. Stage 4 — Finishing / scoring

Authority:

- `STAGE_4_FINISHING.md`

Implement versioned:

- `RivalFinishingRewardV1`;
- `RivalFinishingCurriculumV1`;
- `RivalFinishingEnvV1`;
- `RivalFinishingEvaluationV1`.

This is the **first** positive scoring lesson.

Teach:

> use learned ground/aerial control to deliberately put the ball into the designated opponent goal.

Reward:

- small bounded control-retention bridge;
- small bounded ball-to-target progress;
- correct goal `+10`;
- own goal `-10`.

Capability evaluation must include ground-control-qualified and aerial-control-qualified goal rates; accidental/passive goals alone cannot pass.

Maximum: 25 learner-simulated hours / 10,800,000 active learner steps.

On exact success decision, stop. Do not begin opponent training.

## 10. Progressive campaign controller

Implement a resumable Rival-owned progressive controller recording at least:

- current stage/phase;
- source/passing checkpoint paths and hashes;
- stage active-learner steps and simulated hours;
- total progressive active-learner steps and simulated hours;
- campaign/stage wall-clock elapsed;
- wall-clock remaining;
- latest clean recovery checkpoint;
- current evaluation boundary;
- gate decision;
- passed prerequisite checkpoint registry;
- next authorized stage;
- stop reason.

Writes must be atomic/recoverable. Re-running after interruption must resume cleanly without repeating a completed stage or discarding a passing prerequisite checkpoint.

## 11. PPO defaults

Unless a stage authority explicitly overrides them:

- gamma `0.9987444968227265`;
- GAE lambda `0.9872585449014338`;
- rollout 96,000 trainable learner steps;
- PPO batch 96,000;
- minibatch 24,000;
- one epoch;
- clip 0.2;
- actor LR `1e-4`;
- critic LR `1e-4`;
- analog entropy coefficient `0.0002`;
- button entropy coefficient `0.001`;
- max gradient norm `1.0`.

Start worker count from 56. A bounded worker recheck is allowed at a new stage only if environment changes materially affect throughput. Select on stable **active learner steps/sec**, never dummy-inclusive row count.

## 12. Evidence and interpretation

Training reward/loss/entropy/throughput prove infrastructure only. Skill mastery is determined by each stage's deterministic frozen corpus plus unseen generalization and prerequisite-retention gates.

At minimum push stable Git boundaries:

- after progressive implementation/preflights;
- after every evaluation boundary;
- after every successful stage transition;
- at any failure/stop closeout;
- after final Stage-4 completion or overnight wall-time closeout.

Do not leave important campaign state/results only in terminal output.

Create umbrella report:

`docs/MILESTONE_10_2_RESULTS.md`

and compact evidence under:

`training/results/milestone10_2/`

with stage subdirectories.

## 13. Failure behavior

On any stage failure/no-learning/exploit/budget stop:

- stop the entire progression;
- preserve current best/recoverable checkpoint and all previously passed stage checkpoints;
- do not retune reward weights mid-stage;
- do not skip ahead;
- write/push honest evidence;
- leave production unchanged.

## 14. Final verification

Before final response/closeout, whether success, failure, or wall-clock exhaustion:

- run production tests;
- run relevant training tests;
- Ruff training code;
- compileall training code/scripts/tests;
- parse all new JSON;
- independently reload current/final checkpoint and held actor outputs;
- verify all passed prerequisite checkpoints remain intact;
- verify initial v10.1 source checkpoint unchanged;
- verify frozen Wisp hashes/configuration unchanged;
- verify no progressive training worker remains;
- verify checkpoint binaries/RLViser remain ignored as appropriate;
- reconcile actual wall time, simulated learner hours, and active learner steps;
- push final stable state;
- verify local `main` == `origin/main`;
- report final commit SHA and exact authority decision.

No production promotion is authorized by this package.