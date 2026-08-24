# Rival v10.2 — Progressive Stage 1→4 Execution Protocol

## Purpose

This file authorizes Codex to implement and execute **Stages 1 through 4 sequentially without waiting for a human between successful stages**.

The package is prerequisite-driven, not time-driven.

Codex may advance only after the current stage emits its exact success decision. If a stage hits any documented stop/failure condition, Codex must stop the entire progression, preserve evidence/checkpoints, push the stable closeout, and report the failed prerequisite. It must **never skip a failed stage**.

## Authorized state machine

```text
Stage 0 — locomotion (already learned in v10.1)
        |
        v
Stage 1 — ball acquisition
        |
        | exact decision: ball_acquisition_skill_passed_unlock_ground_control
        v
Stage 2 — ground control / dribbling
        |
        | exact decision: ground_control_skill_passed_unlock_aerial_control
        v
Stage 3 — aerial acquisition / air-dribble control
        |
        | exact decision: aerial_control_skill_passed_unlock_finishing
        v
Stage 4 — finishing / scoring
        |
        | exact decision: finishing_skill_passed_unlock_opponent_pressure
        v
STOP FOR HUMAN REVIEW
```

Stage 5 opponent pressure and Stage 6 self-play are **not authorized** by this package.

## Universal stage-transition contract

At every successful stage transition:

1. Finish the current evaluation boundary completely.
2. Independently reload the passing checkpoint and reproduce held deterministic outputs.
3. Write the current stage results/decision and compact evidence.
4. Preserve the exact passing actor checkpoint immutably.
5. Push a coherent stable Git boundary before beginning the next stage.
6. Start the next stage from **actor weights only** from that passing checkpoint.
7. Initialize a fresh `RivalCriticV1`.
8. Initialize fresh actor Adam state.
9. Initialize fresh critic Adam state.
10. Run the next stage's implementation/preflight gates before consuming its real training budget.
11. Measure the previous skill on the next stage's frozen retention corpus before the first next-stage PPO update.

Do not carry critic values or optimizer moments across reward-contract changes.

Do not reset the actor to an earlier stage unless a documented checkpoint-integrity failure forces a stop.

## Frozen global architecture

All Stages 1–4 use the same:

- `RivalPolicyV1` actor topology;
- `RivalObsV1` 714-float observation;
- `RivalActionV1` native physical controls;
- 120-Hz physics and 120-Hz policy;
- one-tick action delay;
- shared canonical RocketSim/RLBot state adapters;
- CPU/export/live deployment contract;
- one active trainable learner plus one inert non-learning dummy during isolated skill lessons.

No Wisp/Nexto actor or action lookup enters the scratch policy.

No active opponent is authorized through Stage 4.

## Skill reward removal principle

Once a prerequisite passes, its direct reward must be **removed or substantially reduced** in the next lesson.

The intended progression is:

### Stage 1

```text
closer to ball -> small reward
touch ball     -> maximum event reward
```

No scoring reward.

### Stage 2

```text
first touch          -> small bridge
follow-up touch      -> maximum event reward
keep ball reachable  -> small bounded bridge
```

Generic speed reward remains zero. Scoring reward remains zero.

### Stage 3

```text
reach airborne ball    -> small bounded bridge
first aerial touch     -> acquisition reward
follow-up aerial touch -> maximum control reward
```

Ground-control reward is not allowed to dominate. Scoring reward remains zero.

### Stage 4

```text
retain contact/control -> small bounded bridge
ball toward target     -> small bounded bridge
score correct goal     -> +10 terminal maximum
own goal               -> -10
```

This is the first stage where scoring is a learning objective.

## Stage-specific authority

### Stage 1

Authority files:

- `BALL_ACQUISITION_REWARD.md`
- `BALL_ACQUISITION_CURRICULUM.md`
- `EVALUATION_AND_EXIT_GATES.md`
- `IMPLEMENTATION_GATES.md`
- `M10_2_CAMPAIGN.json`

Maximum: 15 learner-simulated hours.

### Stage 2

Authority file:

- `STAGE_2_GROUND_CONTROL.md`

Maximum: 20 learner-simulated hours.

### Stage 3

Authority file:

- `STAGE_3_AERIAL_CONTROL.md`

Maximum: 30 learner-simulated hours.

### Stage 4

Authority file:

- `STAGE_4_FINISHING.md`

Maximum: 25 learner-simulated hours.

Total maximum authorized learner-simulated experience across Stages 1–4 if every stage consumes its full budget:

**90 hours = 38,880,000 active-learner 120-Hz steps.**

This is a ceiling, not a target. Stop each stage immediately once its final two-boundary success requirement is satisfied.

## Common preflight for Stages 2–4

Before real training begins in each later stage, Codex must create and execute a focused preflight proving at least:

1. exact actor-only load from the prior passing checkpoint;
2. prior checkpoint remains byte-identical after load;
3. fresh critic initialization;
4. fresh actor/critic optimizer states;
5. reward truth-table tests including exact zero reward for locked future skills;
6. authoritative physical touch/event classification tests relevant to that stage;
7. dummy agent cannot enter PPO rows/loss/GAE;
8. at least 10,000 randomized resets for every active curriculum phase/family with finite/legal physics and balanced team/mirror geometry;
9. frozen deterministic gate corpus is created before the first real PPO update;
10. disjoint generalization corpus is created before the first real PPO update;
11. source actor is evaluated on the new stage's gate/retention corpus;
12. one disposable real CUDA PPO update passes with finite gradients in analog/button actor heads and critic;
13. disposable update is discarded and the real campaign starts again from the exact transferred actor plus fresh critic/optimizers;
14. frozen production Wisp hashes/config remain unchanged.

If any preflight fails, stop before spending real stage experience.

## Campaign implementation requirement

Codex should implement one resumable **progressive skill campaign controller** (name may follow repository conventions) that records:

- current stage;
- stage phase;
- source checkpoint path/hash;
- stage-active learner steps;
- stage simulated hours;
- total progressive learner steps/hours;
- current evaluation boundary;
- gate decision;
- exact passing checkpoint for completed stages;
- next authorized stage;
- stop reason if any.

State must be atomic/recoverable. Re-running after interruption must resume from the latest clean stage boundary or rolling recovery checkpoint without spending a completed stage twice.

The controller may orchestrate separate stage-specific trainers/environments; it does not need to force all rewards into one giant class.

## PPO defaults

Unless a stage document explicitly overrides them, use the Stage-1 one-active-learner PPO physical-time defaults:

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
- max grad norm `1.0`.

At every new stage start, a bounded worker-count check is allowed only if the new environment materially changes throughput. Select using stable **trainable learner steps/sec**, not dummy-inclusive raw transitions.

## Evaluation discipline

For every stage:

- deterministic gate corpora are frozen before training;
- source actor baseline is measured before training;
- boundaries are independently checkpointed/reloaded before evaluation;
- training reward/loss/entropy/throughput are diagnostics, not capability proof;
- unseen generalization corpus is run only on apparent passes;
- exact stage success requires the documented consecutive-pass rule;
- prior-stage retention gates must pass where specified;
- accidental goals in Stages 1–3 are reward-neutral and cannot be capability evidence.

## Failure behavior

On **any** stage failure/stop decision:

1. stop all Rival training workers for this campaign;
2. preserve the best/recoverable checkpoint;
3. preserve all already-passed prerequisite checkpoints;
4. write the stage result and umbrella progressive summary;
5. state exactly why the prerequisite failed;
6. do not modify reward weights to chase the gate;
7. do not advance to the next stage;
8. push the stable closeout to `origin/main`;
9. leave frozen production unchanged.

## Successful Stage-4 closeout

When Stage 4 passes:

1. preserve exact Stage-1, Stage-2, Stage-3, and Stage-4 passing actor checkpoints;
2. write `docs/MILESTONE_10_2_RESULTS.md` as an umbrella report with all lesson results;
3. write compact machine-readable evidence under `training/results/milestone10_2/` with per-stage subdirectories;
4. record final authority decision:

`finishing_skill_passed_unlock_opponent_pressure`

5. stop. Do not start opponent pressure or self-play.
6. no production promotion is authorized.

## Repository progress preservation

Push coherent commits at minimum:

- after progressive implementation/preflights;
- after every evaluation boundary;
- at every stage transition;
- at any failure closeout;
- at final Stage-4 closeout.

Do not leave the only copy of campaign authority, results, or stage decisions in terminal output/chat.