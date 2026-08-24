# Rival Stage 3 — Aerial Acquisition / Air-Dribble Control

## Unlock condition

Stage 3 is **locked** until Stage 2 ends with exactly:

`ground_control_skill_passed_unlock_aerial_control`

Start from the exact preserved passing Stage-2 actor checkpoint. Transfer actor weights only; initialize a fresh `RivalCriticV1` and fresh actor/critic Adam states because the reward changes.

Preserve all Stage-1 and Stage-2 passing checkpoints byte-for-byte.

## Objective

Teach the next prerequisite:

> **Intentionally leave the ground, reach an elevated moving ball, make an aerial contact, and then sustain additional controlled aerial contacts.**

This stage teaches aerial acquisition and air-dribble-like control. It does **not** teach scoring yet.

The policy must already own locomotion, ball acquisition, and repeated ground control. Those remain prerequisites and are measured for retention rather than paid heavily again.

## Frozen architecture

Do not change:

- `RivalPolicyV1`;
- `RivalObsV1` (714 floats);
- `RivalActionV1`;
- native 120-Hz controller decisions;
- one-tick transport;
- canonical adapters;
- actor topology/export path.

No active opponent is used. Keep the inert dummy only for the unchanged observation contract and exclude dummy rows from PPO.

## Reward — `RivalAerialControlRewardV1`

Stage 3 has two phases under one versioned reward family.

### Shared rules

Zero direct reward for:

- speed;
- throttle/steer/yaw/pitch/roll inputs;
- jump button use;
- boost button use;
- dodge use;
- recovery;
- ball-to-goal progress;
- scoring/conceding;
- named mechanics.

Goals terminate/reset but reward `0.0`.

Aerial credit is based on physical contact plus authoritative car/ball state, not on action presses.

Define an `aerial_touch` as a genuine new learner-ball contact where, at contact:

- learner is not in surface contact; and
- learner center Z >= 80 uu; and
- ball center Z >= 220 uu.

Record exact state values for every classified event. Do not reward ground touches as aerial touches.

### Phase A — aerial acquisition

Before the first aerial touch, use a small 3D car-caused distance bridge:

```text
progress_3d_uu =
    distance(previous_car_position, current_ball_position)
    - distance(current_car_position, current_ball_position)

acquisition_reward = progress_3d_uu / 4600.0
```

Absolute per-episode budget: **0.50**.

Every genuine new aerial touch pays:

```text
+1.0
```

Ground touches pay `0.0` in Stage-3 Phase A.

Phase A is about reaching the airborne ball. Repeated aerial touches still each receive +1.0 and are diagnostic for unlocking the control phase.

### Phase B — aerial follow-up / air control

Once Phase A passes, reduce the first aerial touch to:

```text
+0.25
```

Every later genuine aerial touch in the same air-control chain pays:

```text
+1.0
```

An aerial chain remains active while:

- no surface contact has persisted for >0.35 s after the first aerial touch; and
- no more than 2.0 s elapsed since the previous aerial touch; and
- 3D car-ball separation has not exceeded 1400 uu for >0.5 s.

A short skim/landing that immediately continues into the same reachable aerial may be recorded diagnostically, but for reward purposes a sustained surface-contact break ends the chain.

After the first aerial touch, allow a small control-envelope bridge:

```text
distance_quality = clip(1 - distance_3d(car, ball) / 900, 0, 1)
relative_speed_quality = clip(1 - ||ball_vel - car_vel|| / 2200, 0, 1)
control_quality = distance_quality * relative_speed_quality
control_reward = 0.10 * control_quality * delta_seconds
```

Absolute post-first-touch control-envelope budget: **0.50** per episode.

Pre-first-touch 3D distance budget in Phase B is reduced to **0.25**.

The semantic maximum remains a real follow-up aerial touch.

## Curriculum — `RivalAerialControlCurriculumV1`

### Phase A reset mix

- `low_aerial_reach`: 35%
- `medium_aerial_reach`: 30%
- `moving_aerial_intercept`: 20%
- `awkward_aerial_approach`: 10%
- `ground_control_retention`: 5%

### `low_aerial_reach`

- learner starts on/near floor;
- ball Z roughly 250–600 uu;
- horizontal separation 400–1800 uu;
- ball vertical velocity broad but reachable;
- learner boost 20–100;
- broad yaw offsets.

### `medium_aerial_reach`

- ball Z roughly 550–1100 uu;
- horizontal separation 700–2400 uu;
- moving ball with varied lateral/vertical velocity;
- enough boost to make the state physically reachable.

### `moving_aerial_intercept`

- ball traverses laterally or diagonally through a reachable aerial window;
- learner must predict/intercept rather than drive beneath a stationary target.

### `awkward_aerial_approach`

- heading and lateral geometry deliberately misaligned;
- ball remains reachable with correct turn/jump/boost/orientation sequence.

### `ground_control_retention`

Use Stage-2-like states as diagnostics with Stage-3 reward. These do not need to dominate PPO; they exist to detect catastrophic forgetting.

### Phase B reset mix

Unlock after one Phase-A pass:

- `aerial_followup`: 30%
- `rising_ball_control`: 25%
- `lateral_air_dribble`: 20%
- `wall_to_air_entry`: 15%
- `medium_aerial_reach`: 5%
- `ground_control_retention`: 5%

### `aerial_followup`

Start from a broad physically plausible state immediately after a first aerial-contact-like event: car and ball airborne, nearby, with similar but non-identical velocities. Do not provide a scripted action sequence.

### `rising_ball_control`

Ball and learner are moving upward/forward with reachable relative geometry. Objective is repeated contacts as the play evolves vertically.

### `lateral_air_dribble`

Car and ball begin airborne with substantial lateral component and varying pitch/yaw/roll demands. Avoid only straight-line vertical examples.

### `wall_to_air_entry`

Learner starts on/near sidewall with elevated ball leaving the wall into open space. Keep this a minority family until open-air control is established.

No ceiling-reset or named-mechanic starts in Stage 3.

## Deterministic evaluation

Freeze a seeded corpus before Stage-3 PPO begins and a disjoint unseen corpus.

At least 100 episodes each for:

- low_aerial_reach;
- medium_aerial_reach;
- moving_aerial_intercept;
- awkward_aerial_approach;
- aerial_followup;
- rising_ball_control;
- lateral_air_dribble;
- wall_to_air_entry.

Run Phase-B-only families diagnostically before unlock but do not require them for Phase A.

Record:

- first aerial-touch success;
- time to first aerial touch;
- `>=2`, `>=3`, `>=4` aerial-touch chain success;
- longest aerial chain;
- controlled airborne duration after first touch;
- 3D car-ball separation after first touch;
- boost/jump/dodge/action traces as diagnostics only;
- surface-contact chain breaks;
- ground-control retention suite;
- goals as reward-neutral diagnostics.

## Phase A readiness

One Phase-A pass requires:

- low_aerial_reach first aerial touch >=90%;
- medium_aerial_reach >=82%;
- moving_aerial_intercept >=72%;
- awkward_aerial_approach >=72%;
- aggregate first-aerial-touch success >=80%;
- median successful time-to-first-aerial-touch <=4.5 s;
- no-aerial-touch timeout share <=20%;
- Stage-2 ground-control retention >=80% of the exact Stage-2 passing checkpoint's aggregate `>=3`-touch success.

One pass unlocks Phase B only.

## Final Stage-3 gate

Stage 3 passes after **two consecutive Phase-B boundaries** meeting all frozen-corpus requirements and unseen generalization requirements.

Frozen corpus:

- first aerial touch aggregate >=90%;
- aerial_followup `>=2` aerial-touch chain >=85%;
- rising_ball_control `>=2` >=75%;
- lateral_air_dribble `>=2` >=70%;
- wall_to_air_entry `>=2` >=60%;
- aggregate `>=2` aerial-chain success >=75%;
- aggregate `>=3` aerial-chain success >=50%;
- at least 30% of Phase-B episodes achieve `>=4` aerial touches;
- median controlled airborne duration after first touch >=1.5 s;
- median first-to-third-aerial-touch time <=5.5 s;
- Stage-1 acquisition retention >=85% of its passing gate;
- Stage-2 ground-control retention >=80% of its passing gate.

Unseen corpus:

- first aerial touch aggregate >=85%;
- aggregate `>=2` aerial-chain success >=65%;
- no core family below 55%;
- ground-control retention remains >=75% of Stage-2 pass baseline.

Success decision:

`aerial_control_skill_passed_unlock_finishing`

Preserve the exact passing Stage-3 actor for Stage 4.

## Budget and boundaries

Maximum Stage-3 budget: **30 simulated learner hours** = 12,960,000 active-learner 120-Hz steps.

Evaluate at:

`+2.5, +5, +10, +15, +20, +25, +30` Stage-3 hours.

No-learning stop at +10h if first-aerial-touch aggregate improved <10 percentage points from the Stage-3 source actor and `>=2` aerial-chain success improved <8 points.

Hard stop if not mastered by +30h:

`stop_aerial_control_not_mastered_by_plus_30h`

Do not silently change reward/curriculum. Preserve evidence and stop.