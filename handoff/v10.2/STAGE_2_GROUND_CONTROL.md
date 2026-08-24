# Rival Stage 2 — Ground Ball Control / Dribbling

## Unlock condition

Stage 2 is **locked** until Stage 1 ends with exactly:

`ball_acquisition_skill_passed_unlock_ground_control`

The source actor for Stage 2 is the exact preserved passing Stage-1 actor checkpoint. Preserve that checkpoint byte-for-byte.

Transfer **actor weights only**. Initialize a fresh `RivalCriticV1`, fresh actor Adam state, and fresh critic Adam state because the reward contract changes.

Do not start Stage 2 from v10.1 or from a non-passing Stage-1 checkpoint.

## Objective

Teach the next prerequisite:

> **After reaching the ball, keep it reachable and produce repeated controlled ground contacts instead of hitting it once and losing it.**

This stage is ball control, not scoring.

The policy already knows locomotion and must have proved acquisition. Those primitives become tools; they are no longer primary reward targets.

## Frozen architecture

Do not change:

- `RivalPolicyV1`;
- `RivalObsV1` (714 floats);
- `RivalActionV1`;
- native 120-Hz policy/physics cadence;
- one-tick action delay;
- canonical RocketSim/RLBot adapters;
- actor topology/export path.

No active opponent is used in Stage 2. Retain the inert second car only to preserve the observation contract and exclude it from PPO rows.

## Reward — `RivalGroundControlRewardV1`

### 1. Pre-control reacquisition bridge

Before the learner's first physical touch in an episode, retain a **reduced** Stage-1 car-caused distance-progress signal:

```text
car_progress_uu =
    distance(previous_car_position, current_ball_position)
    - distance(current_car_position, current_ball_position)

reacquisition_reward = car_progress_uu / 4600.0
```

Absolute per-episode budget: **0.25**.

After the first touch, this term becomes zero for the rest of the episode.

There is no speed, facing, boost, jump, dodge, ball-to-goal, or controller-input reward.

### 2. First touch

The first genuine new physical learner touch in an episode receives:

```text
+0.25
```

Stage 1 already taught first contact, so first touch is now only a bridge into the actual lesson.

### 3. Follow-up / repeated physical touches

Every genuine **new separated self touch after the first** receives:

```text
+1.0
```

Repeated touches are the semantic maximum of Stage 2 and are **not budgeted**.

Continuous contact is one event, not one reward per 120-Hz tick. Use authoritative contact/touch semantics and a separation-aware event detector.

A chain remains active while:

- no other car has touched the ball (the dummy should not); and
- no more than 2.5 seconds have elapsed since the learner's previous physical touch; and
- car-ball separation has not exceeded 1500 uu for more than 0.5 continuous seconds.

If the chain breaks, the next learner touch starts a new chain and is treated as a first touch (`+0.25`) rather than a follow-up touch.

### 4. Controlled-proximity bridge

After a learner touch and while the current chain remains active, allow a small cadence-safe control-envelope rate:

```text
planar_distance = ||ball_xy - car_xy||
relative_speed = ||ball_velocity - car_velocity||

distance_quality = clip(1 - planar_distance / 800, 0, 1)
relative_speed_quality = clip(1 - relative_speed / 2000, 0, 1)
control_quality = distance_quality * relative_speed_quality

control_reward = 0.10 * control_quality * delta_seconds
```

Absolute per-episode budget: **0.50**.

This term exists only to bridge the delay between physical follow-up touches. It must never outweigh repeated-touch events.

### 5. Everything else is zero

Explicitly zero reward for:

- raw speed;
- action/button use;
- aerial height;
- scoring/conceding;
- ball progress toward either goal;
- recovery;
- boost economy;
- opponent positioning.

A goal may terminate/reset the environment but has reward `0.0` in Stage 2.

## Curriculum — `RivalGroundControlCurriculumV1`

### Phase A — learn to retouch

Reset mix:

- `followup_close`: 40%
- `rolling_push`: 25%
- `turn_recontrol`: 20%
- `acquisition_to_control`: 10%
- `natural_free_control_holdout`: 5%

Broad randomized families:

### `followup_close`

Start after a plausible first-touch-like state:

- ball 180–500 uu in front/lateral of learner;
- ball ground/bounce height 93–220 uu;
- learner and ball velocities broadly similar but not identical;
- ball speed 200–1200 uu/s;
- learner boost 10–100;
- heading error up to about 45 degrees.

Objective: obtain a second and third physical touch without losing the ball.

### `rolling_push`

- ball rolling 300–1400 uu/s;
- learner 250–800 uu behind or slightly lateral;
- randomized field location/direction;
- no goal-direction preference.

Objective: maintain a sequence of useful contacts while both car and ball travel.

### `turn_recontrol`

- ball within 250–700 uu;
- learner heading error 45–150 degrees;
- low/moderate initial speed;
- randomized left/right geometry.

Objective: turn back into control rather than drive away.

### `acquisition_to_control`

Use broad Stage-1-style medium/moving acquisition states but continue after first touch. This ensures the learned acquisition primitive feeds into control instead of becoming dependent on pre-positioned starts.

### `natural_free_control_holdout`

Normal center-ball/free-play-style start with inert dummy; reward remains Stage-2-only and goals are neutral.

### Phase B — actual dribble/control generalization

Unlock after one Phase-A pass.

Reset mix:

- `followup_close`: 20%
- `rolling_push`: 25%
- `turn_recontrol`: 20%
- `carry_control`: 25%
- `acquisition_to_control`: 10%

`carry_control` adds broad states where the ball is 110–350 uu above/near the hood or immediately in front of the learner, with low-to-moderate relative speed. Do not hard-code a dribble macro or require one exact ball-on-roof geometry.

## Deterministic evaluation

Freeze seeded corpora before the first Stage-2 PPO update.

At least 100 episodes each for:

- followup_close;
- rolling_push;
- turn_recontrol;
- acquisition_to_control;
- carry_control (Phase B; diagnostic before unlock).

Maintain a disjoint unseen corpus of at least 250 episodes.

Record per family:

- first-touch success;
- `>=2` learner-touch chain success;
- `>=3` learner-touch chain success;
- longest chain length;
- median time from first to second touch;
- median time from first to third touch;
- fraction of post-first-touch time ball stays within 1000 uu;
- mean/median car-ball distance after first touch;
- chain-break reasons;
- no-touch and no-retouch timeout shares;
- speed/action diagnostics only as diagnostics;
- goal count as reward-neutral diagnostic.

## Phase A readiness

A boundary passes Phase A only if all hold on the frozen corpus:

- followup_close: `>=2` touches in one chain in >=90% of episodes;
- rolling_push: `>=2` touches in >=80%;
- turn_recontrol: `>=2` touches in >=75%;
- acquisition_to_control: first touch >=85% and `>=2` chain >=70%;
- aggregate `>=2` chain success >=80%;
- aggregate `>=3` chain success >=55%;
- median first-to-second-touch time <=3.0 s;
- after first touch, ball is within 1000 uu for >=65% of evaluated control time.

One Phase-A pass unlocks Phase B only.

## Final Stage-2 gate

Stage 2 passes only after **two consecutive Phase-B boundaries** satisfying the frozen corpus and, at each apparent pass, the unseen corpus.

Frozen-corpus requirements:

- followup_close `>=3` chain success >=92%;
- rolling_push `>=3` >=85%;
- turn_recontrol `>=3` >=82%;
- carry_control `>=3` >=75%;
- acquisition_to_control first touch >=90%, `>=3` chain >=75%;
- aggregate `>=3` chain success >=82%;
- aggregate `>=4` chain success >=60%;
- ball within 1000 uu for >=75% of post-first-touch control time;
- median first-to-third-touch time <=5.0 s.

Unseen-corpus requirements:

- aggregate `>=3` chain success >=75%;
- no family below 65%;
- Stage-1 acquisition retention >=85% of the exact Stage-1 passing checkpoint's frozen-corpus success rate.

Success decision:

`ground_control_skill_passed_unlock_aerial_control`

Preserve the exact passing Stage-2 actor checkpoint for Stage 3.

## Budget and boundaries

Maximum Stage-2 budget: **20 simulated learner hours** = 8,640,000 active-learner 120-Hz steps.

Evaluate at:

`+1, +2.5, +5, +7.5, +10, +15, +20` Stage-2 hours.

Stop for no material learning at +7.5h if aggregate `>=2` chain success has improved <10 percentage points from source and `>=3` chain success improved <8 points.

Hard-stop decision if not mastered by +20h:

`stop_ground_control_not_mastered_by_plus_20h`

Do not retune reward/curriculum mid-stage. Stop and preserve evidence if the gate fails.