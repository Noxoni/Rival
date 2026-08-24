# Rival Stage 4 — Finishing / Scoring With Learned Control

## Unlock condition

Stage 4 is **locked** until Stage 3 ends with exactly:

`aerial_control_skill_passed_unlock_finishing`

Start from the exact preserved passing Stage-3 actor checkpoint. Transfer actor weights only; initialize a fresh `RivalCriticV1` and fresh actor/critic Adam states because the reward changes.

Preserve the passing Stage-1, Stage-2, and Stage-3 checkpoints byte-for-byte.

## Objective

Stage 4 is the first lesson where scoring becomes a positive objective:

> **Use the already learned acquisition, ground-control, and aerial-control skills to deliberately put the ball into the opponent goal.**

This is still an isolated skill lesson, not competitive Rocket League. No active opponent is introduced until Stage 5.

## Frozen architecture

Do not change:

- `RivalPolicyV1`;
- `RivalObsV1`;
- `RivalActionV1`;
- native 120-Hz policy/physics cadence;
- one-tick action delay;
- canonical adapters;
- actor topology/export path.

Keep the inert dummy only to preserve the observation contract; exclude dummy rows from PPO.

## Reward — `RivalFinishingRewardV1`

### 1. Goal outcome

A goal into the designated opponent goal pays:

```text
+10.0
```

An own goal pays:

```text
-10.0
```

This is the terminal semantic maximum.

### 2. Ball-to-target-goal progress bridge

Use a small signed potential/progress bridge based on the ball's movement toward the target goal plane. It exists only to bridge control into a finish and must not dominate goals.

For team-normalized coordinates where opponent goal is +Y:

```text
ball_progress_uu = current_ball_y - previous_ball_y
ball_progress_reward = ball_progress_uu / 12000.0
```

Apply a per-episode absolute budget of **1.0** to this shaping component.

Do not add car speed, button use, facing, boost use, or generic car-to-ball distance reward.

### 3. Control-retention touch bridge

Every genuine learner touch receives a small control-retention reward:

```text
first/new touch = +0.10
```

A genuine follow-up touch within the active possession chain receives:

```text
+0.20
```

Combined absolute touch-shaping budget: **1.0 per episode**.

This is deliberately much smaller than the Stage-2/Stage-3 touch rewards. The prerequisite skills should be retained, but scoring must become the dominant objective.

For aerial-possession families, an aerial follow-up touch still receives only the same +0.20; do not add a separate aerial bonus in Stage 4.

### 4. Everything else is zero

Explicitly zero direct reward for:

- speed;
- action/button use;
- boost use;
- jumping/dodging;
- recovery;
- raw aerial height;
- named mechanics;
- opponent interaction (there is no active opponent yet).

Maximum non-outcome shaping available per episode is therefore <=2.0 absolute before any clipping interactions. One correct goal remains overwhelmingly dominant.

## Curriculum — `RivalFinishingCurriculumV1`

### Phase A — learn to finish from owned possession

Reset mix:

- `ground_possession_finish`: 40%
- `aerial_possession_finish`: 30%
- `awkward_ground_finish`: 15%
- `wall_air_finish`: 10%
- `natural_free_finish`: 5%

All starts are team-normalized and mirrored/randomized. The designated scoring direction is always explicit in canonical coordinates.

### `ground_possession_finish`

Start from broad Stage-2-like controlled-ground states:

- ball already reachable/controllable;
- 1200–4500 uu from target goal;
- broad central/lateral lanes;
- varied car-ball relative velocity and angle;
- enough field space that one-touch rockets are not the only viable solution.

### `aerial_possession_finish`

Start from Stage-3-like airborne control states:

- learner and ball airborne/reachable;
- roughly 1200–4500 uu from target goal;
- varied height 300–1500 uu;
- varied lateral offset and velocity;
- physically plausible boost.

### `awkward_ground_finish`

Ground possession begins off-angle or lateral to goal. The policy must redirect/control rather than merely drive straight.

### `wall_air_finish`

Start from broad sidewall-to-air possession states with a reachable path toward goal. Do not require named wall mechanics.

### `natural_free_finish`

Free-play-style start with inert dummy. Rival must acquire/control/finish without an opponent. This is a holdout for chaining prior skills together.

### Phase B — broader finishing generalization

Unlock after one Phase-A finishing pass.

Reset mix:

- `ground_possession_finish`: 25%
- `aerial_possession_finish`: 25%
- `awkward_ground_finish`: 20%
- `wall_air_finish`: 15%
- `natural_free_finish`: 15%

Increase start distance, lateral offset, ball velocity diversity, and approach geometry within physically reachable ranges. Do not add an active opponent in Stage 4.

## Qualified-goal diagnostics

A plain goal is a valid reward event, but **capability evaluation must distinguish whether the learned prerequisite skill was actually used.**

### Ground-control-qualified goal

A goal is ground-control-qualified if, before the goal in the same possession sequence:

- learner made at least **2 genuine touches**; and
- at least one interval between those touches kept the ball within 1000 uu; and
- the final goal was not an untouched/passive ball trajectory from the reset.

### Aerial-control-qualified goal

A goal is aerial-control-qualified if, before the goal:

- learner made at least **2 genuine aerial touches** in the same aerial-control chain.

These classifications are evaluation evidence, not extra goal reward multipliers.

## Deterministic evaluation

Freeze seeded gate and unseen corpora before the first Stage-4 PPO update.

At least 100 episodes each for:

- ground_possession_finish;
- aerial_possession_finish;
- awkward_ground_finish;
- wall_air_finish;
- natural_free_finish.

Record:

- goals for/own goals;
- goal success rate;
- time to goal;
- touches before goal;
- ground-control-qualified goal rate;
- aerial-control-qualified goal rate;
- no-goal timeout share;
- ball terminal location on failures;
- possession/touch-chain diagnostics;
- retained Stage-1 acquisition, Stage-2 ground-control, and Stage-3 aerial-control probes;
- action/boost/speed diagnostics only as diagnostics.

## Phase A readiness

One Phase-A pass requires:

- ground_possession_finish goal success >=75%;
- aerial_possession_finish >=55%;
- awkward_ground_finish >=60%;
- wall_air_finish >=45%;
- natural_free_finish >=35%;
- aggregate goal success >=60%;
- own-goal rate <=5%;
- ground-control-qualified goals >=55% of ground-family episodes;
- aerial-control-qualified goals >=35% of aerial-family episodes;
- Stage-1 acquisition retention >=85% of Stage-1 pass baseline;
- Stage-2 ground-control retention >=80% of Stage-2 pass baseline;
- Stage-3 aerial-control retention >=75% of Stage-3 pass baseline.

One pass unlocks Phase B only.

## Final Stage-4 gate

Stage 4 passes after **two consecutive Phase-B boundaries** satisfying both frozen and unseen corpora.

Frozen corpus requirements:

- ground_possession_finish goal success >=88%;
- aerial_possession_finish >=72%;
- awkward_ground_finish >=78%;
- wall_air_finish >=62%;
- natural_free_finish >=60%;
- aggregate goal success >=75%;
- own-goal rate <=3%;
- ground-control-qualified goals >=70% of ground-family episodes;
- aerial-control-qualified goals >=55% of aerial-family episodes;
- median successful finish time <=12 s for possession families;
- Stage-1 acquisition retention >=85% of its passing baseline;
- Stage-2 ground-control retention >=82% of its passing baseline;
- Stage-3 aerial-control retention >=78% of its passing baseline.

Unseen corpus requirements:

- aggregate goal success >=68%;
- ground-family aggregate >=75%;
- aerial-family aggregate >=58%;
- ground-control-qualified goal rate >=60%;
- aerial-control-qualified goal rate >=45%;
- own-goal rate <=5%;
- prerequisite retention remains within the same limits above minus at most 5 percentage points.

Success decision:

`finishing_skill_passed_unlock_opponent_pressure`

This is the terminal success of the unattended Stage-1-through-4 package. **Do not proceed into Stage 5 automatically.** Preserve the Stage-4 passing actor and stop for human review before active-opponent training.

## Budget and boundaries

Maximum Stage-4 budget: **25 simulated learner hours** = 10,800,000 active-learner 120-Hz steps.

Evaluate at:

`+2.5, +5, +10, +15, +20, +25` Stage-4 hours.

No-learning stop at +10h if aggregate goal success improved <10 percentage points from the Stage-4 source actor and both qualified-goal rates improved <8 points.

Hard stop if not mastered by +25h:

`stop_finishing_not_mastered_by_plus_25h`

Do not change reward weights/curriculum mid-stage. Preserve the best/recoverable checkpoint and stop.