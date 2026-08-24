# Rival v10.2 — Ball Acquisition Reward

## Objective

Teach exactly one missing prerequisite:

> **Move the car so that it closes distance to the ball, then physically touch the ball.**

The v10.1 actor already learned locomotion/speed. v10.2 must therefore pay nothing merely for being fast.

## Reward components

`RivalBallAcquisitionRewardV1` has only these learning components.

### 1. Car-caused distance progress

At each native 120-Hz transition, compute progress using the **current ball position on both sides of the comparison** so a stationary car is not rewarded merely because the ball rolls toward it:

```text
car_progress_uu =
    distance(previous_car_position, current_ball_position)
    - distance(current_car_position, current_ball_position)
```

Interpretation:

- positive: the car's own movement reduced separation to the current ball location;
- negative: the car's own movement increased separation;
- approximately zero: the car did not materially change its distance to the current ball location.

Reward proposal:

```text
distance_progress_reward = car_progress_uu / 2300.0
```

Do not add any speed, facing, boost, or controller-input term to this quantity.

The signed reward is intentionally simple. Moving closer pays; moving away costs. This prevents repeatedly backing away and re-approaching from becoming free positive reward.

Because a 120-Hz transition can contain pathological one-tick geometry, apply a conservative per-transition safety clip only after computing the exact signed quantity. Record both unclipped and clipped values. The clip must be selected from observed legal transition statistics during preflight and documented; do not choose a value that materially truncates ordinary acquisition movement.

### 2. Physical touch event

Every genuine **new physical ball touch** by the active learner pays:

```text
+1.0
```

This is the largest individual reward event in v10.2.

Repeated touches are explicitly desirable. A later genuine touch receives another `+1.0`.

Do **not**:

- impose a chain penalty;
- decay touch value because the learner already touched the ball;
- require the touch to send the ball toward goal;
- require a minimum ball velocity change;
- distinguish ground versus aerial touch for reward;
- suppress a genuine separated retouch because it occurred quickly.

However, sustained physical contact must not yield `+1.0` on every 120-Hz tick. Implement a true new-contact detector from the authoritative RocketSim/RLGym touch/contact semantics. Credit each actual new touch/contact event once. Audit this against synthetic and live RocketSim traces.

### 3. Everything else is zero

Explicitly zero learning reward for:

- raw planar or 3D speed;
- throttle;
- steer/yaw/pitch/roll activity;
- boost use;
- jump;
- dodge/flip;
- handbrake;
- recovery alignment;
- ball progress toward either goal;
- possession duration;
- aerial height;
- scoring;
- conceding;
- opponent-relative positioning.

A goal still terminates/resets the environment, but:

```text
goal_for_reward = 0.0
goal_against_reward = 0.0
```

Scoring is a locked future skill.

## Reward hierarchy and budgets

The touch event is the semantic maximum. Dense distance shaping exists only to bridge the sparse gap between an arbitrary reset and first contact.

Use an **absolute per-episode budget of 0.75** for signed distance-progress shaping. Once the absolute spend reaches `0.75`, distance shaping is clipped to zero for the remainder of that episode.

Physical touch rewards are **not** included in that `0.75` budget. Repeated genuine touches continue to earn `+1.0` each because repeated ball interaction is exactly the behavior this stage wants.

There is no outcome reward in this stage.

This guarantees:

- one real touch is worth more than the entire maximum dense acquisition shaping available in that episode;
- endlessly driving without contact cannot outscore one touch merely by farming dense reward;
- repeated real touches remain strongly valuable.

## Cadence requirements

The reward operates on native 120-Hz transitions and must remain numerically stable if diagnostic code aggregates several physics ticks.

Distance progress is geometric displacement, not a per-second rate copied from a lower-frequency environment. Do not multiply a 30-Hz coefficient by 120-Hz frequency.

Touch reward is an event and is never cadence-scaled.

## Required telemetry

At every evaluation/reporting boundary record at least:

- cumulative signed distance-progress reward;
- cumulative absolute distance-progress spend;
- fraction of episodes where the `0.75` dense budget saturates;
- true physical touch count;
- touches per 100k agent-steps;
- touch reward total;
- mean/median car-caused progress per active tick;
- mean/median time to first touch;
- no-touch timeout share;
- mean initial and terminal ball distance;
- goal count as a diagnostic only, clearly marked reward-neutral.

## Anti-exploit checks

Preflight must explicitly verify:

1. stationary car + ball rolling toward car does not produce meaningful positive car-progress reward;
2. car driving away from a stationary ball produces negative reward;
3. car driving toward a stationary ball produces positive reward;
4. approach → retreat → approach does not create net free dense reward apart from unavoidable clipping precision;
5. one sustained contact is counted once, not once per native tick;
6. two genuinely separated contacts are both rewarded;
7. scoring with no touch gives zero outcome reward;
8. goal/concede cannot override the touch/distance contract;
9. no reward implementation accepts or reads the controller/action vector directly.
