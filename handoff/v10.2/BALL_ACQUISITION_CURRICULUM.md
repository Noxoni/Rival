# Rival v10.2 — Ball Acquisition Curriculum

## Purpose

The curriculum should make the next prerequisite learnable without asking Rival to solve possession, scoring, defense, or opponent interaction at the same time.

The active learner's only task is:

> **Reach the ball and touch it.**

The reset distributions are broad and randomized. They are not fixed training-pack shots and do not encode a solution trajectory.

## Environment roles

Keep two cars present so `RivalObsV1` remains unchanged.

Per episode:

- randomly select blue or orange as the **active learner**;
- the active learner is controlled by `RivalPolicyV1` and contributes PPO experience;
- the other car is a **non-interfering dummy** with zero controller output and no policy/critic update;
- place the dummy far enough from the acquisition geometry that it cannot reach or disturb the ball before the active learner under ordinary legal physics;
- randomize dummy side/position so the policy does not learn one fixed irrelevant opponent vector.

If the current `rlgym-ppo` manager cannot natively exclude the dummy agent from PPO updates, implement the smallest Rival-owned wrapper needed to mask dummy trajectories from loss construction while leaving the observation/action contract untouched. Do not train a second copy of Rival against the learner in this stage.

## Phase A reset mixture

Use five acquisition families:

| Family | Weight | Purpose |
|---|---:|---|
| `stationary_close` | 30% | Learn basic steering/throttle convergence to an easy nearby ball |
| `stationary_medium` | 25% | Extend acquisition across meaningful field distance |
| `moving_chase` | 20% | Learn to track a ball that is already translating |
| `awkward_heading` | 20% | Require turning/reorientation instead of simply driving straight |
| `natural_kickoff_holdout` | 5% | Preserve a small natural-state check without dominating learning |

Weights are reset probabilities, not elapsed-time guarantees.

## Family definitions

### `stationary_close` — 30%

- ball resting on the ground;
- active car ground distance from ball broadly `400–1400 uu`;
- randomized relative bearing over the full horizontal circle;
- car may begin stationary or with modest legal planar velocity;
- randomized yaw independent of ball bearing;
- boost broadly randomized `0–60`;
- no requirement that the car initially face the ball.

This family should make first successful contacts common enough to provide a strong event signal while still requiring steering.

### `stationary_medium` — 25%

- ground ball stationary;
- active car distance broadly `1400–3500 uu`;
- broad full-circle relative bearing;
- randomized yaw;
- starting planar speed from stationary through moderate movement;
- boost `0–80`.

The learner must retain the already-learned ability to move quickly but receives no reward for speed itself.

### `moving_chase` — 20%

- ground or low-bounce ball with legal initial height;
- ball planar speed broadly `200–1400 uu/s`;
- ball direction randomized independently of active-car heading;
- initial active-car distance broadly `700–3000 uu`;
- include both same-direction chase and lateral/intercept geometries;
- avoid high aerial trajectories in this stage;
- boost `10–80`.

The objective remains physical acquisition, not predicting an optimal shot.

### `awkward_heading` — 20%

- ball ground/low and reachable;
- distance broadly `500–2500 uu`;
- deliberately sample many starts where the ball is behind or far off the car's forward axis;
- randomized legal car speed, including lateral/away-from-ball motion;
- randomized yaw and moderate angular velocity;
- boost `0–70`.

This prevents the stage from degenerating into a straight-line throttle lesson.

### `natural_kickoff_holdout` — 5%

- normal symmetric Rocket League 1v1 kickoff state;
- one Rival active learner, one non-interfering dummy rather than a competitive opponent;
- ordinary ball-at-center geometry and legal kickoff spawn.

This family is diagnostic/minority training exposure. It must not dominate readiness.

## Episode termination

Terminate/reset on:

- active learner makes a goal only because Soccar requires state reset — reward remains zero;
- no active-learner touch for **12 seconds**;
- ordinary maximum episode duration **45 seconds**;
- invalid/non-finite physics.

Do **not** terminate immediately on first touch. Repeated touches are desirable and should remain possible after initial acquisition.

However, evaluation must separately record first-touch acquisition success/time so later repeated touches do not hide failure to acquire from the initial state.

## Phase B within v10.2

Do not create a new reward when the learner improves. If Phase A readiness is passed once, continue the same acquisition reward for another evaluation interval with a harder distribution:

| Family | Weight |
|---|---:|
| stationary_close | 10% |
| stationary_medium | 25% |
| moving_chase | 30% |
| awkward_heading | 25% |
| natural_kickoff_holdout | 10% |

Widen medium/moving geometry within legal bounds, but still do not introduce elevated aerial-control tasks or active opponents.

The purpose of Phase B is retention/generalization, not a new skill.

## Symmetry and randomization

Every family must randomize:

- active team;
- left/right field geometry;
- starting yaw;
- ball/car relative bearing;
- starting boost within family bounds;
- legal starting speed within family bounds.

Use the existing episode-stable X mirror augmentation consistently where applicable.

## What this curriculum intentionally excludes

No:

- Wisp or Nexto opponent;
- current-policy self-play;
- defender/challenger;
- dribble carry states;
- aerial/ceiling possession;
- recovery-specific resets;
- easy-finishing states;
- goal-direction curriculum;
- named mechanics.

Those are later dependencies.
