# RivalAgencyBootstrapRewardV1

## Goal

This temporary reward exists because the scratch policy has not yet crossed the basic motor/interaction barrier. It must make simple useful behavior easy for PPO to discover while keeping scoring as the dominant objective.

The reward hierarchy is:

`goal > sustained useful possession/interactions > single ball touch > useful motion > inactivity`

A goal for Rival remains **+10** and a goal against remains **-10**.

The **maximum absolute non-outcome shaping spend per agent per episode is 7.5**, so no amount of bootstrap shaping can be worth more than one scored goal.

## Components and budgets

| Component | Mechanism | Maximum absolute episode spend |
| --- | --- | ---: |
| `outcome` | `+10` goal / `-10` concede | uncapped event |
| `useful_speed_rate` | cadence-safe rate for actual car speed, weighted toward motion toward the ball | 1.0 |
| `ball_approach_potential` | potential difference for reducing distance / closing / facing the ball | 1.0 |
| `ball_touch_event` | reward per distinct logical self touch | 2.0 |
| `aerial_touch_event` | extra reward for a real airborne touch on an elevated ball | 1.5 |
| `touch_chain_event` | increasing bonus for consecutive self touches before opponent interruption | 1.5 |
| `ball_progress_potential` | small potential reward for moving the ball toward the opponent goal | 0.5 |

Total shaping budget: **7.5**.

Do not carry the normal bootstrap-irrelevant `recovery_potential`, `boost_waste_rate`, or `dodge_resource_event` into this phase. Recovery already learned disproportionately well and the immediate goal is useful agency/ball interaction. These normal terms return after the bootstrap phase.

## 1. Useful speed

Going fast should be positively reinforced, but pure wall-circling must not become the objective.

For each native transition:

```text
speed_norm = clip(||car_velocity|| / 2300, 0, 1)
to_ball = normalize(ball_position - car_position)
velocity_dir = normalize(car_velocity)
toward_ball = max(0, dot(velocity_dir, to_ball))
useful_factor = 0.35 + 0.65 * toward_ball
reward_rate_per_second = 0.015 * speed_norm^2 * useful_factor
useful_speed_rate = reward_rate_per_second * delta_ticks / 120
```

Properties:

- any real speed receives some positive signal;
- moving quickly toward the ball receives the strongest signal;
- standing still receives zero;
- reversing/driving elsewhere can still teach throttle/steering, but is substantially less rewarding than useful speed;
- the 1.0 episode budget prevents endless high-speed farming.

Do not reward throttle, boost, steering magnitude, or action changes directly.

## 2. Ball approach

Use a cadence-safe potential, not a raw per-tick distance payment.

The potential should strongly represent:

- smaller car-ball distance;
- positive closing speed;
- facing the ball.

Reuse the proven canonical geometry and potential-difference pattern from `RivalScratchRewardV1`, but define/version the bootstrap potential separately so the ordinary v9 reward remains unchanged.

Starting bootstrap weight: **0.20**.

Keep the absolute episode budget at **1.0**.

This term answers the most basic learning question: `how do I make the ball get closer to me?`

## 3. Distinct ball touches

Each **distinct logical self touch** earns:

`+0.30`

before chain/aerial bonuses.

A touch reward is an event, not a per-contact-frame rate.

### Anti-contact-farming rule

Do not pay once per physics tick while the hitbox remains in continuous contact.

A rewarded logical touch must be constructed from the actual RocketSim touch event stream and satisfy the existing event semantics plus a minimum separation of **8 native ticks (~66.7 ms)** from the same player's previous rewarded touch unless an opponent touch occurred in between.

If multiple low-level touch records occur during one native tick, aggregate them into one logical rewarded touch.

Track raw touch records and rewarded logical touches separately so this debounce is auditable.

## 4. Aerial touch bonus

A rewarded logical touch receives an additional:

`+0.45`

when both conditions are true at the touch:

- Rival is not in surface contact; and
- ball center `z >= 180 uu`.

This is intentionally broad. It rewards actually reaching an elevated ball, not a named aerial mechanic.

Do not reward airtime by itself. Random jumping/spinning without a ball touch is worth nothing from this component.

## 5. Multi-touch chain

Multiple useful contacts should become increasingly attractive.

Maintain a per-agent possession-touch chain. A chain resets on:

- an opponent logical touch;
- a goal;
- episode reset;
- more than **2.5 seconds** since Rival's previous rewarded logical touch.

Additional bonus by touch number in the current chain:

| Touch in chain | Additional bonus |
| ---: | ---: |
| 1 | 0.00 |
| 2 | +0.10 |
| 3 | +0.20 |
| 4 | +0.35 |
| 5+ | +0.50 each, subject to budget |

Therefore a fifth aerial touch can propose:

`0.30 base + 0.45 aerial + 0.50 chain = 1.25`

before episode-budget clipping.

This intentionally makes controlled repeated interaction much more valuable than one accidental collision.

Record maximum chain length and full chain-length histogram in training/evaluation metrics.

## 6. Ball progress

Keep only a small attacking-progress potential so the policy receives a hint that touches which move the ball toward the opponent goal are preferable to arbitrary touches.

Starting weight: **0.05** with a **0.5** absolute episode budget.

Do not let this term dominate touch discovery.

## 7. Goal remains ultimate

A goal is `+10`; conceding is `-10`.

No goal reward multiplier is needed because the entire non-outcome shaping budget is bounded to 7.5.

The policy can therefore bootstrap from easy dense signals without learning that reward farming is superior to finishing the play.

## No raw control rewards

Explicitly forbidden:

- reward for pressing jump;
- reward for pressing boost;
- reward for steering/yaw/roll magnitude;
- reward for changing actions frequently;
- reward for airtime without interacting with the ball;
- named mechanic identity rewards.

Control-use metrics should still be logged. They are diagnostics, not objectives.

## Required reward tests before PPO

1. `0` reward for a stationary no-event transition except potential numerical effects that must themselves be zero at unchanged state.
2. Higher useful forward speed toward the same ball yields greater speed reward than standing still.
3. Equal speed directed away from the ball yields less speed reward than toward it.
4. One distinct ground touch yields exactly the expected base event proposal.
5. One aerial touch yields base + aerial proposal.
6. Touch 2/3/4/5 in one chain receives the documented increasing bonus.
7. Opponent touch and 2.5-second gap reset the chain.
8. Continuous contact across adjacent ticks cannot produce a touch-reward explosion.
9. Total shaping absolute spend can never exceed 7.5.
10. `+10/-10` goal events remain unchanged and not budget-clipped.
11. All rate terms integrate equivalently across 1/2/4-tick diagnostic sampling within tolerance.
12. Left/right and team inversion preserve reward symmetry.
