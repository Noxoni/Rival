# Rival Milestone 02 — Evidence Event Definitions

These definitions are for **offline evidence detection and ranking**, not gameplay control.

A heuristic firing creates a **candidate event**, not a confirmed defect. Preserve raw measurements and make thresholds configurable so later analysis can change without recollecting matches.

## Common event envelope

Every extracted event should contain at least:

```text
event_id
class
session_id
opponent
source = natural_match | controlled_probe
start_game_time
end_game_time
anchor_decision_tick
pre_window_record_range
post_window_record_range
raw_features
derived_features
outcome
ranking_score
ranking_explanation
replay_path/replay_timestamp if available
fixture_path if curated
```

Default context window suggestion:

- 1.5–2.0 seconds before anchor;
- 2.5–5.0 seconds after anchor depending on event class.

Use decision ticks/game time rather than wall-clock time for gameplay analysis.

---

# 1. `resource_stressed_aerial`

## Question

Does Rival begin or continue an aerial/high-cost airborne play when the available resources and geometry make the play low-value or unrecoverable?

## Candidate anchor

Anchor on a transition into an aerial-like sequence, or a meaningful recommitment while already airborne.

Possible observable signals:

- selected action begins jump;
- selected action uses boost while airborne;
- non-trivial pitch/yaw/roll while airborne with ball elevated;
- repeated aerial-like actions across consecutive policy decisions;
- ball is sufficiently high/distant that the sequence plausibly represents an aerial rather than a small recovery hop.

Do **not** use one hard-coded boost threshold as the definition.

## Required measurements

At/around anchor:

- self boost;
- boost delta over play window;
- boost pickups during window;
- car position/velocity/orientation/angular velocity;
- ball position/velocity/height;
- self ETA and opponent ETA;
- self/ball distance;
- score/clock;
- own-goal/opponent-goal side context if derivable;
- selected action and top policy alternatives;
- whether the opponent is pressuring;
- touch sequence;
- landing/recovery time;
- possession proxy after the play;
- shot/goal/counterattack outcome where available.

## Useful derived features

Examples, not mandatory formulas:

- `start_boost`
- `boost_spent`
- `boost_gained`
- `net_boost_delta`
- `airborne_duration`
- `time_to_first_touch`
- `touch_obtained`
- `self_touch_count`
- `opponent_touch_within_2s_after_play`
- `possession_before`
- `possession_after`
- `recovery_time_to_ground/control`
- `distance_to_ball_at_commit`
- `ball_height_at_commit`
- `opponent_eta_margin`

If an approximate resource-cost model is added, record its components separately rather than only the final score.

## Candidate ranking

A high-ranked candidate may combine several of:

- low starting boost relative to observed play demand;
- substantial distance/height;
- no reachable boost pickup;
- failed/no touch;
- opponent wins next meaningful touch;
- long recovery;
- possession flips;
- immediate dangerous counterattack/goal.

A low-boost aerial that makes a critical save or scores is not automatically a defect.

## Controlled-probe ground truth

The resource-stressed aerial probe should vary boost and geometry while holding other factors as stable as practical. Use the probe parameter set as part of the event record so the same state can be rerun later.

---

# 2. `boost_detour_possession_loss`

## Question

Does Rival leave a useful possession/pressure path to collect boost, then lose possession or surrender disproportionate pressure?

This event exists to distinguish **good boost denial** from **boost greed**.

## Candidate anchor

Potential anchors include:

- trajectory rotates materially toward a boost pad while Rival has a favorable possession/ETA state;
- a boost pickup occurs after a period of favorable possession/pressure;
- path length to ball/opponent goal increases substantially while a pad is approached;
- Rival abandons ball proximity shortly before a boost pickup.

Prefer sequence-based detection over a single-frame distance threshold.

## Required measurements

- self boost before/after;
- pad identity, size, location, active/timer state if available;
- actual pickup inferred from pad state + boost delta and/or RLBot accolade/input data;
- ball/self/opponent positions and velocities;
- self/opponent ETA;
- latest-touch ownership/timestamps;
- possession proxy before and after;
- ball field progress before/after;
- opponent boost when available;
- score/clock;
- next meaningful ball touch;
- shot/goal/counterattack outcome.

## Useful derived features

- `boost_gained`
- `route_deviation_angle`
- `distance_added_to_ball`
- `eta_advantage_before`
- `eta_advantage_after`
- `time_from_detour_to_pickup`
- `time_from_pickup_to_opponent_touch`
- `possession_flip`
- `opponent_boost_denied`
- `pressure_loss`
- `ball_progress_loss`

## Candidate ranking

Rank higher when:

- Rival had clear possession/pressure;
- boost gain was strategically modest relative to route cost;
- opponent did not meaningfully need/contest that boost;
- possession flips soon afterward;
- ball retreats or opponent gains a dangerous touch.

Do not penalize a boost detour that preserves possession, starves the opponent, or is required for defense/recovery.

Causality must remain qualified: the analyzer identifies correlation windows for review/testing.

---

# 3. `apparent_vs_actual_challenge`

## Question

Can Rival distinguish an opponent that *appears* to challenge from one actually committed to a ball-intersecting challenge?

This is the primary measurement for avoiding the known Nexto-style fake-challenge weakness.

## Two evidence modes

### Natural match

Commitment is inferred from RLBot v5 state/input data.

### Controlled probe

The probe behavior is known and should be recorded as ground truth:

- `true_commit`
- `boost_then_brake`
- `boost_then_veer`
- `jump_fake`
- `delayed_challenge`
- additional variants if useful.

Controlled-probe labels are stronger evidence than natural-match inference.

## Required measurements

For both cars:

- position/velocity;
- forward/up/orientation;
- angular velocity;
- boost;
- `last_input` including throttle/steer/jump/boost/handbrake;
- jump/dodge/air state;
- latest touch and touch game time;
- distance to ball;
- ETA to ball;
- velocity-to-ball alignment;
- forward-to-ball alignment;
- closing speed;
- projected/intersection geometry where practical.

For Rival policy:

- selected action;
- top-N alternatives;
- confidence/margin;
- whether it flicks/jumps/throws a hard touch/releases possession;
- touch sequence after apparent pressure.

## Derived commitment features

Useful features may include:

- `challenger_closing_speed`
- `challenger_velocity_ball_alignment`
- `challenger_forward_ball_alignment`
- `challenger_eta_to_ball`
- `challenger_boost_input`
- `challenger_jump_input`
- `predicted_ball_intersection_error`
- `commitment_score`

Keep each component in the event record. Do not retain only an opaque commitment score.

## Rival response features

- `rival_jump_or_flick_after_pressure`
- `rival_hard_touch_after_pressure`
- `rival_ball_separation_after_response`
- `rival_retains_next_touch`
- `opponent_gets_next_touch`
- `possession_retained_after_1s/2s`
- `shot_created`
- `goal_conceded/goal_scored`

## Candidate ranking

For controlled fake probes, rank highest when:

1. ground truth says opponent did not commit;
2. Rival responds as though forced to release possession;
3. ball separation increases sharply or opponent wins the next useful touch;
4. Rival policy confidence in the release action is high.

For true-commit probes, also flag the opposite failure: Rival holds possession too long and is cleanly challenged when a release/evasion was needed.

The objective is not "never flick on pressure." The objective is **calibrated response to actual commitment**.

---

# Outcome attribution

Where RLBot v5 exposes direct player inputs, latest touches, accolades, score events, and match state, prefer those fields over geometry-only guesses.

At minimum, analyzers should distinguish:

- self gets next ball touch;
- opponent gets next ball touch;
- no touch in window;
- shot/save/goal if observable;
- possession proxy improves/degrades;
- recovery/ground-control restored;
- boost gained/spent;
- match state reset due to goal/kickoff.

Do not carry an event outcome across a goal reset as though normal play continued.

# Fixture selection

Curate fixtures based on usefulness, not only highest ranking score.

A useful initial fixture set should include:

- one obvious/high-confidence example;
- one ambiguous/borderline example;
- one counterexample where similar surface conditions led to a good outcome.

This prevents Milestone 03 from overfitting to only spectacular failures.
