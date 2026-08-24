# Rival v10.2 — Stage 1 Ball-Acquisition Evaluation and Exit Gates

## Scope

This file governs **Stage 1 only** inside the progressive Stage-1-through-4 package.

The primary unit of competence is an episode-level acquisition attempt:

> From this reachable reset, did the active learner physically touch the ball before the no-touch timeout?

Do not use PPO reward, loss, entropy, throughput, or aggregate goals as capability evidence.

Do not make `touches / 100k agent-steps` the primary gate. It remains useful telemetry, but episode-level acquisition success and time-to-first-touch are the Stage-1 capability measures.

## Deterministic evaluation suite

At every boundary, freeze the current actor and evaluate deterministically with the same seeded state set.

Minimum evaluation corpus per boundary:

- 100 `stationary_close` episodes;
- 100 `stationary_medium` episodes;
- 100 `moving_chase` episodes;
- 100 `awkward_heading` episodes;
- 100 `natural_kickoff_holdout` episodes.

500 episodes total.

Balance:

- active team 50/50;
- left/right mirror geometry;
- seed set frozen before the first Stage-1 PPO update;
- exact same state corpus reused at every boundary and on the v10.1 source actor.

Separately maintain a second unseen generalization corpus of at least 250 episodes with disjoint seeds. Run it only when a checkpoint appears to pass the frozen corpus.

## Metrics

Per family and overall, record:

- first-touch success count/share;
- no-touch timeout count/share;
- mean, median, p90, and p95 time-to-first-touch among successful episodes;
- initial car-ball distance;
- terminal car-ball distance for failures;
- mean signed car-caused progress;
- physical touch count;
- touches per episode;
- touches per 100k active-learner steps;
- dense reward and touch reward separately;
- percentage of episodes saturating the distance-shaping budget;
- goals as reward-neutral diagnostics;
- deterministic throttle/steer/jump/dodge/boost use as diagnostics only.

## Source baseline

Before Stage-1 training, evaluate the exact v10.1 +10 actor on both:

- the frozen 500-episode gate corpus;
- the disjoint generalization corpus.

Publish those values as `source_v10_1_plus10`.

Do not infer thresholds from M09/M10 protocols.

## Phase A readiness

A boundary passes Phase A only if all of the following hold on the frozen gate corpus:

| Family | First-touch success requirement |
|---|---:|
| stationary_close | >= 95% |
| stationary_medium | >= 85% |
| moving_chase | >= 75% |
| awkward_heading | >= 80% |
| natural_kickoff_holdout | diagnostic only in Phase A |

And globally across the four acquisition families:

- aggregate first-touch success >=85%;
- aggregate no-touch timeout share <=15%;
- median successful time-to-first-touch <=5.0 s;
- mean terminal distance on failed episodes is at least 25% lower than mean initial distance on those same failures;
- no family regresses by more than 10 percentage points versus the previous boundary once it has exceeded 70% success.

A single Phase-A pass unlocks Phase B only.

## Phase B final acquisition gate

A checkpoint is acquisition-ready only when it passes the following at **two consecutive evaluation boundaries**.

### Frozen gate corpus

- `stationary_close` >=97%;
- `stationary_medium` >=92%;
- `moving_chase` >=88%;
- `awkward_heading` >=90%;
- `natural_kickoff_holdout` >=80%;
- aggregate across all families >=90%;
- overall no-touch timeout share <=10%;
- median successful time-to-first-touch <=4.0 s.

### Disjoint generalization corpus

At each apparent pass require:

- aggregate first-touch success >=85%;
- no acquisition family below 75%;
- no-touch timeout share <=15%.

### Reward-integrity checks

Also require:

- physical touch reward corresponds one-for-one with validated new-contact events;
- speed reward remains exactly absent;
- goal/concede reward remains exactly zero;
- distance shaping contributes less cumulative positive return than touch events on successful episodes once repeated touches occur;
- no evidence that a stationary learner receives positive distance reward simply because the ball approaches.

## Stage-1 success transition

When acquisition readiness passes twice consecutively, emit exactly:

`ball_acquisition_skill_passed_unlock_ground_control`

Then:

1. preserve the exact passing Stage-1 actor checkpoint immutably;
2. write/push the Stage-1 boundary evidence;
3. independently reload and reproduce held outputs;
4. **do not promote to production**;
5. if the progressive 10-hour wall-clock authority still permits another stage, transition to Stage 2 under `STAGE_2_GROUND_CONTROL.md` and `PROGRESSIVE_STAGE_PROTOCOL.md`;
6. transfer actor weights only and create a fresh critic plus fresh actor/critic optimizers.

Do **not** continue spending Stage-1 budget after mastery.

## Failure / stop rules

Evaluation boundaries are at added Stage-1 learner-simulated hours:

`+1, +2.5, +5, +7.5, +10, +12.5, +15`

Hard Stage-1 maximum: **15 learner-simulated hours**.

### No-learning stop at +5h

By +5h, if:

- aggregate first-touch success across the four acquisition families improved by <10 absolute percentage points over the source actor **and**
- no-touch timeout share improved by <10 absolute percentage points,

stop with:

`stop_ball_acquisition_no_material_learning_by_plus_5h`

This stops the entire progressive campaign. Do not advance to Stage 2.

### Exploit/regression stop

At any boundary after +2.5h, stop if deterministic speed/activity rises materially while:

- acquisition success falls for two consecutive boundaries, or
- mean terminal distance on failures worsens for two consecutive boundaries,

and reward-integrity inspection shows dense shaping is being exploited without contact improvement.

Stop the entire progressive campaign and preserve evidence.

### Stage budget stop

If +15h has not passed acquisition readiness:

`stop_ball_acquisition_not_mastered_by_plus_15h`

Do not advance.

### Global overnight wall-clock stop

`OVERNIGHT_10H_BUDGET.md` also applies. If the global wall envelope is exhausted before Stage-1 mastery, preserve the current clean checkpoint/evidence and emit:

`stop_progressive_overnight_wall_clock_budget_exhausted`

## Deliberately not Stage-1 gates

Do not require:

- goals;
- saves;
- dribble duration;
- aerial touches;
- Wisp/Nexto win rate;
- self-play performance;
- named mechanics.

Those are later lessons.