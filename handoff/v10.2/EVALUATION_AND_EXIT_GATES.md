# Rival v10.2 — Evaluation and Exit Gates

## Evaluation principle

The primary unit of competence is an **episode-level acquisition attempt**:

> From this reachable reset, did the active learner physically touch the ball before the no-touch timeout?

Do not use PPO reward, loss, entropy, throughput, or aggregate goals as capability evidence.

Do not make `touches / 100k agent-steps` the primary gate. It remains useful telemetry, but episode-level acquisition success and time-to-first-touch are much easier to interpret for this stage.

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
- seed set frozen before the first v10.2 PPO update;
- exact same state corpus reused at every boundary and on the v10.1 source actor.

Separately maintain a second unseen generalization corpus of at least 250 episodes with disjoint seeds. Run it only when a checkpoint appears to pass the frozen corpus, so repeated evaluation does not turn the gate set into an implicit training target.

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
- deterministic action diagnostics: throttle/steer/jump/dodge/boost use, but none are readiness requirements.

## Source baseline

Before training, evaluate the exact v10.1 +10 actor on both:

- the frozen 500-episode gate corpus;
- the disjoint generalization corpus.

Publish those values as `source_v10_1_plus10`.

Do not infer readiness thresholds from M09/M10 metrics gathered under different protocols.

## Phase A readiness

Phase A is the easy-to-broad acquisition distribution.

A boundary passes Phase A only if **all** of the following hold on the frozen gate corpus:

| Family | First-touch success requirement |
|---|---:|
| stationary_close | >= 95% |
| stationary_medium | >= 85% |
| moving_chase | >= 75% |
| awkward_heading | >= 80% |
| natural_kickoff_holdout | diagnostic only in Phase A |

And globally across the four acquisition families:

- aggregate first-touch success >= 85%;
- aggregate no-touch timeout share <= 15%;
- median successful time-to-first-touch <= 5.0 s;
- mean terminal distance on failed episodes is at least 25% lower than mean initial distance on those same failed episodes;
- no family regresses by more than 10 percentage points versus the previous boundary once it has exceeded 70% success.

A single Phase A pass does **not** exit the milestone. It authorizes Phase B generalization for the next interval.

## Phase B final acquisition gate

Phase B uses the harder reset mix from `BALL_ACQUISITION_CURRICULUM.md` with the same reward.

A checkpoint is **acquisition-ready** only when it passes the following at **two consecutive evaluation boundaries**:

### Frozen gate corpus

- `stationary_close` first-touch success >= 97%;
- `stationary_medium` >= 92%;
- `moving_chase` >= 88%;
- `awkward_heading` >= 90%;
- `natural_kickoff_holdout` >= 80%;
- aggregate across all families >= 90%;
- overall no-touch timeout share <= 10%;
- median successful time-to-first-touch <= 4.0 s.

### Disjoint generalization corpus

At each apparent pass, run the unseen corpus and require:

- aggregate first-touch success >= 85%;
- no acquisition family below 75%;
- no-touch timeout share <= 15%.

### Reward-integrity checks

Also require:

- physical touch reward is nonzero and corresponds one-for-one with validated new-contact events;
- speed reward remains exactly absent;
- goal/concede reward remains exactly zero;
- distance shaping contributes less cumulative positive return than touch events on successful evaluation episodes once repeated touches occur, consistent with touch being the semantic maximum;
- no evidence that the learner receives positive distance reward while remaining stationary as the ball approaches.

## Exit decision

When the acquisition-ready gate passes twice consecutively:

`ball_acquisition_skill_passed_unlock_ground_control`

Then:

1. preserve the best passing actor as the Stage-1 skill checkpoint;
2. write a final v10.2 results document;
3. do not continue spending the v10.2 budget;
4. do not promote to production;
5. the next authorized design target is Stage 2: **ground ball control / dribbling**.

## Failure / stop rules

Evaluation boundaries are at added v10.2 simulated game-hours:

`+1, +2.5, +5, +7.5, +10, +12.5, +15`

Hard maximum: **15 added simulated game-hours**.

Stop early for diagnosis if either occurs:

### No-learning stop at +5h

By +5h, if:

- aggregate first-touch success across the four acquisition families has improved by less than 10 absolute percentage points over the source actor **and**
- no-touch timeout share has improved by less than 10 absolute percentage points,

stop with:

`stop_ball_acquisition_no_material_learning_by_plus_5h`

### Exploit/regression stop

At any boundary after +2.5h, stop if deterministic speed/activity rises materially while:

- acquisition success falls for two consecutive boundaries, or
- mean terminal distance on failures worsens for two consecutive boundaries,

and reward-integrity inspection shows dense shaping is being exploited without contact improvement.

### Budget stop

If the final +15h boundary has not passed acquisition readiness:

`stop_ball_acquisition_not_mastered_by_plus_15h`

Do not silently retune reward weights mid-campaign. Any material reward/curriculum change requires a new versioned intervention.

## What is deliberately not a gate

Do not require:

- goals;
- saves;
- dribble duration;
- aerial touches;
- Wisp/Nexto win rate;
- self-play performance;
- named mechanics.

Those skills are not being taught yet.
