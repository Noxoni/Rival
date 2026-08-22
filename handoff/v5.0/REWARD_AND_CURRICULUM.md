# Rival Reward and Curriculum Design

## Principle

Rival should not be rewarded for doing a mechanic because the mechanic looks advanced. It should be rewarded because the mechanic improves the game state.

The hierarchy is:

1. win / score / do not concede;
2. preserve or improve possession and scoring probability;
3. use boost efficiently;
4. recover quickly and remain defensively relevant after failed offense or possession loss;
5. develop mechanics that help achieve 1-4.

Mechanic-specific rewards are small curriculum aids, not the objective.

---

## Reward v1

Implement reward components separately and log every component so reward farming can be seen.

### A. Outcome reward — dominant

Use `GoalReward` or an equivalent signed goal/concede reward as the dominant component.

A goal for Rival must outweigh accumulating a handful of style rewards. A concession must meaningfully punish reckless offense.

Do not optimize only instantaneous ball velocity or aerial height.

### B. Possession / useful touch value

Add a modest stateful reward for touches that improve control rather than simply touching the ball.

Useful signals may include:

- maintaining/recovering likely next-touch advantage;
- keeping the ball within a controllable region after contact;
- advancing toward the opponent goal without immediately handing the next touch away;
- beating the opponent while remaining recoverable;
- avoiding touches that convert controlled possession into a low-probability throwaway.

This reward should specifically make behavior like 'carry ball to ceiling and blindly hit it toward net, giving away possession' less attractive without hard-coding 'never take the ball to the ceiling'. If a ceiling play beats the defender or creates a real shot, it should still be valuable.

### C. Ball/scoring progress

Use a small `GoalProbReward`, `BallTravelReward`, advanced touch reward, or equivalent shaping signal to make useful offensive progress easier to learn.

Keep its weight below actual scoring outcome.

### D. Boost efficiency

Use small `BoostChangeReward` / `BoostKeepReward` style signals plus custom event accounting where needed.

The desired behavior is not 'always save boost'. It is:

- spend boost when it materially improves the play;
- avoid burning the entire tank for an aerial that could use a flip or a more efficient trajectory;
- retain enough boost to recover after a failed attack when possible.

Track at minimum:

- boost consumed per simulated second;
- boost consumed during airborne windows;
- useful ball touches per boost spent;
- boost remaining at possession loss / landing / defensive recovery.

### E. Recovery efficiency

Create a small stateful `RecoveryValueReward` or equivalent.

Trigger meaningful evaluation after states such as:

- Rival loses possession;
- Rival makes an attacking touch/shot and no longer has immediate control;
- Rival is airborne/falling with the opponent gaining the ball;
- Rival lands from an aerial play.

Reward improvement in defensive relevance, for example:

- landing sooner in a useful orientation;
- regaining useful ground/wall speed quickly;
- moving goal-side / toward the developing play;
- gaining speed with low boost expenditure;
- retaining boost;
- avoiding a concession during the recovery window.

This is the main route for learning wall dashes, wavedashes, zap-dash-like speed recovery, fast landings, and related movement. Do not hard-code those mechanics by name.

### F. Aerial efficiency

Add a small event/state reward around aerial usefulness, not mere airtime.

Useful factors:

- useful ball contact / scoring progress while airborne;
- distance or momentum gained per boost spent;
- retaining flip/reset options;
- landing/recovery quality afterward;
- avoiding low-boost aerial commitments that lead to a concession.

`AerialDistanceReward` may be part of this, but airtime/distance alone must not dominate.

### G. Reset / dash curriculum rewards — small and annealable

`rlgym-tools` currently exposes `FlipResetReward` and `WaveDashReward`. They may be used as small curriculum bonuses.

A reset acquisition should ideally receive additional value only when it becomes useful soon afterward: useful touch, outplay, shot, possession continuation, or recovery.

Do not give a large unconditional bonus for obtaining a flip reset; that invites farming.

Do not create permanent large rewards for musties, breezis, Meeri pops, stalls, or named mechanics.

---

## Curriculum

### Stage 0 — Wisp bootstrap

Teacher imitation only. No need to learn fancy mechanics yet. Goal is to begin from competent 1v1 behavior.

### Stage 1 — natural self-play

Primary training distribution is ordinary 1v1 RocketSim self-play.

Use the dominant outcome reward plus low-weight possession/progress/boost/recovery shaping.

Train long enough to establish that the Wisp-derived student remains competent while PPO is functioning.

### Stage 2 — mechanics-capable natural self-play

Enable the expanded action table and 4-tick cadence if not already active.

Continue natural 1v1. Let the richer control space begin producing more efficient aerials, recoveries, flips, and reset attempts.

Do not require named mechanics to appear before continuing.

### Stage 3 — broad curriculum reset mix

Only after normal self-play works, add a **minority** of broad reset distributions to increase rare but important exposure.

Suggested starting mix:

- 70-80% ordinary/natural 1v1 resets/continuations;
- 5-10% randomized wall/ceiling possession;
- 5-10% randomized aerial possession;
- 5-10% randomized awkward recovery / low-boost states;
- optional replay-derived competitive states.

These percentages are starting points, not sacred constants. Measure whether they improve real natural-play outcomes.

Use `WeightedSampleMutator`, `ReplayMutator`, `RandomPhysicsMutator`, or equivalent tools rather than a library of exact coordinates.

### Stage 4 — opponent diversity

Later, add a mix of:

- current Rival self-play;
- historical Rival checkpoints;
- frozen Wisp;
- potentially other offline benchmark policies;
- replay-derived human states.

This prevents the policy from specializing only to its current clone.

---

## Metrics that matter

Training reports must include more than total reward.

At minimum track:

- goals for/against and goal differential in evaluation;
- episode return and each reward component;
- possession/next-touch proxy statistics;
- touches and useful-touch outcomes;
- boost consumption and boost at recovery transitions;
- airborne duration and aerial touches;
- flip/reset acquisitions and productive follow-ups;
- wavedash/reset curriculum event counts if enabled;
- average time to useful recovery after possession loss / failed attack;
- action frequency distribution, especially appended actions;
- PPO entropy / loss / KL metrics;
- environment steps/sec and simulated seconds/sec.

A reward component that rises while natural-match performance degrades is suspicious and should not be celebrated.

---

## Promotion rule

A checkpoint is not promoted because training reward increased.

Later promotion into the live Rival RLBot bot requires natural evaluation against the fixed benchmark suite (at minimum frozen Wisp and Nexto), with multiple full games and aggregate metrics.

Mechanics are evidence of capability. Winning, possession quality, resource efficiency, and recovery determine whether the capability is useful.

---

## Existing tools reviewed

`rlgym-tools` v2.x includes useful pieces such as:

- `AdvancedTouchReward`
- `AerialDistanceReward`
- `BallTravelReward`
- `BoostChangeReward`
- `BoostKeepReward`
- `FlipResetReward`
- `GoalProbReward`
- `VelocityPlayerToBallReward`
- `WaveDashReward`
- `ReplayMutator`
- `RandomPhysicsMutator`
- `WeightedSampleMutator`
- replay parsing / replay-frame conversion
- training-pack state support

Use these where they match the intended outcome; do not force every available tool into Reward v1.
