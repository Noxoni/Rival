# Milestone 06 Reward and Curriculum

## Principle

Reward **winning, possession quality, efficient movement, and recoverability**. Mechanics are means, not the objective.

Milestone 05's `RivalRewardV1` is the starting point. Build `RivalRewardV2` only where the new terms are independently logged and bounded.

## 1. Outcome remains dominant

Keep signed goal/concede outcome at the top of the hierarchy. Do not reduce the importance of scoring/defending in order to make mechanic counters rise.

Maintain separate logged components so a checkpoint cannot look healthy merely because shaping reward increased.

Required top-level components:

- outcome;
- possession/useful touch;
- offensive progress;
- boost efficiency;
- recovery;
- mechanics/resource shaping (new, low weight, separately decomposed).

## 2. Boost and aerial efficiency

The target is not `reward flip`. The target is: **produce useful aerial movement/touches while spending less boost and retaining recovery resources.**

Improve the existing boost-efficiency accounting so metrics can distinguish:

- boost spent while airborne;
- useful ball progress/touch per boost spent;
- aerial distance or closing progress per boost spent;
- boost retained at the end of an aerial commitment;
- whether an airborne dodge/flip was available and whether it was used;
- recovery quality after the aerial.

Use these primarily as metrics and modest shaping. Avoid a strong raw `do not boost` penalty that teaches passive flight.

## 3. Recovery value

Extend recovery tracking beyond the landing instant.

After lost possession, failed offense, or an aerial commitment, measure/reward small improvements in:

- wheels-down/surface-stable landing;
- useful speed retained or regained;
- velocity/orientation toward a defensively relevant position;
- boost retained;
- time until the car is goal-side or otherwise able to defend;
- avoiding a concession during the immediate recovery window.

Do not reward driving toward own net when that is tactically unnecessary; recovery shaping must remain small beneath game outcome.

## 4. Low-weight mechanics aids

RLGym Tools exposes useful primitives such as flip-reset, wavedash and aerial-distance rewards. They may be used as **curriculum aids**, not primary objectives.

A single mechanics event should be small relative to a goal. Calibrate/clamp event reward scales from measured random/natural rollouts rather than assuming package defaults are on the right scale.

Allowed low-weight aids:

- flip/reset acquisition;
- wavedash event;
- bounded aerial-distance/usefulness;
- generic airborne dodge-resource acquisition (including ceiling/ball reset sources if the state exposes it reliably).

Prefer a generic `airborne dodge resource regained` signal over separately rewarding `ceiling reset` by name.

If feasible, distinguish **productive follow-up** from mere acquisition: a reset followed by maintained control, useful touch, shot pressure or opponent outplay is more valuable than collecting a reset and immediately losing possession.

Do not directly reward:

- musty flick;
- breezi flick;
- Meeri pop;
- zap dash;
- wall dash;
- sidewall skim.

If those sequences create speed, control, boost savings, recovery or goals, the outcome/recovery system should make them valuable naturally.

## 5. Reward contribution audit

Before the 100M campaign, run a bounded natural/random stress sample and record for every reward component:

- count;
- mean;
- absolute mean;
- min/max;
- cumulative signed value;
- cumulative absolute value;
- share of total absolute shaping.

Reject obviously dominant mechanic/recovery shaping before long training.

During training, log the same contribution ratios at each 5M evaluation boundary. A rise in mechanic reward accompanied by falling goal performance is a warning, not success.

## 6. Broad curriculum distribution

Natural 1v1 is always the majority.

Minority reset distributions should create **families of states**, not drills:

### Aerial/wall possession family

Randomize ball and car height, wall distance, orientation, velocity, boost, field side and opponent pressure. Include sidewall/backboard/ceiling-adjacent states without requiring a specific named mechanic.

### Recovery family

Randomize airborne/wall/awkward orientations and speeds after plausible possession-loss or failed-offense states. The task is to become useful again quickly.

### Low-resource aerial family

Randomize elevated-ball situations with a wide boost distribution, including low boost, so the policy has incentive to discover momentum-preserving flips and selective aerial commitment rather than relying entirely on boost.

These families together should remain a minority of resets. They exist to expose opportunities that natural kickoff-only episodes may reach too rarely, not to define the bot's behavior.

## 7. Mechanics metrics (not promotion by themselves)

Track naturally occurring rates and outcomes for:

- appended-action usage overall and by action family;
- airborne flip/dodge use;
- reset/resource acquisition;
- productive reset follow-up;
- wavedash-like events;
- wall contact + speed recovery events;
- aerial boost spent per useful aerial distance/touch;
- boost remaining after aerial attack;
- recovery time after lost possession / missed offense;
- concessions during recovery windows.

Mechanics counts are diagnostic. Promotion still depends on 1v1 results.
