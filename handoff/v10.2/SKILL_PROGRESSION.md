# Rival prerequisite skill progression

This document freezes the prerequisite order for scratch-policy training.

The governing rule is simple:

> **A later skill is not trained until the earlier skill is demonstrably reliable. Once a prerequisite is learned, its direct reward is removed or reduced so it becomes a tool for the next lesson instead of a permanent reward exploit.**

## Stage 0 — Locomotion / agency

**Status:** learned sufficiently in v10.1 to stop rewarding directly.

Rival demonstrated sustained controller use and much higher arena speed. That did not produce ball interaction, but it proved the actor can learn a simple motor primitive from reinforcement.

Carry the v10.1 actor forward. Do not keep a speed reward.

## Stage 1 — Ball acquisition

**Authorized/current first stage of v10.2.**

Question:

> Can Rival reliably locate, approach, and physically touch the ball from broad reachable states?

Reward only:

- small signed car-caused reduction in car-to-ball distance;
- every genuine new physical learner touch.

No scoring reward.

Exit decision:

`ball_acquisition_skill_passed_unlock_ground_control`

Only then may Stage 2 start.

## Stage 2 — Ground ball control / dribbling

**Authorized after Stage 1 passes.**

Question:

> After reaching the ball, can Rival keep it reachable and deliberately produce repeated controlled ground contacts instead of merely colliding once and losing it?

Progression:

1. second touch after acquisition;
2. third/fourth touch chain;
3. moving push/control while keeping the ball nearby;
4. turning back into control;
5. broad carry/dribble-like control with low relative car-ball velocity.

The first touch is now only a small bridge. Follow-up physical touches are the high-value event. A small bounded proximity/control signal may bridge between contacts.

Still no positive scoring objective.

Exit decision:

`ground_control_skill_passed_unlock_aerial_control`

Only then may Stage 3 start.

## Stage 3 — Aerial acquisition and air-dribble control

**Authorized after Stage 2 passes.**

Question:

> Can Rival intentionally leave the ground, acquire an elevated ball, and sustain useful repeated aerial contacts?

Progression:

1. reachable low aerial contact;
2. medium/moving aerial intercept;
3. aerial follow-up contact;
4. repeated controlled aerial touches;
5. sustained air-dribble-like possession;
6. lateral and wall-to-air variations after basic open-air control becomes reliable.

Do not reward jump/boost button presses directly. Reward the physical result: reaching and touching the airborne ball, then retaining aerial control.

Still no positive scoring objective.

Exit decision:

`aerial_control_skill_passed_unlock_finishing`

Only then may Stage 4 start.

## Stage 4 — Finishing with learned control skills

**Authorized after Stage 3 passes.**

This is the **first stage where scoring becomes a deliberate positive training objective**.

Question:

> Can Rival use the ground and aerial ball-control skills it already owns to deliberately put the ball into the opponent goal?

Train broad finishing from:

- ground possession/dribble states;
- awkward/off-angle ground possession;
- aerial/air-dribble possession;
- wall-to-air possession;
- free-play-style chained acquisition/control/finish states.

A correct goal becomes the terminal maximum reward. Own goals are negative. Previous control signals are retained only as small bounded bridges.

Capability evaluation must separately track:

- ordinary goal success;
- ground-control-qualified goals;
- aerial-control-qualified goals;
- retention of Stages 1–3.

Exit decision:

`finishing_skill_passed_unlock_opponent_pressure`

**Stop for human review after this decision.** Stage 5 is not automatically authorized.

## Stage 5 — Defensive response and opponent pressure

**Future/locked. Not authorized by the unattended v10.2 package.**

Only after Rival can independently acquire, control, and finish should an active opponent be introduced.

Expected skills:

- react to challenges;
- preserve possession under pressure;
- recover after possession loss;
- intercept/defend threatening balls;
- saves, clears, challenges, shadowing;
- attack/defend role selection.

Difficulty should progress from controlled opponents to stronger fixed bots.

## Stage 6 — Full self-play / opponent league

**Future/locked.**

Only after Rival can actually play coherent Rocket League in isolation and under basic pressure should self-play become the primary tactical curriculum.

Opponent pool may include:

- current Rival;
- historical Rival snapshots;
- Wisp;
- Nexto;
- later exploiters/specialized bots.

Primitive shaping should be heavily reduced or absent. Outcome, possession quality, resource efficiency, and tactics can dominate.

## Stage 7 — Advanced mechanics and refinement

**Future/locked.**

Advanced mechanics are not substitutes for basic ball control.

Potential later capabilities:

- flip/ceiling resets;
- wavedash, wall dash, zap-dash-like movement;
- stalls;
- sidewall skim/recovery;
- Meeri-pop-like transitions;
- musty/Breezi-like control;
- momentum-preserving aerial flips;
- high-efficiency recoveries/boost use.

Named mechanics are capability labels/diagnostics, not macro buttons.

## Stage-transition rule

Every Stage 1–4 transition follows:

```text
learn current skill
-> evaluate deterministic frozen corpus
-> evaluate unseen generalization on apparent pass
-> require documented consecutive passes
-> preserve exact passing actor checkpoint
-> reset critic + actor/critic optimizer states
-> retain actor weights
-> reduce/remove old skill reward
-> begin next lesson
```

If a stage fails its stop/budget gate:

```text
stop
-> preserve evidence/checkpoint
-> do not retune silently
-> do not skip prerequisite
```

The unattended v10.2 package is authorized only through successful completion of Stage 4.
