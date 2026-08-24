# Rival prerequisite skill progression

This document freezes the training order for the next scratch-policy development stages. The ordering is intentionally prerequisite-driven: a later skill is not trained until the earlier motor/control skill is demonstrably reliable.

## Stage 0 — Locomotion / agency

**Status:** learned sufficiently to stop rewarding directly.

Capability learned in v10.1:

- produce sustained useful controller output;
- accelerate and travel around the arena;
- use throttle/steer/jump/dodge controls rather than remaining mostly inert.

The v10.1 actor is retained as the starting actor for Stage 1, but speed is no longer rewarded. A primitive that has been learned should become a means to the next objective rather than a permanent source of reward.

## Stage 1 — Ball acquisition

**Current stage: v10.2.**

Question:

> Can Rival reliably locate, approach, and physically touch the ball from broad reachable ground states?

Reward only:

- signed car-caused reduction in car-to-ball distance;
- every genuine new physical ball touch.

No scoring reward exists.

Exit only when first-touch acquisition is reliable across close, medium, moving/intercept, and awkward-heading families.

## Stage 2 — Ground ball control / dribbling

Locked until Stage 1 passes.

Question:

> After reaching the ball, can Rival keep it close and deliberately produce additional controlled contacts instead of merely colliding once and losing it?

Expected future learning signals:

- controlled ball proximity after first touch;
- repeated separated self touches;
- ball velocity compatible with continued possession;
- carrying/pushing the ball while retaining reachability;
- eventually ball-on-car / dribble-like control.

At this stage the generic first-touch acquisition reward should be reduced substantially or removed once acquisition remains retained.

Still do **not** make scoring the primary target.

## Stage 3 — Aerial acquisition and air-dribble control

Locked until ground control is reliable.

Question:

> Can Rival leave the ground intentionally, acquire an elevated ball, and sustain useful repeated aerial contacts?

Progression inside the stage should move from:

1. reachable single aerial contact;
2. aerial follow-up contact;
3. multiple controlled aerial touches;
4. sustained air-dribble-like possession;
5. wall-to-air and ceiling-origin variations after basic open-air control is reliable.

The objective is aerial ball control, not named mechanics and not scoring yet.

## Stage 4 — Finishing with learned control skills

Locked until ground and aerial possession/control prerequisites are established.

This is the **first stage where scoring becomes a deliberate positive training objective**.

Question:

> Can Rival use the ball-control skills it already owns to put the ball into the opponent goal?

Train broad finishing from both learned modes:

- ground dribble / carry / push into shots;
- aerial / air-dribble finishing;
- simple direct strikes may also emerge, but the curriculum should not erase possession skills simply because a faster shot is available.

Goal reward becomes the terminal maximum outcome signal here.

## Stage 5 — Defensive response and opponent pressure

Locked until Rival can independently acquire, control, and finish the ball.

Introduce active opponents progressively rather than immediately demanding full self-play competence.

Skills:

- react to an opponent reaching/challenging the ball;
- preserve possession under pressure;
- recover after losing possession;
- intercept/defend threatening balls;
- distinguish when to attack versus defend;
- learn saves, challenges, shadowing, and useful clears through outcome pressure.

Use controlled opponent difficulty first, then stronger fixed bots.

## Stage 6 — Full self-play / opponent league

Only after Rival can actually play Rocket League in isolation and under basic pressure should self-play become the primary tactical curriculum.

Opponent pool can then include:

- current Rival;
- historical Rival snapshots;
- Wisp;
- Nexto;
- later specialized exploiters or additional strong fixed bots.

At this point most primitive shaping should be heavily reduced or absent. Outcome, possession quality, resource efficiency, and tactical performance can dominate.

## Stage 7 — Advanced mechanics and refinement

Advanced mechanics are not prerequisite substitutes. They are learned after the policy can already play coherent Rocket League.

Target capabilities may include:

- flip and ceiling resets;
- wavedash, wall dash, zap-dash-like recoveries;
- stalls;
- sidewall skim/recovery;
- Meeri-pop-like transitions;
- musty/Breezi-like control sequences;
- momentum-preserving aerial flips;
- highly efficient recoveries and boost use.

Prefer outcome/usefulness-driven emergence and broad state distributions. Named mechanics are diagnostic capability labels, not macro actions.

## Governing principle

Each stage follows this pattern:

1. identify the next missing prerequisite;
2. reward that prerequisite simply and directly;
3. measure it deterministically;
4. remove/reduce its direct reward once retained;
5. unlock exactly the next dependency.

The policy should not be asked to optimize scoring before it can reliably acquire and control the ball, and it should not be asked to solve tactical self-play before it can independently execute the motor skills required for meaningful play.
