# Rival v10.2 — 10-Hour Overnight Execution Budget

## Purpose

The progressive Stage-1-through-4 package is intended to run unattended for approximately the next **10 real wall-clock hours**.

This file adds a wall-clock envelope on top of the stage-specific learner-simulated-hour and active-step ceilings. Capability gates remain authoritative: wall time never authorizes skipping a prerequisite or extending a failed lesson by silently changing its reward.

## Global wall-clock authority

From the moment the progressive campaign controller begins real Stage-1 work after implementation/preflight:

```text
total_wall_clock_authority = 10 hours
```

This ten-hour envelope includes:

- real PPO collection/update time;
- deterministic evaluations;
- generalization evaluations on apparent passes;
- checkpointing;
- stage-transition preflights;
- repository commits/pushes;
- final closeout/reporting.

Reserve the final **20 minutes** of the 10-hour envelope for safe finalization.

Therefore the controller must not start another ordinary PPO iteration when:

```text
remaining_wall_time <= 20 minutes + projected_safe_iteration_and_boundary_overhead
```

Use a rolling measured duration from completed PPO iterations and recent evaluations rather than assuming a fixed duration.

If the wall envelope expires mid-stage, finish the current atomic PPO update if already in progress, write/reload a clean recovery checkpoint, run only the bounded evaluation/reporting needed for an honest closeout if time permits, push the stable state, and stop with:

`stop_progressive_overnight_wall_clock_budget_exhausted`

Do not start a new stage when less than **30 minutes** remain before the finalization reserve.

## Planning split

The stage simulated-hour ceilings remain:

| Stage | Skill | Simulated learner-hour ceiling | Active learner-step ceiling | Approximate overnight planning share |
|---|---|---:|---:|---:|
| 1 | Ball acquisition | 15 h | 6,480,000 | ~1.5 real h |
| 2 | Ground control / dribbling | 20 h | 8,640,000 | ~2.0 real h |
| 3 | Aerial acquisition / air control | 30 h | 12,960,000 | ~3.0 real h |
| 4 | Finishing / scoring | 25 h | 10,800,000 | ~2.5–3.0 real h |

These approximate shares are **planning reservations, not independent hard wall timers**.

The total Stage-1-through-4 experience ceiling remains:

```text
90 learner-simulated hours
38,880,000 active-learner 120-Hz steps
```

At roughly the one-active-learner throughput implied by the completed v10.1 56-worker campaign, that experience ceiling is of the same order as one overnight 10-hour wall-clock run once evaluation/transition overhead is included. Measure the actual one-learner throughput in preflight and report the observed conversion.

## Carry-forward rule

If a stage masters early, all unused wall time carries forward automatically to later authorized stages.

Example:

```text
Stage 1 planned ~1.5h, passes in 0.8h
unused ~0.7h -> available to Stages 2–4
```

Do not keep training a mastered stage merely to spend its nominal share.

## Prerequisite-over-allocation rule

If a stage reaches its approximate planning share but has **not** passed, do not skip it just to preserve time for later stages.

The current stage may continue consuming the remaining global wall envelope while it remains inside:

- its own simulated-hour/step ceiling;
- its documented no-learning/exploit stop gates;
- the global 10-hour wall-clock ceiling.

This means later stages can receive less or no time if an earlier prerequisite proves difficult. That is intentional. A later lesson is useless if the prerequisite has not been learned.

## Stage failure rule

If any stage reaches one of its documented failure/stop decisions before the 10-hour wall envelope expires:

- stop the entire progressive campaign;
- do not spend the remaining wall time brute-forcing or redesigning the reward;
- preserve the best/recoverable checkpoint and all previously passed stage checkpoints;
- write/push the failure evidence;
- leave production unchanged.

Unused overnight time is deliberately left unused in this case.

## Stage success rule

If Stage 4 reaches:

`finishing_skill_passed_unlock_opponent_pressure`

before the 10-hour wall envelope expires:

- stop immediately;
- do not begin Stage 5;
- use remaining wall time only for verification, evidence cleanup, and final push.

## Required wall-time telemetry

At every training/evaluation boundary record:

- campaign wall-clock elapsed;
- wall-clock remaining to 10-hour limit;
- current stage wall-clock elapsed;
- PPO collection/update wall time;
- deterministic evaluation wall time;
- generalization evaluation wall time when run;
- checkpoint/report/push wall time if measurable;
- trainable active-learner steps/sec;
- learner-simulated hours per real wall hour;
- projected time to the next scheduled boundary;
- whether the controller is still allowed to start another PPO iteration.

The final umbrella report must reconcile:

- total real wall time;
- total learner-simulated hours;
- total active-learner steps;
- time spent in each stage;
- time spent in evaluation/preflight/finalization;
- unused authorized wall time, if any.

## Governing priority

When constraints conflict, use this priority order:

1. **prerequisite capability gate**;
2. stage failure/no-learning/exploit stop rule;
3. global 10-hour wall-clock safety envelope;
4. stage simulated-hour/step ceiling;
5. nominal per-stage planning share.

The nominal split exists to make an overnight run plausible. It must never override evidence.