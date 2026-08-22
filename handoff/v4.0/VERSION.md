# Rival Codex Handoff v4.0

**Handoff version:** v4.0  
**Created:** 2026-08-22  
**Required starting commit:** `e4cc175a4259202d5cc7ee437abef224b731354f`  
**Frozen Wisp-equivalent baseline:** `4f2b21c00e2fcb7108ab1006fd950b066fbd0484`

## Version boundary

Milestone 03 is complete and rejected. Preserve it exactly as historical evidence. Do not rewrite or squash `f025344b5ab325c3b6bfb082770b95a81e6f9809` or `e4cc175a4259202d5cc7ee437abef224b731354f`.

Rival's normal gameplay remains challenge-calibration `off` until a later treatment passes a deterministic paired-exposure gate.

## v4.0 purpose

v4.0 is a **deterministic paired-exposure milestone**, not another tactical-tuning milestone.

The immediate problem is that Milestone 03 baseline and treatment trajectories diverged even when the treatment applied zero interventions. v4 must eliminate that confound before any challenge-calibration parameter is tuned again.

The primary suspected sources are:

- Wisp `CustomObs.build_obs()` uses process-global `random.shuffle()` for teammate/opponent observation slots on every decision;
- the observation includes `player.prev_action`;
- Rival keeps `prev_actions`, old/new actions, tick-window state, policy tick, RocketSim adapter state, challenge-tracker state, and related runtime history across controlled state-setting cases unless a timer rewind/reset path occurs.

## Next version

A later gameplay-treatment version may resume challenge calibration only after v4 demonstrates reproducible off-vs-observe exposure from synchronized controlled cases. A failed reproducibility milestone is acceptable and must not be hidden by tuning the treatment.