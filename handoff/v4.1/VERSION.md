# Rival Codex Handoff v4.1

**Purpose:** replace the unexecuted v4.0 deterministic-scenario direction with a natural-play optimization loop.

**Starting implementation:** Milestone 03 result at `e4cc175a4259202d5cc7ee437abef224b731354f` plus later handoff-only commits.

## Decision

Do not spend Milestone 04 trying to make individual Rocket League situations replay bit-for-bit. Rocket League is inherently trajectory-sensitive, and Rival must ultimately react to the state that actually exists rather than memorize or optimize around hand-authored scenarios.

The primary development loop is now:

`natural accelerated matches -> telemetry -> recurring behavior patterns -> state-conditioned policy adjustment -> natural accelerated matches -> aggregate comparison`

Controlled fixtures and probes may remain as lightweight smoke/regression tools, but they are not the main training set, acceptance gate, or source of gameplay targets.

## Supersedes

`handoff/v4.0/` remains historical project documentation but is **superseded before execution** by v4.1. Do not implement the deterministic-pairing milestone merely because it exists in history.

## Gameplay baseline

The rejected Milestone 03 challenge calibration remains disabled by default. Do not resurrect its rejected parameter sets as the next treatment.

## Speed policy

Use approximately **5x effective simulation speed** for natural automated matches when measured simulated-game-time progression confirms acceleration and bots remain responsive. The stale packet-reported `game_speed=1.0` field is not by itself a reason to reject a regime that is measurably running at ~5x wall-clock acceleration.
