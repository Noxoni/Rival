# Rival Handoff v4.1 — Natural-play optimization

This handoff supersedes the unexecuted v4.0 deterministic-pairing direction.

## Start here

Codex must execute:

`handoff/v4.1/CODEX_START_PROMPT.md`

and read:

`handoff/v4.1/NATURAL_PLAY_LOOP.md`

before changing implementation code.

## Direction

Rival should improve from naturally occurring Rocket League play at accelerated simulation speed. The primary loop is:

`natural matches -> telemetry -> recurring state/outcome pattern -> one live state-conditioned adjustment -> natural matches -> aggregate comparison`

Scripted probe labels are not normal gameplay inputs and are not the main training/evaluation universe.

## Current gameplay state

Milestone 03's challenge-calibration experiment was rejected and remains disabled by default. Rival still uses the Wisp-derived baseline unless a new v4.1 treatment is explicitly accepted from natural-play evidence.

## Throughput

Use full five-minute Soccar at approximately 5x effective wall-clock acceleration when bot responsiveness and telemetry remain healthy. Goal replays are skipped; replay auto-save, debug rendering and performance overlay are disabled for automated runs.

Previous handoffs remain historical and recoverable under `handoff/`.