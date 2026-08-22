# Rival Handoff v2.0 — Evidence Harness

Milestone 01 is complete and remotely verified. Rival is a runnable RLBot v5 bot using the Wisp v2-75B policy baseline with structured policy inspection, tactical metrics, and toggleable JSONL telemetry.

This handoff begins **Milestone 02: repeatable evidence collection and controlled failure reproduction**.

## Start here

Codex must execute:

`handoff/v2.0/CODEX_START_PROMPT.md`

and use:

- `handoff/v2.0/MILESTONE_02_SPEC.md`
- `handoff/v2.0/EVENT_DEFINITIONS.md`

as acceptance criteria and event semantics.

## Frozen baseline

Milestone 02 starts from:

`4f2b21c00e2fcb7108ab1006fd950b066fbd0484`

This commit is the recoverable Rival Milestone 01 baseline. Gameplay output should remain unchanged during Milestone 02.

## Why this milestone exists

Watching a strong bot make a bad decision is useful, but not enough to improve it safely. We need to preserve the exact state around bad decisions, identify recurring event classes, and rerun the same situations after changes.

Milestone 02 therefore adds:

1. reproducible Rival-vs-Nexto and Rival-vs-Wisp match launching;
2. session-aware telemetry;
3. direct RLBot v5 player input/touch data where available;
4. event-window extraction;
5. controlled fake-challenge probes;
6. controlled low-resource aerial probes;
7. baseline reports and small replayable fixtures;
8. no strategic override yet.

## Primary behavioral hypotheses

### Nexto reference

Useful behavior:

- strong possession acquisition;
- rapid controlled dribble to flick conversion;
- strong ground offense.

Known weakness to avoid in Rival:

- reacting too early to apparent challenges / fake challenges.

### Wisp reference

Useful behavior:

- boost denial and starvation;
- advanced aerial mechanics;
- strong general play;
- clock/possession tendencies while leading.

Known weaknesses to investigate:

- boost-route greed relative to possession;
- long aerial/air-dribble commitments with poor resource margin.

Milestone 02 must measure these rather than hard-code reactions to them.
