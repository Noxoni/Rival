# Rival Milestone 02 — Evidence Harness Specification

## Goal

Create a repeatable RLBot v5 evidence system around the frozen Rival Milestone 01 policy so we can identify and replay real strategic failures before changing gameplay.

## Baseline invariant

The Milestone 01 gameplay baseline is commit:

`4f2b21c00e2fcb7108ab1006fd950b066fbd0484`

Milestone 02 may refactor telemetry/session plumbing as required, but the normal controller output for a given equivalent observation/action-mask state must remain unchanged.

`STRATEGIC_OVERRIDES_ENABLED` remains false.

## Suggested repository additions

Exact paths may be adjusted, but preserve these responsibilities:

```text
bot/
  telemetry/
    ... schema-v2/session logging ...
  analysis/
    ... richer packet/state extraction ...

tools/evidence/
  analyze.py
  events.py
  io.py
  report.py
  fixtures.py

scripts/
  run_evidence_suite.py
  generate_match_config.py
  run_probe_suite.py

matches/
  templates/

probes/
  fake_challenge/
  resource_aerial/

fixtures/evidence/
  README.md
  <small curated JSON fixtures>

docs/
  EVIDENCE_HARNESS.md
  MILESTONE_02_RESULTS.md
```

Do not force the tree if a cleaner implementation fits the existing project better.

---

# A. Telemetry schema v2

## Required session identity

Every decision record must be attributable to a specific evidence session.

Required session metadata:

- `schema_version`;
- `session_id`;
- source type (`natural_match`, `controlled_probe`, synthetic test);
- Rival Git commit;
- model SHA-256 values;
- opponent identity;
- team/color assignment;
- match/probe parameters;
- runtime dependency versions;
- UTC start/end timestamps;
- final score when applicable;
- termination reason;
- raw telemetry file path + hash after completion;
- replay path + hash if available.

## Required direct packet fields

Extend the current snapshot using RLBot v5 packet data where available.

For Rival and the opponent, capture:

- name/player id/team;
- position;
- linear velocity;
- rotation/orientation;
- angular velocity;
- boost;
- last controller input;
- latest-touch information;
- jump/dodge/air state;
- relevant accolades/event markers.

For the ball:

- position;
- linear velocity;
- enough attribution data to identify touch sequence via player latest-touch information;
- rotation/angular velocity only if useful.

For boost pads:

- active state;
- timer if available;
- pad size/location mapping;
- nearest useful opportunities.

## Performance

Telemetry must not materially destabilize the bot.

- avoid full-logit dumps by default;
- batch/line-buffer reasonably;
- measure decision/runtime impact during live runs;
- safely disable logging after write errors;
- close/finalize the session manifest cleanly.

---

# B. Match runner

Use the current RLBot v5 Python interface and `MatchManager.start_match(Path(...))` or an equivalent supported v5 API.

The runner must discover installed Wisp/Nexto configs using `reference_manifests/v1/MANIFEST.json` and validate that the expected files still match recorded provenance where practical.

Generate machine-local match configs rather than committing hard-coded absolute user paths into portable templates.

Required natural match modes:

```text
Rival Blue  vs Nexto Orange
Rival Orange vs Nexto Blue
Rival Blue  vs Wisp Orange
Rival Orange vs Wisp Blue
```

Primary evidence settings:

- Soccar;
- standard map;
- five-minute game;
- default game speed;
- default boost/gravity/ball physics;
- deterministic Rival;
- state setting enabled only when needed;
- replay auto-save when reliable;
- goal replays may be skipped to reduce dead time.

Runner requirements:

- unique session id per match;
- isolated telemetry output path;
- clear stdout status;
- match end detection;
- final score capture;
- robust cleanup;
- no installed BotPack mutation.

---

# C. Controlled probes

## C1. Fake challenge probe

Create reproducible state-setting scenarios around a plausible Rival possession state.

The controlled challenger must support at least:

- true commit;
- boost then brake;
- boost then veer;
- jump fake;
- delayed/shadow challenge.

Parameterize useful variables such as:

- initial separation;
- lateral offset;
- challenger speed;
- fake/abort timing;
- Rival dribble/ball position;
- Rival/opponent boost.

Do not make the scenario physically absurd merely to force a policy response.

Each run must have a known ground-truth probe label and a bounded analysis window.

## C2. Resource-stressed aerial probe

Create reproducible offensive states spanning a compact, documented parameter grid.

At minimum vary:

- Rival boost;
- ball height;
- car-ball distance;
- relevant car/ball velocity;
- opponent pressure;
- field position.

Capture Rival's unmodified policy response and outcome. Keep the grid small enough to finish but broad enough to expose transitions.

A useful starting grid can be adaptive rather than Cartesian: begin broad, then add states near interesting action transitions.

---

# D. Evidence analyzer

Implement a CLI that can process one file, one session directory, or the complete current evidence set.

Example shape (exact CLI may differ):

```text
python -m tools.evidence.analyze telemetry/session.jsonl
python -m tools.evidence.analyze evidence/raw/ --format both
```

Outputs:

1. JSON report suitable for later tooling;
2. Markdown summary suitable for Git review.

Required event classes are defined in `EVENT_DEFINITIONS.md`.

The analyzer must preserve configurable detector parameters in the output report so an event ranking can be reproduced later.

## Sequence correctness

Handle:

- kickoff/goal resets;
- match end;
- missing records;
- duplicated or non-monotonic records;
- player identity/team side changes only if encountered;
- telemetry schema version validation.

Do not treat events spanning a goal reset as one continuous play.

---

# E. Curated replayable fixtures

Do not commit the whole raw evidence corpus.

Create compact fixtures from useful event windows.

Each fixture should contain:

- source session id;
- source event id;
- source raw file hash;
- game-time window;
- relevant sequential records and/or minimal reconstructed state;
- opponent identity;
- event class;
- detector version/parameters;
- expected baseline behavior observations;
- no assertion yet about what the improved behavior must be.

Where state setting can reconstruct the scene, include enough data for the probe runner to initialize a near-equivalent state.

---

# F. Baseline evidence collection

## Natural matches

Preferred target:

| Opponent | Complete matches | Side handling |
| --- | ---: | --- |
| Nexto | 3+ | alternate blue/orange |
| Wisp v2-75B | 3+ | alternate blue/orange |

Minimum acceptable if operational constraints arise: one complete match versus each reference bot plus enough live/probe data for meaningful candidate events.

Record all actual counts; do not imply unrun matches.

## Controlled probes

Target at least:

- 5 repetitions per fake-challenge behavior after scenario stabilization;
- enough resource-aerial states to expose at least two distinct policy-response regions or clearly document that no transition appeared.

The exact count may adapt to what the policy does. Prefer informative coverage over arbitrary volume.

---

# G. Tests and verification

Run the entire existing test suite plus new tests.

Required new coverage:

- schema-v2 session lifecycle;
- v5 direct packet field extraction;
- last-input extraction;
- latest-touch extraction;
- event boundary segmentation;
- detector parameter persistence;
- deterministic ranking;
- probe parameter serialization;
- portable match-config generation;
- installed reference read-only validation;
- unchanged policy action selection.

Also run:

- Ruff on Rival-authored/modified code;
- `compileall`;
- `git diff --check`;
- a real-model smoke test;
- at least one real live match after instrumentation changes.

## Policy freeze verification

At minimum retain the existing masked-argmax parity tests and add a recorded/synthetic observation regression set demonstrating that Milestone 02 instrumentation does not change selected action indexes.

---

# H. Results document

Create `docs/MILESTONE_02_RESULTS.md` containing:

- exact implementation commit;
- environment versions;
- natural-match sessions and scores;
- probe counts;
- telemetry performance notes;
- detector parameters;
- event counts;
- top candidate events;
- fixture paths;
- raw local artifact paths/hashes;
- limitations;
- recommended Milestone 03 target.

## Selecting the Milestone 03 target

Choose one defect based on:

1. repeated evidence or a very high-confidence controlled reproduction;
2. meaningful competitive cost;
3. ability to evaluate before/after with the v2 harness;
4. limited enough scope that the first gameplay change can be isolated.

Do not pick a behavior simply because it was mentioned in the original hypothesis if the evidence does not support it.

---

# Acceptance criteria

Milestone 02 passes only when:

- Milestone 01 gameplay selection remains frozen;
- telemetry schema v2 is implemented and tested;
- direct RLBot v5 controller/touch/orientation data is captured;
- match evidence against Nexto and Wisp exists;
- fake-challenge probes run;
- resource-aerial probes run;
- analyzers produce ranked events;
- at least some useful events have compact fixtures;
- tests/verification pass;
- results are documented honestly;
- stable work is pushed to `Noxoni/Rival`;
- one evidence-backed Milestone 03 target is recommended but not yet corrected.
