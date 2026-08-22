# Codex Start Prompt — Rival Milestone 02

You are continuing implementation of the high-end offline Rocket League 1v1 bot **Rival** for RLBot v5.

Do not respond with a new high-level plan. Work directly in `Noxoni/Rival`, implement Milestone 02, run the available tests and live evidence collection, commit stable work, and push it to `origin/main`.

## Required starting point

Canonical repository:

`https://github.com/Noxoni/Rival`

Expected pre-v2 baseline commit:

`4f2b21c00e2fcb7108ab1006fd950b066fbd0484`

At the start of the run:

1. fetch `origin/main`;
2. inspect any commits after the baseline;
3. confirm they are only the `handoff/v2.0/` package and compatible documentation changes;
4. preserve the Milestone 01 implementation baseline as recoverable history;
5. read every file under `handoff/v2.0/` before modifying code.

Do not reset or delete legitimate handoff commits just to make HEAD equal the baseline SHA.

## Central rule for Milestone 02

**Do not improve gameplay yet. Improve our ability to measure, reproduce, and compare gameplay.**

Rival's selected controller action must remain the Wisp-derived baseline action under the normal path. `STRATEGIC_OVERRIDES_ENABLED` must remain false. Do not change policy logits, action masks, argmax selection, model weights, observation semantics, tick skip, or action delay for gameplay reasons.

The objective is to finish this run with real evidence showing where the baseline fails and a repeatable way to rerun those cases later.

## Use RLBot v5 data directly

Milestone 01 telemetry was intentionally conservative. RLBot v5 exposes more useful information in `GamePacket` than Rival currently records.

Where available in the installed RLBot 2.0.0b54 schema/runtime, add telemetry for both Rival and the 1v1 opponent including:

- player name and player id;
- full physical rotation/orientation sufficient to reconstruct forward/up vectors;
- angular velocity;
- `last_input` controller state;
- `latest_touch` information and touch game time;
- jump/dodge state (`air_state`, `has_jumped`, `has_double_jumped`, `has_dodged`, `dodge_timeout`, etc. when available);
- boost;
- position and velocity;
- relevant per-tick accolades if useful for BallHit/Shot/Save/Goal event attribution.

Record ball physical rotation/angular velocity only if useful; ball position/velocity and touch attribution are required.

Do not guess field names. Inspect the installed flatbuffer objects and support missing fields safely.

## Telemetry schema v2

Upgrade telemetry deliberately rather than silently changing schema v1.

Add session-aware logging with a stable schema. A match/evidence session must have a unique id and enough metadata to identify:

- Rival Git commit SHA;
- Rival model hashes;
- opponent identity (`Nexto`, `Wisp v2-75B`, controlled probe, etc.);
- opponent local config path and executable/model hash where practical;
- match config/mutators;
- blue/orange assignment;
- start/end timestamps;
- telemetry configuration;
- RLBot/Python/Torch/RocketSim versions;
- whether the run is a natural match or controlled scenario;
- replay path if RLBot/Rocket League saved one;
- final score and termination reason.

Prefer either:

1. explicit `rival_session_start`, `rival_policy_decision`, and `rival_session_end` JSONL record types; or
2. a small session manifest JSON plus decision JSONL keyed by `session_id`.

Keep normal per-decision records compact. Full raw logits remain opt-in.

## Reproducible match runner

Implement a repository-local evidence runner using the RLBot v5 Python interface (`rlbot.managers.MatchManager`) rather than depending only on manual GUI interaction.

The runner should be able to launch at least:

- Rival vs installed Nexto;
- Rival vs installed Wisp v2-75B.

Use the already committed `reference_manifests/v1/MANIFEST.json` to discover the installed source directories and bot config files. Do not modify or copy over the installed BotPack bots.

Primary evidence must use normal Rocket League physics and **Default** game speed. `TimeWarp` may be explored for throughput only if Codex proves it does not alter Rival decision cadence/performance or contaminate the comparison; do not use TimeWarp evidence as the authoritative baseline without that proof.

Recommended match settings for evidence:

- Soccar;
- standard arena;
- normal boost/physics;
- 5-minute matches;
- skip replays after goals for throughput if compatible;
- auto-save replay when practical;
- deterministic Rival policy;
- strategic overrides false.

Alternate team/color sides across repeated runs where practical.

If launching a fully automated series is blocked by Rocket League/RLBot behavior, implement the runner as far as possible and use the same UI automation workflow that succeeded in Milestone 01 for the remaining live runs. Do not fabricate match evidence.

## Natural-match baseline set

Collect a bounded but meaningful baseline, not an endless tournament.

Target:

- at least 3 complete 5-minute matches Rival vs Nexto;
- at least 3 complete 5-minute matches Rival vs Wisp v2-75B;
- alternate Rival blue/orange when practical.

If six full matches are operationally unreasonable in one Codex run, collect at least one complete match against each opponent and continue until there are enough candidate events for the analyzers below. State exactly what was collected.

Record final scores, session ids, replay paths when available, decision record counts, missed/invalid telemetry counts, and process/performance anomalies.

This is baseline evidence, not proof of superiority.

## Controlled scenario probes

Natural matches alone are not enough. Build controlled RLBot v5 state-setting probes so important hypotheses can be replayed repeatedly.

### Probe family A — fake challenge

Goal: determine whether baseline Rival releases or throws away controlled possession because an opponent merely *looks* committed.

Create a small controlled probe opponent/script and repeatable state initialization. The exact state may be tuned during implementation, but the scenario should place Rival in a plausible ground-possession/dribble opportunity and place the opponent at a realistic challenge distance.

Parameterize challenge behavior, including at minimum:

- true commit toward ball;
- boost toward ball then brake/stop;
- boost toward ball then veer away;
- jump as if challenging without making ball contact;
- delayed challenge/shadow.

Capture the opponent's actual `last_input`, geometry, time-to-ball estimates, Rival's chosen action/top policy alternatives, ball-touch sequence, and whether Rival retains useful possession after the probe window.

The probe is measurement-only. Do not teach Rival the answer yet.

### Probe family B — resource-stressed aerial

Goal: find states where baseline Rival elects to begin or continue expensive aerial offense despite poor boost/resources.

Use state setting to create a compact grid of plausible offensive states varying at least:

- Rival boost level;
- ball height;
- ball distance;
- car/ball velocity relationship;
- opponent pressure/ETA;
- whether a useful ground alternative exists where practical.

Do not encode a gameplay rule like `boost < X => no aerial`. Candidate thresholds are allowed for **offline event detection only**, as long as they are labeled heuristics and raw measurements are preserved.

Capture whether Rival initiates/continues aerial-like actions, boost spend, pad pickups, touch success, possession after the play, recovery state, and counterattack/goal outcome within a bounded window.

## Offline evidence analyzer

Implement a CLI under a sensible repository path such as `tools/evidence/` or `scripts/` that reads telemetry schema v2 and emits machine-readable + human-readable reports.

It must segment and rank at least these candidate event classes:

1. `resource_stressed_aerial`
2. `boost_detour_possession_loss`
3. `apparent_vs_actual_challenge`

Use `handoff/v2.0/EVENT_DEFINITIONS.md` as the semantic contract.

Every candidate event should include:

- event id;
- session id/opponent;
- start/end game time;
- relevant raw measurements;
- derived metrics;
- outcome window;
- confidence/severity score used only for ranking;
- pointers to surrounding decision records;
- replay path and approximate timestamp when available.

Do not label an event a confirmed defect solely because a heuristic fired.

## Replayable evaluation fixtures

For the most useful candidate windows, preserve small reproducible fixtures without committing huge raw logs.

Prefer a compact JSON fixture containing enough state/metadata to:

- feed the policy/analysis layer offline where possible; and/or
- initialize an RLBot state-setting probe near the original state.

Create a curated fixture directory with a README and schema. Start with at least one fixture per event class that is actually observed. If an event class does not occur, document that rather than inventing a fixture.

## Tests

Add/extend tests for:

- telemetry schema v2 serialization and backward-safe reading where practical;
- session start/end metadata;
- direct v5 `last_input` / `latest_touch` extraction with missing-field safety;
- event segmentation boundaries;
- event ranking determinism;
- analyzer on synthetic/curated fixture data;
- match-config generation/discovery without modifying installed BotPack;
- controlled-probe configuration/state generation;
- proof that normal Rival policy action selection is unchanged by Milestone 02 instrumentation.

Run the existing 15 tests as well. Do not regress Milestone 01.

## Evidence and repository hygiene

Commit:

- implementation code;
- tests;
- match/probe templates or generators;
- analyzer code;
- documentation;
- small curated fixtures;
- small summary reports;
- session/evidence manifests that identify raw local artifacts by hash/path.

Do not commit by default:

- large raw JSONL telemetry sets;
- Rocket League `.replay` binaries;
- installed Wisp/Nexto executable trees;
- virtual environments;
- machine caches.

Update `.gitignore` as needed while retaining traceable manifests.

## Milestone completion

Milestone 02 is complete when:

1. Rival gameplay policy remains frozen relative to Milestone 01.
2. Automated or reproducible launch exists for Rival vs Nexto and Wisp.
3. Telemetry schema v2 captures direct opponent/player control/touch/orientation information.
4. At least one real live evidence session against each reference bot was collected.
5. Controlled fake-challenge and low-resource-aerial probes actually run, unless a concrete framework blocker is documented.
6. The analyzer produces ranked candidate events.
7. Useful event windows can be replayed or reconstructed.
8. Tests pass.
9. Stable commits are pushed to `Noxoni/Rival`.
10. The final report identifies **one recommended first behavior defect for Milestone 03**, selected from evidence, without implementing the correction yet.

## End-of-run report

Return:

- pushed commit SHA(s);
- local/remote HEAD verification;
- files changed;
- tests and exact results;
- live match/session count by opponent;
- final scores only as baseline observations, not skill claims;
- controlled probe counts;
- event counts by class;
- top 3–5 candidate events with session/time references;
- curated fixture paths;
- raw evidence paths/hashes that remain local;
- any framework/runtime blockers;
- the single recommended Milestone 03 defect and why the evidence supports choosing it first.

Prioritize executable tooling, real evidence, and reproducibility over prose.
