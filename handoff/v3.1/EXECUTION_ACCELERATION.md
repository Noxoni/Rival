# Rival v3.1 — Accelerated Live-Test Execution

## Objective

Preserve full-match coverage while reducing wall-clock testing time.

The default target for live automated validation is **5.0x Rocket League game speed**. Do not replace five-minute natural matches with one-minute matches merely for convenience.

---

## A. 5x game-speed implementation

RLBot v5 exposes raw game-speed state through desired match/game state. Implement accelerated live runs by state-setting the game-speed multiplier, not by using the in-game `TimeWarp` mutator.

Use the installed RLBot v5 Python API available in this repository. The expected shape is equivalent to:

```python
manager.set_game_state(
    match_info=flat.DesiredMatchInfo(game_speed=5.0)
)
```

Exact class/import names must follow the installed `rlbot`/flatbuffer version.

### Requirements

1. Add a runner-level parameter/environment setting for requested live game speed. Prefer a clear surface such as:

   `RIVAL_TEST_GAME_SPEED=5.0`

2. Full natural-match configuration remains five minutes, normal boost, normal gravity, normal physics, normal score/overtime rules.

3. Enable state setting only as required to apply the raw speed multiplier. Do not state-set cars, ball, boost, score, or clock during a natural match.

4. Apply 5.0x after the match reaches a usable phase. If Rocket League resets game speed after goals/kickoffs/match transitions, detect that through packets and re-apply 5.0x without modifying any other state.

5. Record both requested and observed game speed in every session manifest.

6. Verify observed packet `match_info.game_speed` is approximately 5.0 for sustained active play before calling a session accelerated.

7. Record wall duration and effective acceleration ratio:

   `advanced game seconds / elapsed wall seconds`

8. Preserve Wisp/Nexto/Rival tick skip, policy cadence code, model weights, observation semantics, and action delay. Do not compensate for acceleration by changing the bots.

---

## B. Stability gate for 5x

The goal is 5x, not a theoretical purity test. Attempt 5.0x first.

Before spending the full experiment budget, run a short bounded live smoke for Rival plus each installed reference type used in the milestone and verify:

- the game actually reports about 5.0x speed;
- both bots remain connected and controlling cars;
- no repeated bot crashes/restarts;
- telemetry remains parseable;
- decision logging continues throughout active play;
- RLBot queue/missed-packet warnings do not become severe enough to make results obviously unusable;
- match state progresses normally through goals/kickoffs/end states.

If 5.0x is stable, use it for all remaining automated live Milestone 03 validation.

If 5.0x is not stable, do not silently pretend it worked. Record the failure and use the highest stable multiplier found with a very small bounded search, preferably 4x then 3x. Do not drop below 2x without documenting why.

The milestone result must state the actual requested/observed multiplier used.

---

## C. Full natural matches remain full natural matches

At 5.0x, a five-minute regulation clock should consume roughly one minute of wall time plus goal/kickoff/overtime/loading overhead.

Do not shorten the game clock for Stage 4.

The v3.0 natural-match cap remains:

- at most 3 treatment matches vs Nexto;
- at most 3 treatment matches vs Wisp v2-75B;
- no extra matches because results are inconvenient.

Any baseline/off comparisons specifically required by the experiment matrix should also use the same accelerated mode wherever possible so treatment and control experience the same execution regime.

---

## D. Parallel live-match capability test

Parallelism is desirable, but RLBot v5's normal `MatchManager` workflow discovers/attaches to an existing RLBotServer. Starting two ordinary runners is **not** sufficient isolation and may cause both runners to control/restart the same Rocket League session.

Attempt exactly one bounded capability test for **two independent live lanes**.

A lane is valid only if it has:

- its own RLBotServer process/connection;
- a distinct server port;
- its own Rocket League process/session;
- its own bot child processes tied to the correct `RLBOT_SERVER_PORT`;
- independent match configuration and state;
- independent telemetry/session directories;
- no process, port, state, or match interference with the other lane.

Do not count two clients attached to the same RLBotServer/Rocket League match as parallel matches.

### Parallel test procedure

1. Preserve the stable single-lane 5x path first.
2. Attempt two isolated lanes only.
3. Run a short smoke match/window in both simultaneously.
4. Verify independent packet clocks, teams/scores, process trees, server ports, session IDs, and telemetry.
5. Stop both cleanly and verify neither lane killed/restarted the other unexpectedly.

### If two lanes work

Use a concurrency of **2** for live Milestone 03 games. Do not automatically increase beyond 2 in this milestone.

Schedule independent match jobs across the two lanes while keeping baseline/treatment pairing and session metadata explicit.

### If two lanes do not work cleanly

Stop the parallel experiment after that bounded attempt. Record the limitation and continue **sequentially at 5x**.

Do not spend the milestone reverse-engineering Steam/Epic launcher restrictions, cloning installations, patching RLBot core, or constructing VMs merely to obtain concurrency.

Offline fixture/unit/analyzer work may still run concurrently with the single live lane when safe.

---

## E. Result integrity

Every accelerated live session manifest/report must capture at minimum:

- requested game speed;
- observed median/min/max game speed during active play;
- wall duration;
- game-duration advanced;
- effective wall-clock acceleration;
- lane identifier;
- RLBotServer PID/port when practical;
- Rocket League PID when practical;
- bot PIDs when practical;
- packet/decision counts;
- parse/integrity errors;
- relevant queue/missed-packet warnings;
- final score/status;
- whether the session was sequential or parallel.

Do not compare a 1x and 5x result as though execution regime were irrelevant; label execution speed in reports. The primary Milestone 03 A/B should use the same speed regime for both sides of the comparison.
