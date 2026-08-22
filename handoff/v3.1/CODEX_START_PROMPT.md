# Codex Start Prompt — Rival Milestone 03 v3.1

Continue the high-end offline Rocket League 1v1 bot **Rival** in `Noxoni/Rival`.

Do not produce another high-level design pass. Execute Milestone 03 and push stable work/results to `origin/main`.

## Required reading

Read these in order before modifying implementation code:

1. `handoff/v3.1/VERSION.md`
2. `handoff/v3.1/EXECUTION_ACCELERATION.md`
3. every file under `handoff/v3.0/`
4. `docs/MILESTONE_02_RESULTS.md`
5. `evidence/results/v2/candidate_events.md`
6. the curated challenge fixture(s)

The full Milestone 03 gameplay design, experiment matrix, acceptance criteria, and evidence rules remain those in `handoff/v3.0/`. **v3.1 overrides only how live tests are executed.**

## Starting state

Completed Milestone 02 evidence baseline:

`e7b68c6e33faf6fc644a3fc9a07e811d43d2918e`

Frozen original Wisp-equivalent gameplay baseline:

`4f2b21c00e2fcb7108ab1006fd950b066fbd0484`

Preserve all legitimate handoff commits after the evidence baseline. Do not reset, squash, or delete historical handoffs.

Leave the user's untracked `bot.7z` and unrelated local files untouched.

## Gameplay scope

Exactly the same as v3.0:

**Challenge-commitment calibration only.**

Do not fix boost greed, aerial-resource behavior, model architecture, observations, tick skip, action delay, or unrelated code in this milestone unless required to correct a blocking test/runtime defect.

Use Wisp's existing legal discrete action set and conservative policy re-ranking. Do not synthesize controller values.

## Execution-speed override

For live automated testing, preserve full five-minute games and target a raw Rocket League game-speed multiplier of:

`5.0x`

Do **not** shorten natural matches to one minute merely to accelerate the workflow.

Implement runner support equivalent to:

`RIVAL_TEST_GAME_SPEED=5.0`

using RLBot v5 desired match/game state. Do not use the in-game `TimeWarp` mutator as a substitute.

The exact implementation and stability checks are mandatory in:

`handoff/v3.1/EXECUTION_ACCELERATION.md`

Before consuming the full Stage 3/4 live-test budget, validate that 5x is actually observed in packets and that Rival plus each reference bot remain stable and responsive.

If 5x is stable, use it for the remaining live Milestone 03 tests.

If 5x is objectively unstable, record the evidence and use the highest stable multiplier from a small fallback search. Do not silently run at 1x.

## Parallelism override

After the single-lane 5x path is stable, perform **one bounded capability test** for two simultaneous independent live matches.

Two lanes count as valid only if each has a distinct RLBotServer/port, Rocket League process/session, agent process tree, session state, and telemetry output.

Simply starting two MatchManager clients connected to the same server/session does not count.

If two isolated lanes work, use concurrency `2` for the remaining live match jobs.

If they do not work cleanly, stop trying and continue sequentially at 5x. Do not spend the milestone modifying RLBot core, fighting launcher restrictions, building VMs, or cloning the Rocket League install for concurrency.

## Experiment order

Keep the v3.0 order:

1. static/unit verification;
2. offline fixture/evidence evaluation;
3. paired controlled A/B;
4. natural validation only if controlled gates pass.

Use 5x for controlled live probes and natural matches after the 5x smoke gate passes. Baseline/off and treatment/intervene comparisons must use the same execution-speed regime.

## Natural-match budget

Do not reduce the five-minute match clock.

The existing v3.0 cap remains:

- maximum 3 treatment natural matches vs installed Nexto;
- maximum 3 treatment natural matches vs installed Wisp v2-75B;
- no additional natural matches because results are inconvenient.

If valid two-lane parallelism exists, schedule independent jobs across both lanes. Otherwise run the same full games sequentially at 5x.

## Required reporting additions

In addition to every v3.0 required result, report and commit:

- requested game-speed multiplier;
- observed game-speed statistics;
- effective game-seconds/wall-second acceleration;
- 5x smoke result;
- whether two-lane concurrency was supported;
- if supported: lane/server-port/process isolation evidence and concurrency used;
- if unsupported: exact reason and confirmation that testing continued sequentially;
- wall duration for each live session;
- packet/decision counts and queue/missed-packet warnings by execution regime.

Do the work in this run. The goal is the same evidence-quality Milestone 03 experiment as v3.0, but without spending five real minutes watching every five-minute Rocket League game.
