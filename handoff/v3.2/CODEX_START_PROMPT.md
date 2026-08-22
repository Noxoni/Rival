# Codex Start Prompt — Rival Milestone 03 v3.2

Continue Rival Milestone 03 in `Noxoni/Rival`.

Do not redesign the project. Execute the existing challenge-commitment experiment and push stable work/results to `origin/main`.

## Required reading order

1. `handoff/v3.2/VERSION.md`
2. `handoff/v3.2/CONFIG_OPTIMIZATIONS.md`
3. every file under `handoff/v3.1/`
4. every file under `handoff/v3.0/`
5. `docs/MILESTONE_02_RESULTS.md`
6. `evidence/results/v2/candidate_events.md`
7. the curated challenge fixture(s)

v3.2 overrides only automated match configuration. v3.1 remains authoritative for 5x acceleration and the bounded two-lane parallelism test. v3.0 remains authoritative for gameplay logic, A/B design, acceptance criteria, match budget, and evidence semantics.

## Required config cleanup before live testing

Update the automated runner so test matches use the configuration contract in `CONFIG_OPTIMIZATIONS.md`.

In particular:

- keep `skip_replays = true`;
- change automated live sessions to `auto_save_replay = false`;
- use `DebugRendering.AlwaysOff` (or installed-version equivalent) for automation;
- use `PerformanceMonitor.NeverShow` for automation;
- keep `auto_start_agents = true`;
- keep `wait_for_agents = true`;
- keep normal kickoff countdowns for natural matches (`instant_start = false`);
- keep `ExistingMatchBehavior.Restart` for independent natural evidence games;
- keep standard five-minute Soccar gameplay mutators;
- allow state setting on accelerated natural runs only as needed to maintain the v3.1 raw 5x game-speed multiplier;
- do not state-set cars, ball, boost, score, or clock during natural matches.

Add tests for these runner invariants before live execution.

## Live execution

Target 5.0x raw game speed as defined in v3.1.

Keep full five-minute natural matches. Do not replace them with shortened matches.

After the single-lane accelerated path is stable, perform the one bounded two-lane isolation test from v3.1. If it fails, continue sequentially at the highest validated acceleration multiplier without spending the milestone on multi-instance infrastructure.

## Everything else

Follow `handoff/v3.0/CODEX_START_PROMPT.md` and its supporting design/spec/experiment matrix exactly for challenge calibration.

The experiment may be accepted or rejected based on evidence. Do not add unrelated gameplay rules to force success.

Leave the user's untracked `bot.7z` and unrelated local files untouched.

At the end, include the v3.1 acceleration evidence plus confirmation that the v3.2 automated-match configuration was actually used in live sessions.
