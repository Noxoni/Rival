# Rival

Rival is a high-end offline Rocket League 1v1 bot project for **RLBot v5**. The current default gameplay remains the verified Milestone 02 Wisp v2-75B-derived baseline. Milestone 03 implemented and tested an experimental challenge-commitment calibration layer, but the treatment was rejected and remains disabled.

Rival is for offline RLBot play only. It must not be used to cheat or otherwise break Rocket League's terms of service.

## Current verified baseline

- RLBot display name: `Rival Dev`
- Default agent id: `noxoni/rival/dev-v1`
- Runtime config: `bot/rival.bot.toml`
- Completed Milestone 03 result commit: `e4cc175a4259202d5cc7ee437abef224b731354f`
- Completed Milestone 02 evidence baseline: `e7b68c6e33faf6fc644a3fc9a07e811d43d2918e`
- Frozen Wisp-equivalent gameplay baseline: `4f2b21c00e2fcb7108ab1006fd950b066fbd0484`
- Baseline models: unchanged Wisp v2-75B `POLICY.lt` and `SHARED_HEAD.lt`
- Challenge calibration default: `off`
- Six Milestone 02 natural baseline matches completed: three vs Nexto, three vs Wisp v2-75B

See `docs/MILESTONE_03_RESULTS.md`, `docs/MILESTONE_02_RESULTS.md`, and `evidence/results/` for the current evidence.

## Current direction

Milestone 03 showed that trying to judge gameplay from independently scripted challenge trajectories was not useful enough: both treatment attempts applied zero interventions while their trajectories still diverged. The project is therefore moving away from scenario-specific tuning as the primary optimization method.

Rival should improve from **natural accelerated Rocket League play**:

`natural matches -> telemetry -> recurring state/outcome pattern -> one live state-conditioned adjustment -> natural matches -> aggregate comparison`

Gameplay logic should use current observable opponent/ball/Rival state and short history, not labels from hand-authored test scenarios.

Milestone 03 also demonstrated genuine accelerated simulation: requested 5x produced approximately 4.92x and 5.00x simulated-game-time progression per wall second against Nexto and Wisp. The packet `match_info.game_speed` echo remained at `1.0`, so that field is treated as stale diagnostic data rather than proof that acceleration failed.

## Current Codex handoff — v4.1

Start here:

`handoff/v4.1/CODEX_START_PROMPT.md`

Codex must read the complete `handoff/v4.1/` package before modifying implementation code.

v4.1 supersedes the unexecuted v4.0 deterministic-pairing direction. The main development environment is now full five-minute natural 1v1 matches at approximately **5x effective simulation speed**, with telemetry aggregated across many unrelated trajectories. Scripted probes may remain as optional regression/smoke checks, but they are not the main training set or acceptance gate.

The intended v4.1 run gathers a natural accelerated baseline batch against installed Nexto and Wisp v2-75B, ranks recurring high-impact behavior from telemetry, implements one state-conditioned correction using live observations, then runs another natural accelerated batch and compares aggregate outcomes. The rejected Milestone 03 challenge parameters remain disabled unless a new natural-play result supports a different treatment.

Parallel Rocket League instances remain optional; do not spend substantial engineering time on them. Sequential 5x matches are already the primary throughput improvement.

Previous handoffs remain under `handoff/` as recoverable project history.

## Distribution

For another Windows PC, do not distribute only `bot/`: the development TOML depends on the repository-local `.venv`. Build the self-contained Windows x64 ZIP with `scripts/build_windows_release.ps1`; the complete procedure and verification contract are in `docs/BUILD_WINDOWS_RELEASE.md`.

## Local RLBot BotPack references

The installed RLBot v5 bots used as read-only references are under:

`C:\Users\patri\AppData\Local\RLBot5\bots`

The primary local references are Wisp v2-75B and Nexto. Exact local and upstream provenance is recorded in `docs/SOURCE_PROVENANCE.md`; machine-local snapshots remain Git-ignored.

## Repository policy

This repository is the canonical location for Rival implementation work, tests, provenance, evidence tooling, progress, and stable commits. Meaningful completed work should be committed and pushed here rather than existing only in a local Codex workspace.
