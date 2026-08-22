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
- Milestone 03 natural acceptance matches consumed: **0 / 6**

See `docs/MILESTONE_03_RESULTS.md`, `docs/MILESTONE_02_RESULTS.md`, and `evidence/results/` for the current evidence.

## Milestone 03 result

Challenge calibration is technically implemented with explicit `off`, `observe`, and `intervene` modes, but it is **not enabled**. Neither controlled treatment parameter attempt produced an actual intervention, so differences between independently launched baseline/treatment trajectories were not causal. The controlled gate failed and no natural acceptance matches were launched.

Milestone 03 also demonstrated real accelerated simulation: direct desired match game speed produced approximately 5x/4x/3x/2x simulated-time progression, even though RLBot packets continued reporting `match_info.game_speed=1.0`. That packet field is now treated as stale diagnostic data rather than proof that acceleration failed. The first bounded two-lane concurrency test failed isolation because both RLBotServer instances raced onto the same port.

## Current Codex handoff — v4.0

Start here:

`handoff/v4.0/CODEX_START_PROMPT.md`

Codex must read the complete `handoff/v4.0/` package before modifying implementation code.

Milestone 04 fixes the experiment before revisiting gameplay tuning. The inherited Wisp observation builder randomly shuffles teammate/opponent observation slots on every decision, the model input includes previous controller action, and Rival carries additional runtime history across state-set probe cases. Separately launched `off` and `observe` runs therefore need not receive identical policy inputs even when treatment never acts.

v4 introduces controlled-test-only seeded observation shuffling, a complete controlled-case runtime reset, exact model-input/legal-mask fingerprints, paired case identities, and a bounded search for a **repeatable refined release-sensitive challenge exposure**. Normal gameplay keeps the inherited Wisp shuffle behavior and challenge calibration remains off unless a later causal treatment gate passes.

For controlled testing, 1x establishes the initial deterministic reference. The same paired fixture is then tested at 5x; if pairing remains reproducible, 5x becomes the preferred controlled speed. The packet `game_speed` echo is not a rejection criterion. Full five-minute Soccar remains the eventual natural-match format.

v4 also authorizes one new bounded two-lane capability test after the deterministic harness is stable because the first attempt failed specifically from a server-port race that was subsequently addressed. If Rocket League/Steam still supports only one independent game process, parallel live matches are marked unsupported on this machine and the project moves on.

Previous handoffs remain under `handoff/` as recoverable project history.

## Distribution

For another Windows PC, do not distribute only `bot/`: the development TOML depends on the repository-local `.venv`. Build the self-contained Windows x64 ZIP with `scripts/build_windows_release.ps1`; the complete procedure and verification contract are in `docs/BUILD_WINDOWS_RELEASE.md`.

## Local RLBot BotPack references

The installed RLBot v5 bots used as read-only references are under:

`C:\Users\patri\AppData\Local\RLBot5\bots`

The primary local references are Wisp v2-75B and Nexto. Exact local and upstream provenance is recorded in `docs/SOURCE_PROVENANCE.md`; machine-local snapshots remain Git-ignored.

## Repository policy

This repository is the canonical location for Rival implementation work, tests, provenance, evidence tooling, progress, and stable commits. Meaningful completed work should be committed and pushed here rather than existing only in a local Codex workspace.
