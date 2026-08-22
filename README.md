# Rival

Rival is a high-end offline Rocket League 1v1 bot project for **RLBot v5**. The current `Rival Dev` implementation is an instrumented Wisp v2-75B baseline: it preserves Wisp's masked policy action selection while exposing the decision and tactical/resource state for evidence-driven strategy work.

Rival is for offline RLBot play only. It must not be used to cheat or otherwise break Rocket League's terms of service.

## Current implementation

- RLBot display name: `Rival Dev`
- Agent id: `noxoni/rival/dev-v1`
- Runtime config: `bot/rival.bot.toml`
- Frozen Milestone 01 baseline: `4f2b21c00e2fcb7108ab1006fd950b066fbd0484`
- Baseline models: unchanged Wisp v2-75B `POLICY.lt` and `SHARED_HEAD.lt`
- Policy seam: raw/masked logits, legal mask, selected action, top candidates, probabilities, confidence, margin, ticks, and timestamps
- Measurement-only analysis: boost/resource, ball/possession, closing-speed, ETA, airborne, action-use, score, clock, and boost-map fields
- Toggleable JSONL telemetry
- Strategic overrides remain disabled through Milestone 02

See `docs/RUN_LOCAL.md` for environment setup and the completed Milestone 01 verification record in `docs/VERIFICATION_2026-08-22.md`.

For another Windows PC, do not distribute only `bot/`: the development TOML depends on the repository-local `.venv`. Build the self-contained Windows x64 ZIP with `scripts/build_windows_release.ps1`; the complete procedure and verification contract are in `docs/BUILD_WINDOWS_RELEASE.md`.

## Current Codex handoff — v2.0

Start here:

`handoff/v2.0/CODEX_START_PROMPT.md`

Codex must read the complete `handoff/v2.0/` package before modifying implementation code.

Milestone 02 builds the repeatable evidence harness: session-aware RLBot v5 telemetry, direct player input/touch data, Rival-vs-Nexto and Rival-vs-Wisp match collection, controlled fake-challenge and resource-stressed-aerial probes, offline event extraction, and replayable fixtures. Gameplay policy remains frozen until that evidence selects the first Milestone 03 correction.

Previous handoffs remain under `handoff/` as recoverable project history.

## Local RLBot BotPack references

The user's installed RLBot v5 bots are located at:

`C:\Users\patri\AppData\Local\RLBot5\bots`

Treat the installed BotPack as read-only. The primary local references are Wisp v2-75B and Nexto. Exact local and upstream provenance is recorded in `docs/SOURCE_PROVENANCE.md`; machine-local snapshots remain Git-ignored.

## Repository policy

This repository is the canonical location for Rival implementation work, tests, provenance, evidence tooling, progress, and stable commits. Meaningful completed work should be committed and pushed here rather than existing only in a local Codex workspace.
