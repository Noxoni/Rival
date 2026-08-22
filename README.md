# Rival

Rival is a high-end offline Rocket League 1v1 bot project for **RLBot v5**. The current `Rival Dev` milestone is an instrumented Wisp v2-75B baseline: it preserves Wisp's masked policy action selection while exposing the decision and tactical/resource state for later evidence-driven strategy work.

Rival is for offline RLBot play only. It must not be used to cheat or otherwise break Rocket League's terms of service.

## Current implementation

- RLBot display name: `Rival Dev`
- Agent id: `noxoni/rival/dev-v1`
- Runtime config: `bot/rival.bot.toml`
- Baseline models: unchanged Wisp v2-75B `POLICY.lt` and `SHARED_HEAD.lt`
- Policy seam: raw/masked logits, legal mask, selected action, top candidates, probabilities, confidence, margin, ticks, and timestamps
- Measurement-only analysis: boost/resource, ball/possession, closing-speed, ETA, airborne, action-use, score, clock, and boost-map fields
- Toggleable JSONL telemetry, disabled by default
- Strategic overrides: disabled for Milestone 01

See `docs/RUN_LOCAL.md` for environment setup, tests, telemetry, and the exact RLBot v5 launch procedure. The completed Milestone 01 verification record is in `docs/VERIFICATION_2026-08-22.md`.

## Current Codex handoff

Start here:

`handoff/v1.1/CODEX_START_PROMPT.md`

Codex should read the complete `handoff/v1.1/` package before modifying implementation code.

## Local RLBot BotPack references

The user's installed RLBot v5 bots are located at:

`C:\Users\patri\AppData\Local\RLBot5\bots`

Treat the installed BotPack as read-only. The primary local references are Wisp v2-75B and Nexto. Exact local and upstream provenance is recorded in `docs/SOURCE_PROVENANCE.md`; machine-local snapshots remain Git-ignored.

## Repository policy

This repository is the canonical location for Rival implementation work, tests, provenance, progress, and stable commits. Meaningful completed work should be committed and pushed here rather than existing only in a local Codex workspace.
