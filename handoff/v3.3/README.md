# Rival Handoff v3.3 — Resume Paused Milestone 03 Safely

Milestone 03 was paused after only two local strategy files had been started. No tests, probes, natural matches, commits, or pushes from that partial implementation were running or completed.

This overlay exists so Codex can resume without losing that work and without missing the newer accelerated-testing configuration added after the pause.

## Start here

Execute:

`handoff/v3.3/CODEX_START_PROMPT.md`

First read:

- `handoff/v3.3/RESUME_STATE.md`
- `handoff/v3.2/CODEX_START_PROMPT.md`
- `handoff/v3.2/CONFIG_OPTIMIZATIONS.md`
- every file under `handoff/v3.0/`

## Paused local implementation

Preserve and review:

- `bot/strategy/__init__.py`
- `bot/strategy/challenge_commitment.py`

Do not reset them, recreate them from scratch, or assume they are complete.

The user's `bot.7z` remains unrelated and untouched.

## Testing after resume

Use full five-minute games for broad natural validation, but target 5.0x game speed after validating that accelerated execution is trustworthy enough for comparison.

Keep goal replays skipped, automated replay saving disabled, debug rendering hard-disabled, and the performance overlay disabled. Attempt two truly isolated concurrent match lanes once; if Rocket League/RLBot cannot support that cleanly, fall back immediately to sequential 5x execution.

No Milestone 03 natural-match budget had been consumed before the pause.
