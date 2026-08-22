# Rival Handoff v4.0 — Deterministic Paired Exposure

Milestone 03 is complete and rejected. The challenge-calibration feature remains disabled by default.

This handoff begins the next implementation milestone: make controlled challenge cases reproducible enough that `off` and `observe` produce the same policy inputs, Wisp action choices, and release-sensitive exposure before any treatment is allowed to modify gameplay.

## Start here

Codex must execute:

`handoff/v4.0/CODEX_START_PROMPT.md`

and use:

- `handoff/v4.0/DETERMINISTIC_PAIRING_SPEC.md`
- `handoff/v4.0/SPEED_POLICY.md`

as the execution/acceptance contract.

## Core discovery from Milestone 03

Two supposedly equivalent controlled runs diverged even with **zero applied interventions**. That makes the old A/B result non-causal.

A concrete nondeterminism source exists in the inherited Wisp observation builder: the real opponent is shuffled among padded opponent observation slots using Python's process-global `random.shuffle()` every policy decision. The observation also includes the previous controller action, while Rival carries additional runtime history between state-set controlled cases.

v4 therefore fixes the experiment before revisiting the gameplay rule.

## Acceleration policy

The direct `DesiredMatchInfo.game_speed` path achieved approximately 5x effective simulated-time acceleration. The packet's reported `game_speed` field remained stale at `1.0`; v4 treats effective simulated-time progression and paired reproducibility as the meaningful validation signals instead of requiring that stale field to echo the requested multiplier.

Full five-minute natural matches remain the eventual acceptance format. No natural acceptance-match budget is spent in v4 unless a later explicitly permitted treatment stage is reached.

Previous handoffs remain under `handoff/` as recoverable project history.