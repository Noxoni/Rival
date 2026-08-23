# Rival handoff v8.0 — dual-rate transfer-safe architecture

Milestone 08 is the architecture correction authorized by the completed Milestone 07 transfer diagnosis.

Starting boundary: `10c41f708d6e8145bf719f8f322041e7753f6c3f`.

Do **not** resume the rejected Milestone 06 20M actor as the training base. Start from the verified zero-step trainable Wisp reconstruction and preserve the frozen production Wisp artifacts.

## Goal

Build and validate a transfer-safe dual-rate Rival agent:

- an exact/frozen Wisp strategic branch operating at its native 8-tick cadence;
- a separate trainable 4-tick mechanics/recovery branch that can either PASS or temporarily override with one of the 68 appended mechanics-capable actions;
- a versioned Wisp-compatible observation contract whose training and live implementations agree closely enough that the frozen teacher sees materially the same state;
- explicit temporal action schedulers matching the real live 8-tick and 4-tick execution windows;
- bounded mechanics-head PPO only after zero-step transfer gates pass.

With the mechanics branch disabled, Rival must reduce to the verified zero-step tick-8 behavior. M08 must not alter production defaults or promote a candidate.

## Files

- `CODEX_START_PROMPT.md` — execution authority.
- `MILESTONE_08_SPEC.md` — complete acceptance contract.
- `DUAL_RATE_ARCHITECTURE.md` — branch/gate/action scheduling design.
- `OBSERVATION_CONTRACT.md` — live/training observation parity requirements.
- `TRAINING_AND_EVALUATION.md` — bounded mechanics-head training and RLBot gates.
- `VERSION.md` — boundary/version summary.
- `PACKAGE_MANIFEST.json` — package index.

The optional RLViser spectator from Milestone 07 remains supported and should be adapted to the dual-rate candidate if low-cost, without entering the training hot path.