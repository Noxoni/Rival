# Rival Handoff v7.0 — Sim-to-RLBot Transfer Isolation

Milestone 06 stopped correctly at 20M because the RocketSim result and the real RLBot result strongly disagreed. This handoff does **not** authorize more PPO training. It isolates the transfer failure first.

## Starting boundary

- Completed M06 rollback: `652395a9f512ce835830bfc5bc3a7cb078f6105e`
- Production remains frozen Wisp.
- Rejected 20M checkpoint remains local/opt-in evidence only.

## Core finding that motivates v7

The M06 candidate runtime forces candidate policies to run at tick skip 4. No zero-step reconstructed-Wisp candidate was ever evaluated in RLBot at both tick 8 and tick 4. Therefore the 0-8 RLBot failure cannot yet be attributed cleanly to PPO learning.

Milestone 07 runs a bounded transfer-isolation matrix and a training-vs-live observation-domain audit. It must distinguish, as far as the evidence allows, among:

1. four-tick cadence/action-delay deployment gap;
2. learned drift in legacy Wisp logits;
3. RocketSim/RLGym observation-domain mismatch;
4. deployment/export/action-path defect or an interaction among the above.

## Central rule

Do not resume the 20M checkpoint and do not train another serious candidate until the transfer boundary is understood well enough to choose a specific corrective architecture.

Read `TRANSFER_DIAGNOSTIC_MATRIX.md`, `OBSERVATION_DOMAIN_AUDIT.md`, and then execute `CODEX_START_PROMPT.md`.
