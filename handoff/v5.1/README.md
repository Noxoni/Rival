# Rival Handoff v5.1 — Training Activation

This handoff activates the RLGym/RocketSim training pivot after the clean completion of Milestone 04 / v4.1.

## Completed boundary

Milestone 04 is complete at:

`80f4a24e60c9c9613322b1f46612a30ebf5b2bb4`

The natural low-resource-aerial treatment was rejected. Normal Rival remains on the frozen Wisp-equivalent policy with both experimental adjustment modes disabled by default. The 16-match v4.1 baseline/treatment evidence remains historical benchmark data and must not be reinterpreted as a successful gameplay change.

## Active direction

Do **not** start another Wisp heuristic patch.

Read the complete `handoff/v5.0/` package. Milestone 05 builds Rival's RLGym/RocketSim training foundation so future improvement comes primarily from trainable policies rather than accumulating runtime interventions.

Training target:

`Wisp teacher -> RLGym/RocketSim natural 1v1 -> trainable Rival student -> RLBot deployment/benchmark`

The student action space must preserve the exact Wisp 90-action prefix while extending mechanical control capability, and the training pipeline must be reproducible, checkpointed, resumable, and ready for later serious mechanics/self-play training.

## Start here

`handoff/v5.1/CODEX_START_PROMPT.md`

v5.1 is an activation overlay; the architecture and implementation requirements remain in `handoff/v5.0/`.
