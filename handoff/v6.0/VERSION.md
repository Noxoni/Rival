# Rival Codex Handoff v6.0

**Purpose:** First serious Rival training campaign.

**Required completed boundary:** `4c9aa6f596b3231856107b3a1e59d9a7c4f663db` — Milestone 05 RLGym/RocketSim training foundation.

**Deployment baseline remains:** frozen Wisp-equivalent Rival with both rejected runtime interventions off.

**Training foundation:** RLGym v2 + RocketSim + rlgym-tools + commit-pinned rlgym-ppo, 24 measured workers, 158-action RivalExpandedActionV1, Wisp-compatible 432 observation path, directly reconstructed Wisp student, mechanics4 cadence, resumable PPO/checkpoint/export seam.

## v6.0 objective

Spend compute on actual learning rather than more trainer infrastructure.

Run a staged, resumable natural 1v1 campaign with a ceiling of **100,000,000 agent-steps**, opening the 68 appended mechanics actions conservatively, monitoring Wisp retention and natural mechanics/recovery metrics, and evaluating checkpoints against frozen Wisp plus the RLBot Nexto/Wisp benchmark.

The campaign may stop before the ceiling if a checkpoint earns promotion or a health gate rejects continued training.

No checkpoint replaces production Rival unless the explicit promotion gate passes.
