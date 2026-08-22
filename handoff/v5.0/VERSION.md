# Handoff Version v5.0

## Name

**RLGym Training Foundation**

## Purpose

Replace the project's main improvement loop of small RLBot policy patches with a trainable Rival policy developed in headless RLGym/RocketSim and deployed back into RLBot for real-game evaluation.

## Starting lineage

The v5.0 package was prepared after the v4.1 natural-play run had pushed at least through:

`3c15ff55ba6005777c3ab6457dc3d14e8453a966` — `Record v4.1 natural baseline analysis`

Codex must fetch `origin/main` at execution time and preserve any legitimate newer v4.1 commits/results before beginning Milestone 05.

## Frozen teacher baseline

- Wisp-equivalent gameplay baseline: `4f2b21c00e2fcb7108ab1006fd950b066fbd0484`
- Wisp policy artifact SHA-256: `1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7`
- Wisp shared-head artifact SHA-256: `3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42`
- Current Wisp action count: 90
- Current Wisp/Rival decision cadence: 8 physics ticks

These artifacts remain read-only teacher inputs. Milestone 05 must not rewrite them.

## Primary implementation choice

Use the Python RLGym v2 ecosystem first:

- `rlgym` / `rlgym-rocket-league[sim]`
- `rlgym-tools`
- `rlgym-ppo`
- PyTorch/CUDA where available

Reason: it integrates directly with the existing Python/TorchScript teacher and minimizes porting work. Rust `rlgymppo_rs` remains a later optimization candidate, not Milestone 05 scope.

## What v5.0 must produce

A committed training foundation, a verified headless 1v1 environment, a mechanics-capable action space, Wisp bootstrap feasibility/results, reward instrumentation, throughput measurements, a resumable training/checkpoint path, and a short PPO smoke run.

It does **not** need to finish training a superior final Rival model in one Codex run.
