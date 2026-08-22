# Rival

Rival is a high-end offline Rocket League 1v1 bot project for **RLBot v5**. The current deployed gameplay remains the verified Wisp v2-75B-derived baseline while the project pivots to training a new Rival policy with **RLGym + RocketSim**.

Rival is for offline RLBot play only. It must not be used to cheat or otherwise break Rocket League's terms of service.

## Current verified deployment baseline

- RLBot display name: `Rival Dev`
- Default agent id: `noxoni/rival/dev-v1`
- Runtime config: `bot/rival.bot.toml`
- Frozen Wisp-equivalent gameplay baseline: `4f2b21c00e2fcb7108ab1006fd950b066fbd0484`
- Baseline models: unchanged Wisp v2-75B `POLICY.lt` and `SHARED_HEAD.lt`
- Challenge calibration default: `off`
- Natural adjustment default: `off`
- Completed Milestone 04 / v4.1 boundary: `80f4a24e60c9c9613322b1f46612a30ebf5b2bb4`

## Milestone 04 result

v4.1 completed a natural-play optimization cycle using 16 full five-minute Soccar matches against installed Nexto and Wisp v2-75B at approximately 5x effective simulation speed.

The tested `m04p1-low-resource-aerial-v1` intervention was **rejected**. It changed Wisp's selected action 17 times but produced unfavorable targeted and aggregate results, so normal Rival remains on the exact frozen Wisp policy. The rejected implementation remains available behind an explicit switch for reproducibility only.

The v4.1 evidence now serves as a real deployment benchmark for future trained Rival checkpoints rather than as a reason to add another heuristic patch.

See `docs/MILESTONE_04_RESULTS.md`, `evidence/results/v4.1/milestone_04_decision.json`, and earlier milestone reports under `docs/` and `evidence/results/`.

## Current direction — train Rival

The project is no longer centered on accumulating tactical re-ranking rules around Wisp.

Primary architecture:

`Wisp teacher -> RLGym/RocketSim natural 1v1 -> trainable Rival student -> RLBot deployment/benchmark`

RLGym/RocketSim becomes the training environment. RLBot/Rocket League remains the deployment, telemetry, and benchmark environment.

The training architecture is intended to support learning advanced mechanics and recovery behavior through reinforcement learning rather than hard-coded macros, including:

- flip resets and useful reset follow-ups;
- ceiling resets/control;
- controlled aerial possession and opponent outplays;
- musty/breezi/Meeri-pop-like sequences when useful;
- wavedash/zap-dash/wall-dash-style recovery and acceleration;
- sidewall recovery/skimming;
- use of flips to preserve aerial momentum and conserve boost;
- rapid defensive recovery after missed offense or possession loss.

Winning and useful 1v1 outcomes remain the primary objective. Mechanics are valuable only when they improve those outcomes.

## Current Codex handoff — v5.1

Start here:

`handoff/v5.1/CODEX_START_PROMPT.md`

v5.1 activates the complete architecture/specification in `handoff/v5.0/` after the clean v4.1 completion boundary.

Milestone 05 builds the first functional training foundation:

- isolated RLGym/RocketSim training environment;
- natural headless 1v1 self-play;
- exact Wisp 90-action prefix plus an expanded mechanics-capable action space;
- Wisp teacher bootstrap through verified reconstruction or behavior distillation;
- outcome-dominant reward system with modest mechanics/recovery shaping;
- headless rollout throughput benchmarking;
- bounded PPO smoke with checkpoint save/reload/resume;
- inference/export seam back to RLBot.

Milestone 05 does **not** launch the first long mechanics training campaign. It proves the trainer is correct and resumable so the next milestone can spend compute on actual learning rather than infrastructure debugging.

Previous handoffs remain under `handoff/` as recoverable project history.

## Distribution

For another Windows PC, do not distribute only `bot/`: the development TOML depends on the repository-local `.venv`. Build the self-contained Windows x64 ZIP with `scripts/build_windows_release.ps1`; the complete procedure and verification contract are in `docs/BUILD_WINDOWS_RELEASE.md`.

## Local RLBot references

Installed RLBot v5 bots used as read-only benchmark/teacher references are under:

`C:\Users\patri\AppData\Local\RLBot5\bots`

The primary local references are Wisp v2-75B and Nexto. Exact local and upstream provenance is recorded in `docs/SOURCE_PROVENANCE.md`; machine-local snapshots remain Git-ignored.

## Repository policy

This repository is the canonical location for Rival implementation work, tests, provenance, evidence tooling, training infrastructure, progress, and stable commits. Meaningful completed work should be committed and pushed here rather than existing only in a local Codex workspace.
