# Rival

Rival is a high-end offline Rocket League 1v1 bot project for **RLBot v5**. The current deployed gameplay remains the verified Wisp v2-75B-derived baseline while a new Rival policy is trained with **RLGym + RocketSim**.

Rival is for offline RLBot play only. It must not be used to cheat or otherwise break Rocket League's terms of service.

## Current deployment baseline

- RLBot display name: `Rival Dev`
- Default agent id: `noxoni/rival/dev-v1`
- Frozen Wisp-equivalent gameplay baseline: `4f2b21c00e2fcb7108ab1006fd950b066fbd0484`
- Wisp `POLICY.lt` / `SHARED_HEAD.lt` unchanged
- Challenge calibration: `off`
- Natural adjustment: `off`
- Completed v4.1 natural benchmark: `80f4a24e60c9c9613322b1f46612a30ebf5b2bb4`

The two runtime gameplay-adjustment experiments were rejected and remain disabled. The v4.1 natural benchmark is now used as a deployment reference for trained Rival checkpoints.

## Milestone 05 — training foundation complete

Completed boundary:

`4c9aa6f596b3231856107b3a1e59d9a7c4f663db`

Milestone 05 established a reproducible isolated training stack under `training/`:

- RLGym v2 + RocketSim + RLGym Tools + commit-pinned `rlgym-ppo`;
- CUDA PyTorch training environment separate from production RLBot;
- natural renderer-free 1v1 self-play;
- 120 Hz simulation with `mechanics4` 4-tick student cadence;
- 432-value Wisp-compatible observation path;
- `RivalExpandedActionV1`: exact Wisp actions 0–89 plus 68 mechanics-capable actions for 158 total;
- directly reconstructed trainable Wisp actor with exact first-90 logit parity;
- resumable PPO/checkpoint/export path;
- measured 24-worker throughput of ~14.5k agent-steps/s on the development machine;
- deployment inference/TorchScript seam back to RLBot.

See `docs/MILESTONE_05_RESULTS.md` and `training/` for exact evidence, versions and reproduction commands.

## Current Codex handoff — v6.0

Start here:

`handoff/v6.0/CODEX_START_PROMPT.md`

Milestone 06 is Rival's **first serious training campaign**. It stops adding tactical patches around Wisp and spends compute on actual learning.

Campaign architecture:

`Wisp warm start -> natural RLGym/RocketSim 1v1 -> staged PPO training -> checkpoint evaluation -> RLBot Nexto/Wisp benchmark -> promotion only if earned`

Key rules:

- campaign ceiling: 100M agent-steps, staged and resumable rather than one opaque run;
- natural 1v1 remains the majority training distribution;
- broad randomized aerial/wall/recovery state families may be a minority curriculum;
- actions 90–157 are opened gradually from the conservative Wisp warm start rather than unsuppressed all at once;
- winning remains dominant while boost efficiency, recovery and mechanics/resource signals stay low-weight and independently logged;
- checkpoints are evaluated against frozen Wisp throughout and against installed Wisp/Nexto at major healthy boundaries;
- the production RLBot policy remains frozen Wisp until a trained checkpoint passes the explicit promotion battery.

The action/training system is intended to allow Rival to learn useful flip/ceiling resets, better aerial possession, momentum-preserving aerial flips, wavedash/zap-dash/wall-dash-like recovery, sidewall recovery, and mechanically creative outplays when those behaviors improve 1v1 outcomes.

Previous handoffs remain under `handoff/` as recoverable project history.

## Distribution

For another Windows PC, do not distribute only `bot/`: the development TOML depends on the repository-local `.venv`. Build the self-contained Windows x64 ZIP with `scripts/build_windows_release.ps1`; the complete procedure and verification contract are in `docs/BUILD_WINDOWS_RELEASE.md`.

## Local RLBot references

Installed RLBot v5 bots used as read-only benchmark/teacher references are under:

`C:\Users\patri\AppData\Local\RLBot5\bots`

The primary local references are Wisp v2-75B and Nexto. Exact local and upstream provenance is recorded in `docs/SOURCE_PROVENANCE.md`; machine-local snapshots remain Git-ignored.

## Repository policy

This repository is the canonical location for Rival implementation work, tests, provenance, evidence tooling, training infrastructure, progress, and stable commits. Meaningful completed work should be committed and pushed here rather than existing only in a local Codex workspace.
