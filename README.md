# Rival

Rival is a high-end offline Rocket League 1v1 bot project for **RLBot v5**. The deployed bot remains the verified frozen Wisp v2-75B baseline while the project trains and validates a new Rival policy through **RLGym + RocketSim**.

Rival is for offline RLBot play only. It must not be used to cheat or otherwise break Rocket League's terms of service.

## Current deployment baseline

- RLBot display name: `Rival Dev`
- Default agent id: `noxoni/rival/dev-v1`
- Frozen Wisp-equivalent gameplay baseline: `4f2b21c00e2fcb7108ab1006fd950b066fbd0484`
- Wisp `POLICY.lt` / `SHARED_HEAD.lt` unchanged
- Challenge calibration: `off`
- Natural adjustment: `off`
- Completed v4.1 natural benchmark: `80f4a24e60c9c9613322b1f46612a30ebf5b2bb4`

The runtime gameplay-adjustment experiments were rejected and remain disabled. Production was not replaced by any trained candidate.

## Milestone 05 — training foundation complete

Completed boundary:

`4c9aa6f596b3231856107b3a1e59d9a7c4f663db`

Milestone 05 established the isolated RLGym/RocketSim training stack under `training/`: natural headless 1v1, exact Wisp teacher reconstruction, 158-action mechanics-capable student, CUDA PPO training/checkpointing, measured multiprocess rollout throughput, and an export seam back to RLBot.

See `docs/MILESTONE_05_RESULTS.md` and `training/` for exact evidence and reproduction commands.

## Milestone 06 — serious training campaign stopped at 20M

Completed rollback boundary:

`652395a9f512ce835830bfc5bc3a7cb078f6105e`

M06 reached 20,000,016 agent-steps. RocketSim/headless evaluation against frozen Wisp improved from 42–58 preflight to 59–41 at 20M, but the required RLBot v5 boundary battery regressed severely: the trained candidate went 0–8 against installed Nexto/Wisp with a 27–56 goal line. Stage C/D were correctly stopped and production remained frozen Wisp.

See `docs/MILESTONE_06_RESULTS.md` and `training/results/milestone06/`.

## Milestone 07 — transfer diagnosis complete

Completed boundary:

`10c41f708d6e8145bf719f8f322041e7753f6c3f`

M07 isolated the failure instead of resuming training.

Key findings:

- the same zero-step reconstructed Wisp moved from 2–2 / +4 at tick 8 to 0–4 / −11 at tick 4 before any learning, proving a primary strategic cadence mismatch;
- the rejected 20M actor drifted materially within legacy actions 0–89, reaching only 68.0% masked top-1 agreement with frozen Wisp on held live observations;
- the old training-style 432 observation changed frozen-Wisp top-1 on more than half of held live states, with ETA and touch/handbrake/player-state semantics as the dominant mismatch;
- spatial action parsing/mirroring was exact;
- the generic RocketSim legacy8 delay did not reproduce production Wisp's real eight-tick temporal schedule;
- short-horizon RocketSim physics was close initially but accumulated secondary orientation divergence.

The optional RLViser spectator was also added as a separate process so selected checkpoints can be watched without entering the training hot path.

See `docs/MILESTONE_07_RESULTS.md` and `training/results/milestone07/`.

## Current Codex handoff — v8.0

Start here:

`handoff/v8.0/CODEX_START_PROMPT.md`

Milestone 08 implements the corrective architecture supported by M07 rather than resuming the rejected M06 actor.

Architecture:

`frozen 8-tick Wisp strategic branch + separate 4-tick PASS-or-mechanics branch`

The strategic branch keeps the exact zero-step Wisp policy and legacy actions 0–89. The mechanics/recovery branch is separately trainable and initially chooses only PASS or appended actions 90–157. With mechanics disabled or forced PASS, the complete agent must reduce to the verified tick-8 strategic path.

Before PPO, M08 must repair the live/training Wisp observation contract, especially ETA and touch/handbrake/player-state semantics, and reproduce production Wisp's true eight-tick temporal execution in RocketSim. Frozen-Wisp policy agreement between live and training representations has a 97% hard floor and 99% target before learning is allowed.

If those gates pass, M08 authorizes at most 5M agent-steps of mechanics-head-only PPO with RLBot transfer checks at bounded checkpoints. Strategic Wisp weights remain frozen throughout. Production promotion is not authorized in M08.

Previous handoffs remain under `handoff/` as recoverable project history.

## Distribution

For another Windows PC, do not distribute only `bot/`: the development TOML depends on the repository-local `.venv`. Build the self-contained Windows x64 ZIP with `scripts/build_windows_release.ps1`; the complete procedure and verification contract are in `docs/BUILD_WINDOWS_RELEASE.md`.

## Local RLBot references

Installed RLBot v5 bots used as read-only benchmark/teacher references are under:

`C:\Users\patri\AppData\Local\RLBot5\bots`

The primary local references are Wisp v2-75B and Nexto. Exact local and upstream provenance is recorded in `docs/SOURCE_PROVENANCE.md`; machine-local snapshots remain Git-ignored.

## Repository policy

This repository is the canonical location for Rival implementation work, tests, provenance, evidence tooling, training infrastructure, progress, and stable commits. Meaningful completed work should be committed and pushed here rather than existing only in a local Codex workspace.