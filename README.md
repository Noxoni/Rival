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

The runtime gameplay-adjustment experiments were rejected and remain disabled. Production was not replaced by any trained M06 candidate.

## Milestone 05 — training foundation complete

Completed boundary:

`4c9aa6f596b3231856107b3a1e59d9a7c4f663db`

Milestone 05 established the isolated RLGym/RocketSim training stack under `training/`: natural headless 1v1, exact Wisp teacher reconstruction, 158-action mechanics-capable student, CUDA PPO training/checkpointing, measured multiprocess rollout throughput, and an export seam back to RLBot.

See `docs/MILESTONE_05_RESULTS.md` and `training/` for exact evidence and reproduction commands.

## Milestone 06 — serious training campaign stopped at 20M

Completed rollback boundary:

`652395a9f512ce835830bfc5bc3a7cb078f6105e`

M06 reached 20,000,016 agent-steps. RocketSim/headless evaluation against frozen Wisp improved from 42–58 preflight to 59–41 at 20M, but the required RLBot v5 boundary battery regressed severely: the trained candidate went 0–8 against installed Nexto/Wisp with a 27–56 goal line. Stage C/D were correctly stopped and production remained frozen Wisp.

The failure was not explained by appended-action overuse: the RLBot battery selected zero actions from indices 90–157. M06 evidence leaves three major unresolved causes: legacy-policy drift, the forced four-tick candidate deployment cadence, and RocketSim/RLGym-to-RLBot observation/transition mismatch.

See `docs/MILESTONE_06_RESULTS.md` and `training/results/milestone06/`.

## Current Codex handoff — v7.0

Start here:

`handoff/v7.0/CODEX_START_PROMPT.md`

Milestone 07 does **not** authorize more serious PPO training. It isolates the transfer failure first.

The diagnostic uses the RLGym-style decomposition:

`state s -> observation O(s) -> policy pi(o) -> action function I -> action a -> transition T(s'|s,a)`

It separately audits:

- zero-step reconstructed Wisp in RLBot at tick 8 versus tick 4;
- the rejected 20M actor at tick 8 versus tick 4 with appended actions hard-masked;
- same-live-observation policy/logit parity and legacy-action drift;
- feature-group differences between training and live 432-value observations;
- action parser, mirroring, repeat and delay semantics;
- bounded short-horizon RocketSim versus RLBot physical divergence on natural states.

The goal is a ranked causal diagnosis and a concrete corrective architecture for the next training milestone. Do not resume the 20M checkpoint until the transfer seam is understood.

Previous handoffs remain under `handoff/` as recoverable project history.

## Distribution

For another Windows PC, do not distribute only `bot/`: the development TOML depends on the repository-local `.venv`. Build the self-contained Windows x64 ZIP with `scripts/build_windows_release.ps1`; the complete procedure and verification contract are in `docs/BUILD_WINDOWS_RELEASE.md`.

## Local RLBot references

Installed RLBot v5 bots used as read-only benchmark/teacher references are under:

`C:\Users\patri\AppData\Local\RLBot5\bots`

The primary local references are Wisp v2-75B and Nexto. Exact local and upstream provenance is recorded in `docs/SOURCE_PROVENANCE.md`; machine-local snapshots remain Git-ignored.

## Repository policy

This repository is the canonical location for Rival implementation work, tests, provenance, evidence tooling, training infrastructure, progress, and stable commits. Meaningful completed work should be committed and pushed here rather than existing only in a local Codex workspace.
