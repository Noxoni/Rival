# Rival

Rival is a high-end offline Rocket League 1v1 bot project for **RLBot v5**. The deployed bot remains the verified frozen Wisp v2-75B baseline while a new Rival policy is trained from scratch through **RLGym + RocketSim**.

Rival is for offline RLBot play only. It must not be used to cheat or otherwise break Rocket League's terms of service.

## Current deployment baseline

- RLBot display name: `Rival Dev`
- Default agent id: `noxoni/rival/dev-v1`
- Production policy: frozen Wisp v2-75B
- Production tick skip: 8
- Wisp `POLICY.lt` / `SHARED_HEAD.lt`: unchanged
- Scratch candidate promotion: **not authorized**

Previous runtime adjustment and Wisp-overlay training experiments were rejected or retained only as research history. Production has not been replaced by a trained candidate.

## Milestone 09 — scratch foundation complete

Completed boundary:

`824e328f6bbf4fe9a47b8e54706b5fcf645fd409`

M09 replaced the Wisp-derived training architecture with a true scratch-policy foundation and passed validation Gates 0–14.

The proven scratch stack is:

- `RivalPolicyV1`, no Wisp actor/trunk parameters;
- `RivalObsV1`, one shared train/deploy canonical observation, 714 floats;
- `RivalActionV1`, five continuous analog controller axes plus a joint eight-way jump/boost/handbrake categorical;
- one policy decision per Rocket League physics tick at native 120 Hz;
- no action lookup table, RepeatAction, state-dependent action mask, or mechanics macros;
- independent actor/critic networks;
- cadence-safe outcome-dominant reward;
- majority-natural 70/10/8/8/4 reset curriculum;
- native-120-Hz RLBot deployment/export path and isolated RLViser spectator.

The bounded M09 pilot reached 1,680,214 cumulative agent-steps / 1.9446921296 simulated game-hours. It showed measurable learning (including first deterministic touch behavior and improved recovery/contact metrics) but intentionally stopped before serious training.

See `docs/MILESTONE_09_RESULTS.md` and `training/results/milestone09/`.

## Current Codex handoff — v10.0

Start here:

`handoff/v10.0/CODEX_START_PROMPT.md`

Milestone 10 is the **first serious scratch training campaign**. It resumes the exact final M09 checkpoint instead of resetting or redesigning the architecture.

Authorized M10 budget:

- 100 additional simulated game-hours;
- 86,400,000 additional agent-steps;
- nominal cumulative target 88,080,214 agent-steps;
- immutable evaluation boundaries at approximately +5, +10, +25, +50, and +100 simulated hours.

M10 intentionally keeps the M09 policy, observation, action, reward, PPO hyperparameters, native 120-Hz cadence, 56-worker selection, and 70/10/8/8/4 curriculum unchanged. The primary variable is experience volume.

Production promotion is not authorized.

## Earlier milestone history

- M05 (`4c9aa6f596b3231856107b3a1e59d9a7c4f663db`): RLGym/RocketSim training foundation and exact Wisp reconstruction.
- M06 (`652395a9f512ce835830bfc5bc3a7cb078f6105e`): monolithic 4-tick Wisp-derived campaign stopped at 20M after severe RLBot transfer regression.
- M07 (`10c41f708d6e8145bf719f8f322041e7753f6c3f`): isolated cadence, observation-domain, and legacy-policy-drift causes.
- M08 (`0b8f31351930e0e756c59349360b9c0b8dbda4c6`): frozen-Wisp + mechanics-overlay architecture transferred correctly but learned to prefer PASS; final deterministic mechanics usage remained zero.
- M09 (`824e328f6bbf4fe9a47b8e54706b5fcf645fd409`): completed and validated the native-120-Hz scratch architecture.

Previous handoffs remain under `handoff/` as recoverable project history.

## Distribution

For another Windows PC, do not distribute only `bot/`: the development TOML depends on the repository-local `.venv`. Build the self-contained Windows x64 ZIP with `scripts/build_windows_release.ps1`; the complete procedure and verification contract are in `docs/BUILD_WINDOWS_RELEASE.md`.

## Local RLBot references

Installed RLBot v5 bots used as read-only benchmark references are under:

`C:\Users\patri\AppData\Local\RLBot5\bots`

The primary local references are Wisp v2-75B and Nexto. Exact local and upstream provenance is recorded in `docs/SOURCE_PROVENANCE.md`; machine-local snapshots remain Git-ignored.

## Repository policy

This repository is the canonical location for Rival implementation work, tests, provenance, evidence tooling, training infrastructure, progress, and stable commits. Meaningful completed work should be committed and pushed here rather than existing only in a local Codex workspace.
