# Milestone 05 Results — RLGym/RocketSim Training Foundation

- Date: 2026-08-22
- Activation authority: `handoff/v5.1/CODEX_START_PROMPT.md`
- Inherited design: `handoff/v5.0/`
- Starting remote boundary: `d11d1652cba113c61b85f8a0329238dafce44b5b`
- Completed v4.1 ancestor preserved: `80f4a24e60c9c9613322b1f46612a30ebf5b2bb4`

## Decision

**Milestone 05 passed.** Rival now has a reproducible and resumable natural 1v1 training foundation built on RLGym v2, RocketSim, RLGym Tools, CUDA PyTorch, and a commit-pinned `rlgym-ppo`.

This result proves the training infrastructure. It does not claim that the short smoke checkpoint is stronger than Wisp, and it does not promote that checkpoint into the live RLBot bot. The production Rival runtime and both rejected Milestone 04 intervention modes are unchanged and remain off by default.

## Isolated runtime

Training uses `training/.venv`; the production `.venv` was not modified.

Resolved core versions:

| Component | Version |
|---|---:|
| Python | 3.12.13 |
| PyTorch | 2.13.0+cu130 |
| RLGym | 2.0.1 |
| rlgym-api | 2.0.0 |
| rlgym-rocket-league | 2.0.1 |
| rlgym-tools | 2.6.5 |
| RocketSim | 2.2.1 |
| rlgym-ppo | 1.3.13 at commit `4ffd2e924198bf4b2d59f4bf280b29919d7c07ea` |
| NumPy | 1.26.4 |

CUDA was exercised on an NVIDIA GeForce RTX 5090, compute capability 12.0. A CUDA matrix multiplication completed with finite output.

The published RocketSim Windows wheel has a known metadata defect: its internal `WHEEL` tag says CPython 3.11 even though the generic `RocketSim.pyd` loads and runs under this CPython 3.12 environment. As a result, `python -m pip check` reports `rocketsim 2.2.1 is not supported on this platform`. This was not hidden or patched locally. Direct import, arena construction, cadence smokes, a 5,000-decision stress run, all multiprocess worker candidates, and the PPO run passed. The next clean rebuild should use an officially CPython-3.12-tagged wheel when one exists. See `training/dependency_manifest.json`.

## Natural headless environment

`RivalNatural1v1RocketSimV1` provides:

- one blue and one orange Octane;
- ordinary kickoff resets and natural play;
- RocketSim's Soccar transition engine;
- no renderer and no RocketLeague.exe process;
- RLBot-like one-tick input delay;
- goal termination;
- 30-second no-touch and 300-second episode truncation;
- 120 Hz physics with selectable `legacy8` and `mechanics4` action cadence.

Both cadence modes reset and step with two finite `(432,)` observations and finite rewards. A separate 5,000-decision `mechanics4` stress ran 20,000 physics ticks per environment, produced 10,000 agent-steps, crossed six episodes, and retained finite observations, total reward, and all individual reward components.

Evidence: `training/results/environment_verification.json` and `training/results/environment_stress.json`.

## Action space

`RivalExpandedActionV1` has 158 unique controller rows:

- indices `0..89`: exact Wisp rows, order, and controller semantics;
- indices `90..157`: 68 unique rows appended from `AdvancedLookupTableAction(torque_subdivisions=3, flip_bins=16, include_stalls=True)`;
- controller order: throttle, steer, pitch, yaw, roll, jump, boost, handbrake;
- state-dependent steer/yaw/roll X mirroring matches the live Wisp parser;
- actions repeat for 8 ticks in `legacy8` or 4 ticks in `mechanics4`.

Canonical row-major little-endian float32 hashes:

| Table | Rows | SHA-256 |
|---|---:|---|
| Wisp prefix | 90 | `86baa15c48c42c497f3ea0fe62efeb49e4a8241cb3191957822e453cd2d0b655` |
| Expanded table | 158 | `38ed338273ae09736d81d3e7fb69c45d91397e45d50f1ae97101e3737c0ecd20` |

The prefix proof imports the actual production `bot/action_parser.py`, constructs all 90 `DefaultAction` rows, and compares them directly. Shape, values, and hash are exact with maximum absolute error 0.

Evidence: `training/results/action_prefix_proof.json` and `training/results/action_table_metadata.json`.

## Observation path

`WispCompatible432RLGymV1` preserves the teacher's 432-value interface, normalization, broad feature ordering, six 51-value player slots, previous-action semantics, boost-pad semantics, and prediction horizons of 22, 66, 198, and 594 physics ticks.

The implementation makes its non-bit-parity adaptations explicit:

- RocketSim `BallPredictor` supplies predictions in place of RLBot flatbuffer slices;
- a bounded box-surface landing normal replaces the live arena-SDF call;
- a bounded kinematic ETA replaces the live process-global cached ETA;
- episodic training uses score differential zero unless a later curriculum wrapper supplies a score;
- the training action parser supplies previous inputs for both self-play agents.

The result is teacher-interface compatible, not falsely described as bit-identical to live RLBot observations.

## Wisp bootstrap

The ordered bootstrap stopped successfully at Path A. Direct reconstruction was not opaque:

- shared head: `432 -> 1024 -> 1024`, with LayerNorm and ReLU;
- policy: `1024 -> 1024 -> 512 -> 512 -> 128 -> 90`, with LayerNorm and ReLU on hidden layers;
- expanded student parameters: 3,424,542;
- original 90 output rows copied exactly;
- appended rows initialized with zero weights and bias `-12.0`.

The frozen teacher hashes matched before and after:

- `POLICY.lt`: `1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7`
- `SHARED_HEAD.lt`: `3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42`

On 4,096 seeded randomized 432-value observations, the reconstructed student's first 90 logits had maximum and mean absolute error 0 in the final gate. Teacher/student top-1 agreement was 100%, and appended-action top-1 selection before PPO was 0%.

Behavior distillation was correctly not run because direct reconstruction passed. No dataset was generated. The ignored bootstrap checkpoint is recorded at `training/artifacts/bootstrap/wisp_student_expanded_v1.pt` with its SHA-256 and size in the committed report.

Evidence: `training/results/bootstrap_report.json`.

## Reward v1

`RivalOutcomeRewardV1` independently emits and accumulates:

1. signed goal/concede outcome at +10/-10;
2. modest useful-touch/possession shaping;
3. small signed offensive progress;
4. small boost-efficiency accounting;
5. small stateful recovery value.

No named-mechanic reward is enabled. The random stress evidence contains each component's finite aggregate, which makes reward farming visible rather than hiding everything in a single scalar.

## Measured rollout throughput

The required worker candidates were benchmarked with the Wisp-derived 158-action policy, 4-tick cadence, central CUDA inference, and natural headless 1v1 environments.

| Workers | Agent-steps/s | Aggregate simulated game-seconds/s | CPU sample | Peak allocated GPU MiB | Result |
|---:|---:|---:|---:|---:|---|
| 8 | 5,330.35 | 88.84 | 42.4% | 45.17 | stable |
| 12 | 7,106.63 | 118.44 | 48.6% | 45.25 | stable |
| 16 | 9,724.20 | 162.07 | 55.6% | 45.33 | stable |
| 24 | 14,483.56 | 241.39 | 81.0% | 45.47 | stable |

No candidate errored or stalled. The measured selection rule chose **24 workers**. That value is evidence for this 8-core/16-thread Ryzen 7 9800X3D machine, not a universal default for every PC.

Evidence: `training/results/throughput_report.json`.

## PPO save/reload/resume smoke

The final bounded run used the selected 24 workers, `mechanics4`, the reconstructed Wisp student, 2,048 target agent-steps per iteration, batch size 1,024, minibatches of 256, two PPO epochs, and CUDA.

It used the actual package stack:

- rollout: `rlgym_ppo.batched_agents.BatchedAgentManager` through `Learner`;
- experience/GAE: `ExperienceBuffer` and `Learner.add_new_experience`;
- updates: `rlgym_ppo.ppo.PPOLearner`.

Three iterations completed: two before checkpoint reload and one after a fresh learner reload. All reported metrics were finite. Policy update magnitudes were approximately `0.04433`, `0.01825`, and `0.01713`; the nonzero third value proves resumed learning rather than load-only inference. Fresh reload produced exactly identical logits, restored nonempty optimizer state, and restored the recorded model-update counter.

The first development attempt used a 1,024-step collection target. Because trajectory consolidation occasionally yielded fewer than one full package batch, later calls could report zero-magnitude work. That attempt was rejected rather than counted as success. The committed harness now uses 2,048 target steps and hard-fails any iteration whose policy update magnitude is zero.

Across 6,114 sampled smoke actions, appended actions were selected 0 times. This is expected from the conservative `-12` bootstrap bias and is not presented as mechanics learning. The complete 158-bin distribution is committed. PPO gradients nevertheless moved the appended head away from its zero-weight initialization, proving it participates in optimization.

The ignored full checkpoint contains policy, critic, both optimizer states, and trainer state under `training/checkpoints/milestone05_smoke`. The committed report records every file's SHA-256 and size. A portable actor checkpoint is recorded under `training/artifacts/ppo/`.

Evidence: `training/results/ppo_smoke_report.json`.

## Deployment seam

The post-smoke actor checkpoint loads independently for a single `(432,)` observation, emits 158 finite logits, selects a controller row, and applies exact X-mirror controller semantics. A TorchScript export reloads with maximum absolute logit error 0 in the smoke.

This seam does not replace the production policy. `production_runtime_replaced` is explicitly false.

Evidence: `training/results/deployment_smoke.json`.

## Verification summary

Final checks completed locally:

- existing Rival suite: **70 passed**;
- isolated training suite: **13 passed**;
- direct production Wisp action-prefix proof: passed;
- frozen Wisp hash verification before and after bootstrap: passed;
- both environment cadence smokes: passed;
- 5,000-decision finite environment/reward stress: passed;
- required 8/12/16/24 worker sweep: passed;
- three real PPO iterations with fresh reload and resume: passed;
- actor checkpoint and TorchScript inference/export smoke: passed;
- `ruff check training`: passed;
- `compileall` for training implementation/scripts/tests: passed;
- production policy freeze gate: passed against `4f2b21c00e2fcb7108ab1006fd950b066fbd0484`;
- generated artifact/checkpoint/dataset paths: ignored;
- compact committed evidence contains no machine-local absolute artifact paths.

PyTorch 2.13 emits deprecation warnings for `torch.jit.load/script/save`. They do not invalidate this frozen-teacher compatibility seam, but a later export-format migration should evaluate `torch.export` without changing the frozen input artifacts.

## What remains before serious training

Milestone 06 or an explicit new authority should:

1. choose a bounded-but-material natural self-play training budget and evaluation cadence;
2. retain the frozen Wisp/Nexto deployment benchmark rather than judging checkpoints by reward alone;
3. add natural evaluation for goals, concessions, possession, boost, recovery, and action distribution;
4. verify whether and when to anneal the appended-action bias so richer controls are explored without destroying the Wisp warm start;
5. add minority broad wall/aerial/recovery resets only after natural self-play remains healthy;
6. add frozen/historical opponent diversity later;
7. promote no checkpoint into RLBot until aggregate full-game evidence earns it.

The foundation is ready for that work. The short smoke actor itself is not a gameplay candidate.
