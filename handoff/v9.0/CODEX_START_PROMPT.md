# Codex start prompt — Rival v9 scratch native-control foundation

You are implementing **Rival v9**, a new scratch-trained offline/private RLBot v5 Rocket League 1v1 policy in repository `Noxoni/Rival`.

This is a prospective authority package currently stored on branch `rival-v9-scratch-design` because Milestone 08 was still active when the design was created.

## 0. Reconcile with the completed M08 boundary first

Do **not** reset or force-push `main` to the old v9 design-base commit.

Before changing implementation:

1. fetch `origin` and inspect current `origin/main`;
2. verify M08 has reached a coherent final or explicitly stopped boundary and preserve its final result/evidence/checkpoints;
3. preserve the existing historical stash unless the user explicitly removed it;
4. bring the complete `handoff/v9.0/` package from `origin/rival-v9-scratch-design` onto the current final M08 history by a normal conflict-aware merge/rebase/cherry-pick strategy;
5. read **every file under `handoff/v9.0/`** before implementing;
6. read M07/M08 result documents because their transfer failures motivate the v9 gates;
7. keep production Rival on frozen Wisp throughout this milestone.

If M08 is somehow still actively training when this prompt is given, stop only at the next clean recoverable checkpoint/evaluation boundary, record/push its honest partial/final outcome, then begin v9. Do not throw away already-completed M08 work.

## 1. Architectural authority

The v9 design documents are authoritative unless implementation evidence exposes an internal contradiction that makes them impossible. In that case, resolve the smallest technical issue, document it, and preserve the core goals rather than silently changing the architecture.

Non-negotiable v9 decisions:

- scratch Rival actor; no Wisp parameters in the actor graph;
- Wisp/Nexto are opponents/benchmarks only;
- 120-Hz policy cadence: one actor decision per physics tick;
- no `RepeatAction` in the scratch path;
- no 90/158-row lookup-table action ceiling;
- five continuous native controller axes + joint 8-way jump/boost/handbrake categorical;
- no state-dependent action mask in `RivalActionV1`;
- one canonical train/deploy state/observation implementation;
- rich `RivalObsV1`, not Wisp-compatible 432;
- serious PPO forbidden until observation/action/timing parity gates pass;
- M09/v9 learning pilot ceiling is 2 simulated game-hours, not a large campaign;
- production promotion is not authorized.

## 2. Implement `RivalActionV1`

Follow `RIVAL_ACTION_V1.md` exactly.

Build a native one-tick action parser and custom hybrid PPO distribution:

- tanh-squashed diagonal Gaussian for throttle/steer/pitch/yaw/roll;
- correct transformed-action log probability including tanh Jacobian;
- joint 8-way categorical for jump/boost/handbrake;
- mixed rollout/backprop log probability = analog log-probability sum + categorical log-probability;
- deterministic = tanh(mean) plus categorical argmax;
- store actual physical 8-value controller actions in experience;
- no lookup quantization, no hidden macros, no state mask.

Write exhaustive distribution math/gradient/round-trip tests before using PPO.

## 3. Implement `RivalCanonicalStateV1` and `RivalObsV1`

Follow `RIVAL_OBS_V1.md`.

Create thin adapters:

- RocketSim/RLGym state -> canonical;
- RLBot v5 GamePacket/FieldInfo -> canonical.

After canonicalization there must be **one shared feature implementation**. Do not write a training obs builder and a separately approximated live obs builder.

Generate the schema/index manifest programmatically from the observation definition. Include source/frame/normalization/update/reset metadata and a canonical schema hash.

Implement all logical blocks in the spec, including:

- explicit modern jump/dodge mechanics state;
- self/opponent physics and local geometry;
- match/time/score/touch context;
- ball/goal geometry;
- shared RocketSim ball prediction in both train and live domains;
- all 34 boost-pad entities/timers;
- 8 ticks of physical controller history for both players;
- one-tick motion deltas;
- shared deterministic surface/intercept helpers.

Do not reintroduce Wisp's process-global cached ETA.

## 4. Implement `RivalPolicyV1`

Follow `ARCHITECTURE.md`.

Use schema-driven logical encoders rather than one giant hand-indexed flat MLP:

- core encoder;
- shared boost-pad entity encoder/pooling;
- prediction-horizon encoder;
- controller-history temporal encoder;
- fusion trunk;
- hybrid actor heads.

Use a separate critic with independent parameters. No privileged simulator-only critic in v1.

Do not add recurrence to v1 unless the required implementation becomes impossible without it. The explicit mechanical state/history is the intended v1 temporal context.

## 5. Implement cadence-safe rewards and scratch curriculum

Follow `TRAINING_FOUNDATION.md`.

Critical rules:

- goal/concede outcome dominates;
- dense per-step shaping is potential-based or explicitly multiplied by physical `dt=1/120`;
- do not copy 30-Hz per-step reward magnitude into 120-Hz training;
- no large named-mechanic identity rewards;
- mechanics event detectors are primarily diagnostics;
- natural 1v1 remains the majority reset distribution;
- broad random aerial/wall/recovery/low-resource resets are allowed minority distributions;
- self-play is primary early learning mode;
- introduce strong fixed opponents only after basic interaction metrics exist;
- report experience in simulated game-hours and agent-steps.

Initial physical-time PPO defaults:

- gamma `0.9987444968227265`;
- GAE lambda `0.9872585449014338`;
- rollout target around 200k agent-steps;
- experience buffer around 600k;
- PPO batch around 192k;
- minibatch around 48k;
- 1 PPO epoch initially;
- actor/critic LR starting point `1e-4`.

These sizes may be adjusted prospectively from measured GPU/memory/update evidence while preserving comparable simulated game-time per update. Record every change.

## 6. Execute `VALIDATION_GATES.md` in order

Treat the gates as stop conditions, not paperwork.

In particular:

- prove hybrid distribution math and gradients;
- prove canonical observation parity on a broad natural RLBot corpus;
- benchmark 1/2/4-tick shared ball-prediction refresh;
- prove exact one-tick action timing at native 1x RLBot rate;
- re-run short-horizon RocketSim/RLBot transition comparison;
- prove cadence-safe reward integration;
- run a >=100k-policy-tick finite environment stress;
- run a new 120-Hz worker sweep; do not inherit 56 workers blindly;
- prove real CUDA PPO save/reload/resume;
- prove export/deployment numerical parity and native 120-Hz CPU latency/missed-tick health.

Do not use wins/losses to decide a technical cadence/parity gate.

Do not use 5x RLBot game speed to certify native 120-Hz timing. Native-rate timing comes first.

## 7. Trainer backend

The existing `rlgym-ppo` path is the first proven implementation target.

A bounded `rlgym-learn` spike is allowed because its Rust/shared-memory architecture may help at 120 Hz. Do not let framework migration dominate the milestone.

Only use/switch to `rlgym-learn` if the exact hybrid distribution, checkpoint, metrics and Windows behavior can be implemented without compromising the v9 action/observation contracts. If both paths work, benchmark complete iteration throughput and choose the evidence-backed stable backend.

Keep the contracts trainer-neutral.

## 8. Scratch pilot — only after all pre-PPO/live gates pass

Maximum authorized v9 learning pilot:

**2 simulated game-hours**

At 120 Hz 1v1 this is approximately 1.728M agent-steps.

The pilot exists to prove:

- actual scratch learning is numerically healthy;
- both analog and button heads keep learning/exploring;
- reward/event metrics are bounded;
- checkpoints/reload/resume work;
- behavior changes measurably;
- RLViser can load/watch a scratch checkpoint;
- no train/deploy contract has regressed.

Do **not** claim that a 2-hour scratch policy should beat Wisp/Nexto. Do not continue into a 100M-step or multi-thousand-hour campaign under this authority.

## 9. RLViser

Preserve/adapt the independent M07 viewer so it can load scratch checkpoints without rendering training workers.

At minimum show the car and ball normally. Optional debug overlay should expose checkpoint/simulated-hours and current physical controller output without slowing rollout workers.

## 10. Deployment/protection

Production remains frozen Wisp by default.

Scratch deployment is opt-in only and must carry/check:

- policy version;
- observation schema hash;
- action schema hash;
- canonical-adapter version;
- reward/training config version;
- model artifact hash.

No production promotion in v9.

## 11. Evidence and repository hygiene

Commit/push coherent stable boundaries rather than leaving important work only in the Codex workspace.

Expected compact result locations:

- `docs/MILESTONE_09_RESULTS.md`
- `training/results/milestone09/` for machine-readable compact evidence
- versioned scratch configs/metadata/schema under `training/`

Keep large checkpoints, raw rollout datasets and raw RLBot telemetry Git-ignored. Record exact relative path, size, SHA-256, format and reproduction command in committed reports.

Do not commit secrets, local absolute paths, virtual environments, RLViser binaries, or huge artifacts.

## 12. Final verification/report

Before calling v9 complete:

- run production and scratch test suites;
- Ruff/compile/diff checks;
- verify frozen Wisp production hashes unchanged;
- verify all committed result JSON parses;
- independently reload checkpoint/export artifacts;
- verify production default remains Wisp;
- verify worktree clean;
- push all stable work/results to `origin/main`;
- read back the remote branch/critical result files;
- report final remote SHA and exact pass/fail status for every validation gate.

If a hard gate cannot be made to pass within the coherent v9 implementation, stop at that gate, preserve/push the implementation and evidence, and report the remaining blocker. Do not spend the pilot budget trying to train around a broken action/observation/timing contract.
