# Milestone 09 results — scratch native-control Rival foundation

## Final conclusion

Milestone 09 passed Gates 0 through 14 as a scratch-policy **foundation**, not as a
production promotion or a claim of competitive Rocket League skill.

The milestone replaced the rejected Wisp-as-policy-skeleton direction with a new
`RivalPolicyV1` actor trained from scratch on the exact `RivalObsV1` and
`RivalActionV1` contracts. The scratch path makes one policy decision per 120-Hz
physics tick, emits all five continuous controller axes plus one joint eight-way
jump/boost/handbrake choice, has no lookup table or action mask, and uses one shared
canonical train/live observation implementation.

The bounded learning pilot ended at **1,680,214 cumulative agent-steps**, or
**1.944692 simulated game-hours**. This includes the 576,024 Gate 11 steps and leaves
47,786 steps unused beneath the 1,728,000-step/two-hour ceiling. No more v9 training
is authorized by this package.

The pilot was numerically healthy and changed behavior measurably. Under the same
57,600-agent-step deterministic evaluation protocol, mean distance to the ball fell
from 4,246.96 to 3,455.15 uu, touches rose from 0 to 83.33 per 100k agent-steps, and
recovery-like landings rose from 3.47 to 13.89 per 100k. These are behavior metrics,
not win-rate claims. Both evaluation sides scored zero, and score was explicitly
excluded from every technical gate.

The production default is still frozen Wisp: `POLICY.lt` plus `SHARED_HEAD.lt`, a
432-value observation, 90-action table, and tick skip 8. Scratch deployment remains
diagnostic opt-in only. **No trained checkpoint was promoted.**

## Frozen architecture

| Contract | Final v9 implementation |
|---|---|
| Policy | `RivalPolicyV1`, scratch actor, 1,965,784 parameters |
| Critic | `RivalCriticV1`, independent parameters, 1,959,623 parameters |
| Observation | `RivalObsV1`, 714 normalized floats |
| Canonical state | `RivalCanonicalStateV1` with thin RocketSim and RLBot v5 adapters |
| Action | Five tanh-squashed Gaussian axes plus joint 8-way button categorical |
| Physical controller order | throttle, steer, pitch, yaw, roll, jump, boost, handbrake |
| Cadence | 120-Hz physics and 120-Hz policy; exactly one decision per tick |
| Action delay | Exact selected/pending/applied one-tick transport |
| Prediction | Shared RocketSim ball prediction refreshed every tick |
| Symmetry | Episode-stable left/right reflection; actions and observations share it |
| Reward | `RivalScratchRewardV1`, outcome-dominant and cadence-safe |
| Trainer | Pinned `rlgym-ppo` 1.3.13 at commit `4ffd2e924198bf4b2d59f4bf280b29919d7c07ea` |
| Workers | 56, selected by a new 120-Hz measured sweep |
| Production | Frozen Wisp; scratch is opt-in diagnostic-only |

The actor contains no Wisp or Nexto parameters. Wisp/Nexto are permitted only as
future opponents or benchmarks. The actor has no 90/158-row lookup table, no
`RepeatAction`, no named-mechanic macro, no state-dependent action mask, and no
separate approximate deployment observation builder.

## RLBot v5 geometry authority

The standard-Soccar geometry layer was reconciled against the official
[RLBot v5 Useful Game Values](https://wiki.rlbot.org/v5/botmaking/useful-game-values/),
[Game Data](https://wiki.rlbot.org/v5/botmaking/game-data/), the official
[GameData FlatBuffers schema](https://github.com/RLBot/flatbuffers-schema/blob/main/schema/gamedata.fbs),
and RLBot's guidance for
[extracting exact map meshes](https://wiki.rlbot.org/v5/miscellaneous/extracting-map-meshes/).

The frozen standard values include:

- floor `z = 0` and ceiling `z = 2044`;
- planar side walls at `x = +/-4096`;
- planar back walls and physical goal line at `y = +/-5120`;
- the 45-degree corner plane `abs(x) + abs(y) = 8064` in each quadrant;
- documented physical goal opening 1785.51 uu wide by 642.775 uu high;
- goal depth 880 uu and derived goal back at `abs(y) = 6000`;
- all 34 boost-pad identities in RLBot FieldInfo order, with six full pads and 28
  small pads;
- the official kickoff and demolition spawn coordinates;
- ball radius 91.25 uu, ball maximum speed 6000 uu/s, car maximum speed 2300
  uu/s, and standard gravity magnitude 650 uu/s^2.

The geometry code does **not** pretend a planar approximation gives exact curved-ramp
or rounded-post clearance. Those require collision-mesh queries. A live v5 beta
FieldInfo capture reported approximately 1920-by-752 goal/scoring volumes at `z=312`;
that runtime metadata is preserved separately and is not substituted for the
documented 1785.51-by-642.775 physical opening.

Machine-readable authority and the full 34-pad audit are in
`training/results/milestone09/rlbot_v5_geometry_authority.json`.

## Validation gates

| Gate | Status | Evidence and decision |
|---|---|---|
| 0 — reconcile M08 | Passed | M08 stopped at a clean 4,999,790-step boundary, retained all rejected overlay evidence/checkpoints, spent no remaining M08 budget, and promoted nothing before the v9 package was merged onto newer `main`. |
| 1 — action contract | Passed | Hybrid math, tanh Jacobian, joint categorical, controller round trip, deterministic mode, gradients, no mask, and representative advanced controller traces all passed. |
| 2 — canonical/schema | Passed | One generated 714-float schema, canonical adapters, shared feature implementation, exact field metadata/hash, 34 pads, eight-tick histories, one-tick deltas, prediction, and modern jump/dodge state passed. |
| 3 — broad observation parity | Passed | 9,045 RocketSim/RLBot observation states produced zero mismatches. The native corpus retained 6,000 records with SHA-256 `b69b9f1cca7e77f3f879fd383e5e97340078724b108b153e518a52fad6ef54d8`. |
| 4 — prediction refresh | Passed | Measured 1/2/4-tick refresh selected one tick. Native cadence was retained instead of the design's expected four-tick refresh. |
| 5 — exact timing | Passed | Selected, pending, and applied physical controllers matched the one-tick delay contract at native 1x RLBot rate. No 5x result was used to certify timing. |
| 6 — transition audit | Passed | 64 short-horizon RocketSim/RLBot windows passed the bounded transition comparison. Contact-free physics tolerances, not wins or scores, decided the gate. |
| 7 — reward cadence | Passed | Goal/concede remained +/-10; shaping used potential differences or physical `dt=1/120`; combined absolute shaping budget was 8.75, below one outcome magnitude. |
| 8 — environment stress | Passed | 100,000 policy ticks / 200,000 agent-steps completed with finite observations/actions/rewards and clean reset/history isolation. The raw physics comparison tolerance was honestly amended prospectively from 0.001 to 0.002 after measured float noise; it was not hidden. |
| 9 — worker sweep | Passed | Real v9 actor/environment sweep measured 16/24/32/40/48/56/64 workers. 56 won at 5,072.97 agent-steps/s; 64 remained stable but was slightly slower at 5,060.67. |
| 10 — backend decision | Passed | Pinned `rlgym-ppo` was selected. A bounded `rlgym-learn` 1.0.5 / `rlgym-learn-algos` 0.2.6 Windows API import was viable but not integrated because migration was not justified for the foundation. |
| 11 — real CUDA PPO | Passed | Three real iterations reached 576,024 cumulative steps; GAE/losses were finite, both hybrid heads and critic updated, all branches sampled, both optimizers/counters reloaded, and a fresh parent resumed another update. |
| 12 — export/live inference | Passed | Selected TorchScript matched the training actor within `1e-5`; actor CPU p99 was 0.7365 ms and full observation-to-controller p99 was 4.318 ms. A native RLBot v5 smoke had 2,378 contiguous records with zero post-warmup gaps and callback p99 3.296 ms. |
| 13 — bounded pilot/RLViser | Passed | Six checkpointed PPO updates ended at 1,680,214 steps; every update was finite/nonzero with all branches explored; fixed behavior changed materially; a fresh reload was exact; the pinned RLViser process rendered the final checkpoint in a separate one-environment process. |
| 14 — repository/final verification | Passed | Production tests, v9 tests, Ruff, compile, Wisp self-test/hash/default probes, JSON parsing, ignored-artifact checks, independent checkpoint/export reloads, hygiene scans, push, and remote readback passed. |

Technical cadence/parity gates were based on tick continuity, controller transport,
transition error, latency, and missed-tick evidence. They were never based on whether
Rival won or lost a match.

## Worker sweep

| Environments | Sustained agent-steps/s |
|---:|---:|
| 16 | 2,869.74 |
| 24 | 3,801.49 |
| 32 | 4,224.69 |
| 40 | 4,402.18 |
| 48 | 4,893.30 |
| **56** | **5,072.97** |
| 64 | 5,060.67 |

The selected count is the highest sustained stable throughput, not the largest count
that launched. High CPU utilization was accepted when stable throughput remained
higher. Real 48k CUDA PPO probes at both 56 and 64 workers produced finite updates and
nonzero analog/button gradients.

## Gate 11 technical PPO smoke

The selected Gate 11 run is `gate11-20260823T190944Z`. Its final ignored checkpoint is:

`training/checkpoints/milestone09/gate11-20260823T190944Z/resumed`

- cumulative agent-steps: 576,024;
- simulated game-hours: 0.6666944444;
- final actor SHA-256:
  `e58c34e66d190ead95a63a3a1b36ea9bc6090a0d21ec31472f771f992ccb0666`;
- checkpoint-manifest SHA-256:
  `818c3a2b6fce2d34dafb0ad133082a469fe61255d45665d3fe8969fa0b8b6c81`.

The evidence retains the earlier failed same-parent worker-respawn attempt and its
Windows paging error. The selected result uses fresh parent processes and does not
erase that attempt history.

## Gate 12 deployment result

The selected opt-in deployment artifact remains the Gate 12 export from the Gate 11
checkpoint:

`training/artifacts/milestone09/gate12/rival_v9_scratch.ts`

- format: `rival-v9-torchscript-deterministic-controller-v1`;
- size: 7,881,248 bytes;
- SHA-256:
  `0e5ddbdc8a7ebe6f7119d2e271efe75fa8478e1821bb795a52d78ba519ef383a`;
- metadata SHA-256:
  `7f6f9ce64afdeb1352c157d09f76da5224ab0c2ca7a2a6df7c2e172162ed1277`.

The `torch.export` candidate was numerically valid but not selected because its CPU
probe had large latency outliers. The artifact, export reference, raw live telemetry,
and checkpoints remain Git-ignored; compact parity/latency evidence is committed.

Scratch RLBot deployment requires all of the explicit diagnostic environment
variables documented by the Gate 12 metadata. With a clean environment, production
loads Wisp and tick skip 8.

## Gate 13 pilot

Gate 11 steps counted toward the two-hour limit. Gate 13 added 1,104,190 steps through
five approximately 192k updates and one intentionally smaller 144k update. The smaller
last boundary preserved complete 48k minibatches and a 47,786-step safety margin.

The pilot reset contract is versioned separately from the reproducible Gate 11
diagnostic environment. It is the required approximate mixture:

- 70% natural kickoff/play;
- 10% broad ground possession/challenge;
- 8% wall/aerial/ceiling possession;
- 8% awkward recovery/landing;
- 4% low-resource states.

Observed reset shares across the short pilot were 76.02%, 9.95%, 3.62%, 6.33%, and
4.07%, respectively. Every family occurred; natural play remained the majority. The
policy, action, observation, reward, PPO, cadence, worker count, and architecture were
unchanged by this reset/metric version.

Every completed iteration immediately wrote a full actor, critic, both optimizers,
counters, config/contract, and held reload corpus. The final ignored checkpoint is:

`training/checkpoints/milestone09/gate13-20260823T200008Z/phase2/1680214`

- format: `rival-v9-hybrid-ppo-checkpoint-v1`;
- actor size: 7,877,740 bytes;
- actor SHA-256:
  `12770f082c6cbe1fbab8809580dc775d1d78071825eb9481df4a16d9ee85fbe5`;
- checkpoint-manifest SHA-256:
  `9b08b454f587248ed184a194aac9b57d904fe1c5ba5b8efe8e5b9e84c9ae469e`;
- fresh second-reload maximum parameter error: exactly 0.

One implementation note is intentionally preserved. Phase 1 used a seed-forwarding
subclass of Gym `Box`; rlgym-ppo 1.3.13 checks `type(space) == Box`, so it *reported*
action-space code 0 (discrete). The wrapper's `is_discrete` flag remained false and its
stored physical float actions proved that no casting or quantization happened: every
analog axis had nontrivial continuous range and all eight button combinations were
sampled in every iteration. Commit `334ed3b` replaced the subclass with a concrete
Gym `Box`, and Phase 2 correctly reported action-space code 2 (continuous). Phase 1
was not rerun because doing so would duplicate authorized learning experience without
changing the actual transport.

### Fixed behavior comparison

| Metric | Gate 11 baseline | Gate 13 final |
|---|---:|---:|
| Mean speed (uu/s) | 396.28 | 337.97 |
| Mean planar speed (uu/s) | 360.36 | 335.69 |
| Mean distance to ball (uu) | 4,246.96 | 3,455.15 |
| Mean boost | 50.40 | 45.02 |
| Touches / 100k agent-steps | 0.00 | 83.33 |
| Recovery-like landings / 100k | 3.47 | 13.89 |
| Deterministic first jumps / 100k | 0.00 | 0.00 |
| Deterministic dodges / 100k | 0.00 | 0.00 |
| Deterministic aerial touches / 100k | 0.00 | 0.00 |
| Scores (blue/orange) | 0 / 0 | 0 / 0 |

Ten of 13 fixed signature metrics changed materially and the held-observation policy
fingerprint changed. The policy is still extremely early: deterministic jump/dodge
and aerial-touch metrics remained zero. Stochastic rollouts nevertheless sampled all
five analog axes and all eight joint button combinations, with finite entropy and
nonzero gradients in every update. This is enough for the technical learning gate,
not enough for a skill or mechanics-competence claim.

Mechanic-like event detectors are diagnostics only and do not feed reward. Weak
signatures remain labeled `*-like`; the results do not claim that a named mechanic was
learned from a detector firing.

## Optional scratch RLViser viewer

The scratch viewer is opt-in, disabled by default, and runs one independent
RocketSim/RLViser environment. It never renders training workers. Install the pinned
viewer if needed:

```powershell
./training/install_rlviser_spectator.ps1
```

Preflight the exact checkpoint/observation/action seam without opening a window:

```powershell
training/.venv/Scripts/python.exe training/scripts/run_m09_rlviser_spectator.py `
  --checkpoint current --check
```

Watch the highest-step local Gate 13 checkpoint at approximately real time:

```powershell
training/.venv/Scripts/python.exe training/scripts/run_m09_rlviser_spectator.py `
  --checkpoint current --playback-speed 1
```

The live smoke positively observed pinned RLViser v0.8.2, ran 535 decisions over
5.018 wall seconds with only two missed pacing deadlines, and verified one-tick
120-Hz physics/policy transport. The viewer's wall-clock startup is not used as a
cadence certification; native cadence was already proved by Gate 5 and Gate 12.

## Final verification

The final verification package includes:

- 90/90 production tests passed (two upstream TorchScript deprecation warnings);
- 78/78 v9 scratch tests passed;
- relevant Ruff checks passed;
- compile checks passed;
- the source Wisp self-test passed with 16 collision meshes, finite 90-logit output,
  and exact action compatibility;
- frozen production hashes remained:
  - `POLICY.lt`:
    `1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7`;
  - `SHARED_HEAD.lt`:
    `3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42`;
- a clean-environment config probe selected `frozen_wisp_production`, tick skip 8,
  `POLICY.lt`, and no scratch/M08/candidate path;
- every committed Milestone 09 JSON parsed;
- no committed M09 artifact contained a local absolute path or apparent secret;
- final checkpoint and Gate 12 export independently reloaded and reproduced finite,
  contract-valid outputs;
- checkpoints, deployment artifacts, raw telemetry, and RLViser binary remained
  ignored;
- the historical pre-v4.1 stash remained preserved.

The exact Gate 14 machine-readable result is
`training/results/milestone09/gate14_final_verification.json`. The final remote SHA is
reported in the completion handoff after `origin/main` readback, because a commit
cannot truthfully contain its own final SHA.

## Evidence index

Compact machine-readable results are under `training/results/milestone09/`:

- `gate01_action_contract.json`;
- `gate02_canonical_schema.json`;
- `gate03_native_capture.json` and `gate03_observation_parity.json`;
- `rlbot_v5_geometry_authority.json`;
- `gate04_prediction_cadence.json`;
- `gate05_timing_parity.json`;
- `gate06_transition_audit.json`;
- `gate07_reward_cadence.json`;
- `gate08_environment_stress.json`;
- `gate09_worker_sweep.json`;
- `gate10_backend_decision.json`;
- `gate11_hybrid_ppo.json`;
- `gate12_cpu_runtime_probe.json`, `gate12_scratch_live_smoke.json`, and
  `gate12_export_live_inference.json`;
- `gate13_scratch_pilot.json`;
- `gate14_final_verification.json`.

Large checkpoints, raw rollout data, export artifacts, raw RLBot telemetry, and the
RLViser executable are intentionally absent from Git. Their exact relative paths,
formats, sizes, and SHA-256 values are recorded by the corresponding compact evidence.

## Promotion decision

Milestone 09 authorizes no production promotion. The final decision is:

`not_authorized_not_promoted`

Production Rival remains frozen Wisp until a future authority package defines a
serious scratch campaign and an explicit promotion gate. A future milestone should
start from the preserved 1,680,214-step checkpoint only after reviewing this
foundation; it must not infer that two simulated hours establish the architecture's
skill ceiling.
