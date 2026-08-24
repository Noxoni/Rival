# Milestone 10.1 results — agency-bootstrap intervention

## Final conclusion

Milestone 10.1 completed with a **negative capability result** and stopped at the
mandatory Phase A +10-hour review ceiling. The implementation, preflight, training
transport, checkpointing, evaluation, and final verification all passed technically;
the learning intervention did not pass its basic-agency gate.

The campaign resumed the exact final Milestone 10 +25-hour checkpoint and added
8,641,060 agent-steps, or 10.0012269 simulated game-hours, through 45 complete PPO
updates. The final checkpoint is:

`training/checkpoints/milestone10_1/boundaries/plus-010h/032019870`

- cumulative agent-steps: `32,019,870`;
- cumulative simulated game-hours: `37.0600347222`;
- actor SHA-256:
  `e6b9fd1a38dbb5c6670711a8b47769a628d2f20caf7c88738f372d0839b7a3b6`;
- checkpoint-manifest SHA-256:
  `d1a785ef439b0127b5ab1a9ff1693ade1aa11d850151cd17b9733bbeb98dacb3`.

At +10 bootstrap hours, Rival drove quickly but did not reliably acquire or interact
with the ball. The deterministic bootstrap probe measured 1,262.27 uu/s mean planar
speed, but only 6.40 distinct logical touches per 100k agent-steps, an 83.33% no-touch
timeout share, 0.91 repeated-chain touches per 100k, zero natural-goal episodes, and a
mean ball distance of 3,815.30 uu. Phase A required at least 75 logical touches per
100k, no more than 60% no-touch timeouts, and all other readiness conditions.

The final authority decision is therefore:

`stop_phase_A_readiness_failed_by_plus_10h`

Phase B was never activated. The +15h, +20h, and +25h v10.1 boundaries were not run.
The remaining 12,958,940 authorized agent-steps / 14.9987731 simulated hours were not
spent. No checkpoint was promoted. Production Rival remains frozen Wisp.

## Exact source and frozen architecture

The v10.1 activation source remained the exact M10 +25 checkpoint:

`training/checkpoints/milestone10/boundaries/plus-025h/023378810`

- cumulative agent-steps: `23,378,810`;
- actor SHA-256:
  `5d246a7eee8af22290f6f644a3e408f786551dc893bf10f19d487945329100c1`;
- manifest SHA-256:
  `903a38cdb85d8a171e207c18718d9651cae1ec905003f0fbf3729c5203202784`.

The intervention did not change the scratch actor, critic, observation, action,
canonical adapters, native cadence, one-tick transport, PPO settings, export path, or
worker count.

| Contract | Frozen implementation |
|---|---|
| Policy | `RivalPolicyV1`; no Wisp actor/trunk |
| Critic | `RivalCriticV1` |
| Observation | train/deploy-identical `RivalObsV1`, 714 floats |
| Action | `RivalActionV1`: five continuous axes plus joint jump/boost/handbrake categorical |
| Cadence | native 120-Hz physics and 120-Hz policy; one decision per tick |
| Transport | exact one-tick selected/pending/applied controller delay |
| PPO | frozen M10 values; 192k rollout/batch, 48k minibatches, one epoch |
| Workers | 56 RocketSim environments |
| Production | frozen Wisp, tick skip 8; scratch remains unpromoted |

Only the versioned learning distribution changed:

- `RivalAgencyBootstrapRewardV1`;
- `RivalAgencyBootstrapCurriculumV1`;
- `RivalAgencyBootstrapEnvV1`;
- `RivalAgencyBootstrapMetricsV1`.

## Bootstrap reward and curriculum

The reward retained exact outcome precedence at +10/-10. Non-outcome shaping used
cadence-safe useful speed, ball-approach and ball-progress potential differences,
distinct physical ball-touch events, aerial-touch events, and a bounded repeated-touch
chain schedule. It accepted no action/controller argument and contained no reward for
jump, dodge, boost, handbrake, or other button presses. Combined absolute shaping
budgets summed to 7.5, below one outcome magnitude.

Logical contacts used an eight-native-tick debounce. Continuous contact did not create
multiple touches; a separated self touch incremented the chain; an opponent touch or
300-tick chain timeout reset it. Aerial credit required a physical logical touch plus
the frozen height/surface classification.

Phase A sampled the six broad families at 30/20/20/15/10/5 percent:

- ground acquisition;
- moving-ball chase;
- touch chain;
- easy aerial contact;
- easy finish;
- natural kickoff/play.

The active team and left/right geometry were randomized symmetrically. No-touch
timeout was 10 seconds and the ordinary episode maximum was 120 seconds.

## Preflight

Preflight passed before campaign experience was consumed:

- reward formulas, cadence integration, outcome precedence, component budgets,
  debounce, chain behavior, aerial classification, and absence of direct action reward
  passed focused tests;
- 10,000 resets in each of Phases A, B, and C passed distribution, finite/legal
  physics, active-team balance, and left/right balance checks;
- a dead policy truncated at the frozen 10-second no-touch timeout;
- the exact M10 +25 checkpoint freshly reloaded;
- a disposable 192,008-agent-step CUDA PPO update ran on all 56 workers at 3,726.95
  agent-steps/sec, with both continuous and button branches and the critic updated;
- source-checkpoint and frozen-production hashes remained unchanged;
- all six reset families were directly inspected through a separate one-environment
  RLViser process at native 120 Hz.

The RLViser path is optional and disabled by default. It never renders PPO workers.
The documented entry point is:

```powershell
training/.venv/Scripts/python.exe training/scripts/run_m10_1_rlviser_spectator.py `
  --checkpoint training/checkpoints/milestone10_1/boundaries/plus-010h/032019870 `
  --family all --playback-speed 1
```

## Training boundaries

All 45 PPO updates were finite, updated both hybrid policy branches and the critic,
kept all workers alive, and wrote atomic two-state rolling recovery checkpoints plus
immutable evaluation boundaries.

| Boundary | Achieved hours | Updates | Mean agent-steps/s | Phase A gate | Decision |
|---:|---:|---:|---:|---|---|
| +2.5h | 2.6669815 | 12 | 3,737.56 | Failed | Continue Phase A |
| +5h | 5.1116157 | 11 | 2,373.61 | Failed | Continue Phase A |
| +10h | 10.0012269 | 22 | 2,352.86 | Failed | Stop v10.1 |

The throughput drop after the first boundary was recorded rather than hidden. CPU
remained saturated, GPU work continued, all 56 workers stayed alive, and no
architecture/topology change was authorized mid-campaign.

Training-rollout reward, losses, entropy, throughput, goals, and worker health were
diagnostics only. They were not capability gates. In particular, stochastic resets
often produced goals while logical touches stayed around 6-15 per 100k; those goals
were not treated as proof of agency.

## Deterministic bootstrap capability curve

Each boundary used the same deterministic current-actor self-play protocol: all six
reset families, eight episodes per family, alternating active team, native one-tick
control, and the full frozen environment timeouts. The pre-bootstrap +25 policy was
evaluated once under that same new probe and reused exactly at later boundaries.

| Metric | Pre-bootstrap +25 | +2.5h | +5h | +10h | Phase A threshold |
|---|---:|---:|---:|---:|---:|
| Mean planar speed (uu/s) | 399.85 | 335.28 | 511.35 | **1,262.27** | >=600 |
| Mean ball distance (uu) | 2,433.02 | 2,966.15 | 3,085.77 | **3,815.30** | diagnostic; lower is better |
| Logical touches / 100k | 8.04 | 3.74 | 4.63 | **6.40** | >=75 |
| Aerial logical touches / 100k | 0.00 | 0.94 | 0.00 | **0.91** | diagnostic in Phase A |
| Jumps / 100k | 247.46 | 177.71 | 243.42 | **63.09** | >=5 |
| Dodges / 100k | 217.98 | 139.36 | 199.92 | **58.52** | diagnostic in Phase A |
| Two-plus-chain touches / 100k | 2.68 | 0.00 | 0.93 | **0.91** | diagnostic in Phase A |
| No-touch timeout share | 81.25% | 79.17% | 81.25% | **83.33%** | <=60% |
| Natural goal episodes | 0 | 0 | 0 | **0** | diagnostic in Phase A |
| Easy-finish logical touches | 0 | 0 | 0 | **2** | touch-backed goal required |

The +10 actor passed the speed, jump, and touch-backed easy-finish conditions. It
failed logical-touch density and no-touch-timeout share, so the combined Phase A gate
failed. Its mean ball distance was 56.81% worse than the pre-bootstrap policy, not
materially better.

Easy-finish resets produced several goals at every boundary. At +2.5 and +5 there
were zero logical touches in that family, meaning the initial ball motion could score
without policy contact. Those passive outcomes were explicitly not credited as
finishing success. At +10, the family finally contained two logical touches, but the
overall readiness gate still failed by a large margin.

## Historical fixed-protocol comparison

The original M09/M10 12-episode deterministic protocol was retained separately so
the long learning curve remained comparable. Rates from this historical protocol
must not be mixed numerically with the six-family bootstrap probe.

| Checkpoint | Mean planar speed | Mean ball distance | Touches / 100k | Jumps / 100k | Dodges / 100k | Aerial touches / 100k |
|---|---:|---:|---:|---:|---:|---:|
| M09 Gate 13 | 335.69 | 3,455.15 | 83.33 | 0.00 | 0.00 | 0.00 |
| M10 +10h | 636.62 | 3,844.55 | 10.42 | 0.00 | 0.00 | 0.00 |
| M10 +25h / v10.1 source | 420.57 | 3,209.53 | 3.47 | 246.53 | 225.69 | 3.47 |
| v10.1 +2.5h | 227.54 | 3,343.10 | 3.47 | 123.26 | 98.96 | 3.47 |
| v10.1 +5h | 425.39 | 4,037.89 | 3.47 | 175.35 | 166.67 | 3.47 |
| v10.1 +10h | **1,106.28** | **4,004.12** | **3.47** | **38.19** | **55.56** | **0.00** |

This confirms the same pattern: much faster deterministic driving without increased
historical-protocol contact, with worse ball distance and less fixed mechanics use by
the final boundary.

## Reward-integrity interpretation

At +10, absolute non-outcome reward spend was distributed approximately as follows:

- ball-approach potential: **88.17%**;
- useful-speed rate: 4.89%;
- ball-touch event: 3.42%;
- ball-progress potential: 2.62%;
- aerial-touch event: 0.73%;
- touch-chain event: 0.16%.

The issue was not direct action reward or useful-speed dominance. The policy learned
fast movement while ball-approach potential dominated shaping, yet distinct contacts
remained below the source checkpoint and mean ball distance worsened. Under the
handoff interpretation rules, rising motion/reward without ball agency is evidence of
an unsuccessful or exploited shaping objective, not success.

## Final verification

Final verification passed as repository/checkpoint evidence while preserving the
negative learning conclusion:

- 91/91 production tests passed with two warnings;
- 140/140 training tests passed with three warnings;
- Ruff over `training/` and compileall over training code/tests passed;
- frozen Wisp self-test passed with 16 collision meshes, finite logits, 432-value
  observation, and 90-logit policy output;
- frozen production hashes remained:
  - `POLICY.lt`:
    `1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7`;
  - `SHARED_HEAD.lt`:
    `3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42`;
- clean-environment production still selected `frozen_wisp_production`, tick skip 8,
  and no scratch/M08/candidate path;
- the final checkpoint reloaded twice with zero parameter error, finite held outputs,
  46 actor-optimizer state entries, and 43 critic-optimizer state entries;
- the exact M10 +25 source checkpoint remained byte-identical;
- no later M10 or v10.1 boundary exists and no campaign process remained;
- checkpoint binaries and pinned RLViser remained present, ignored, and untracked;
- the historical `rival-v4-paused-superseded-before-v4.1` stash remained preserved;
- no production promotion was authorized or performed.

The machine-readable final verification is
`training/results/milestone10_1/final_verification.json`.

## Evidence index

Committed compact evidence is under `training/results/milestone10_1/`:

- `preflight.json`;
- `preflight_rlviser.json`;
- `preflight_rlviser_visual_inspection.json`;
- `boundary_plus-002p5h.json`;
- `boundary_plus-005h.json`;
- `boundary_plus-010h.json`;
- `final_summary.json`;
- `final_verification.json`.

Large checkpoints and raw iteration/evaluation reports remain ignored. The compact
boundary evidence records their relative paths, checkpoint hashes, raw-evidence
hashes, gate conditions, and decisions.

## Promotion and next intervention

The promotion decision is:

`not_authorized_not_promoted`

The final +10 actor is a recoverable research checkpoint, not a production candidate.
The v10.1 authority identifies imitation/motor pretraining or an active external
opponent curriculum as evidence-supported categories for a future, separately
versioned intervention. Neither is authorized or implemented by this milestone.
