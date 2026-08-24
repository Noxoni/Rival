# Rival Milestone 10.3 results

Status: **failed and stopped in Stage 1 Phase A**.

Authority decision: `stop_ball_acquisition_no_material_learning_by_plus_5h`.

## Outcome

Milestone 10.3 implemented and verified the Stage-1 V2 anti-idle repair, generated fresh frozen and unseen corpora, evaluated the exact v10.1 +10h source actor before PPO, and ran immutable Stage-1 boundaries at +1h, +2.5h, and +5h. The +5h no-material-learning rule then stopped the entire progressive campaign. No remaining wall-clock budget was spent, Stages 2–4 were not started, and production was not modified or promoted.

The binding +5h comparison was negative:

- acquisition-core first-touch success increased from 10.25% at the frozen source to 16.50%, an improvement of only 6.25 percentage points;
- acquisition-core no-touch timeouts changed from 97.00% to 97.25%, an improvement of −0.25 percentage points;
- both improvements were below the required 10-point material-learning threshold, so the exact stop condition was met;
- failed episodes ended 187.58% farther from the ball relative to their initial distance rather than at least 25% closer;
- the median successful first touch was 1.6125 seconds, but isolated successful timing cannot compensate for the failed success, no-touch, family, and distance gates.

Training reward, PPO loss, entropy, throughput, and goals were used only as infrastructure diagnostics. Capability decisions came from deterministic frozen-corpus ball-acquisition metrics, not match wins or losses.

## Boundary results

| Boundary | Steps | Overall success | Core success | Core no-touch | Close | Medium | Moving | Awkward | Kickoff | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| +1h | 432,017 | 16.0% | 20.0% | 96.75% | 55% | 11% | 14% | 0% | 0% | continue |
| +2.5h | 1,080,010 | 14.4% | 18.0% | 97.50% | 39% | 14% | 15% | 4% | 0% | continue |
| +5h | 2,160,003 | 13.2% | 16.5% | 97.25% | 48% | 5% | 12% | 1% | 0% | stop: no material learning |

The strongest aggregate snapshot was the early +1h boundary, not the final actor. None approached the Phase-A thresholds: 95% close, 85% medium, 75% moving, 80% awkward, 85% core, at most 15% core no-touch, median successful touch at most five seconds, and failed episodes at least 25% closer. Phase B and unseen apparent-pass evaluation were therefore never authorized.

## Implementation and preflight

The V2 lane retained `RivalPolicyV1`, `RivalCriticV1`, `RivalObsV1`, `RivalActionV1`, native 120-Hz control, one-tick action delay, no action repeat, the existing PPO architecture/defaults, one active learner per environment, and dummy-agent exclusion from PPO. It changed only the authorized Stage-1 ordinary-family headings, physically scaled pre-touch idle term, V2 corpora/evaluation telemetry, and the repaired positive final-rollout batching behavior inherited from the M10.2 closeout.

Preflight passed reward truth-table tests, RocketSim touch-trace checks, 10,000-reset-per-family audits for both phases, dummy isolation, source evaluation on both fresh corpora, bounded throughput selection, disposable CUDA update/reload, and restart safety. The stable measured worker optimum was 56 environments at 3,211.850 active learner-steps/second; 64 produced 3,207.270 and was not selected merely because it launched.

The v10.3 package manifest has one preserved non-gating authority discrepancy: its five recorded SHA-256 fields do not match the committed blobs, although all recorded byte counts match and all six inherited v10.2 Git blob identities match exactly. The authority package was not rewritten.

## Checkpoint and accounting

The latest clean recoverable checkpoint is Git-ignored at `training/checkpoints/milestone10_3/stage_1/boundaries/plus-005h/002160003`:

- actor SHA-256: `f855bb8da476bdd8c26e346e050081678c73493278bce80cd75e58cd90f176c0`;
- manifest SHA-256: `45022baca4e043799ed34a799d8d3a666307c9efb6379fcd36846a9b0e8a6eee`;
- 2,160,003 active learner-steps;
- 5.000006944 learner-simulated hours;
- 24 cumulative PPO iterations and 88 model updates;
- independent fresh reload and held-output parity passed with zero maximum absolute error.

The real campaign clock stopped after 1,873.563 seconds, leaving 34,126.437 seconds (9h 28m 46.437s) deliberately unused. Recorded PPO-iteration wall time totaled 718.362 seconds and the three frozen gate evaluations totaled 943.024 seconds. Preflight occurred before the real Stage-1 campaign clock, and post-stop finalization was not charged into the frozen stop-state counter; both are preserved through their component evidence and repository history rather than being misreported as campaign training time.

## Final verification

- 91/91 production tests passed with two warnings;
- 163/163 training tests passed with 64 warnings;
- Ruff and compileall passed for `training`;
- all 15 pre-closeout Milestone 10.3 JSON files parsed strictly;
- independent final-checkpoint reload reproduced actor, critic, both optimizer states, held observations, and held actor/critic outputs exactly;
- the exact v10.1 source actor and manifest remain `e6b9fd1…` and `d1a785ef…`;
- frozen Wisp `POLICY.lt` and `SHARED_HEAD.lt` remain `1bd600a1…` and `3f7b6b36…`;
- checkpoint binaries remain ignored, no progressive/RLBot/Rocket League process remains, and the paused strategy stash remains untouched.

Machine-readable evidence is under `training/results/milestone10_3/`, including `progressive_closeout.json`, `progressive_campaign_state.json`, preflight, corpora, source evaluations, and all three training/evaluation/boundary records.
