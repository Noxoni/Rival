# Milestone 10.5 Stage-1 turn/approach/first-touch experiment

## Outcome

The experiment completed its only authorized skill stage. Rival did **not** consistently learn to turn toward, approach, and physically touch the ball.

The final deterministic 500-episode evaluation produced 15.4% first-touch success, up from 9.0% for the untouched v10.1 +10h source actor. This is a small measured improvement, not a viable ball-acquisition capability. The final Phase-A prerequisite gate failed, the inherited no-material-learning rule was true at +5h, no Stage 2 or later stage was run, and the checkpoint was not promoted.

## Frozen lineage and architecture

- Source checkpoint: `training/checkpoints/milestone10_1/boundaries/plus-010h/032019870`
- Source actor SHA-256: `e6b9fd1a38dbb5c6670711a8b47769a628d2f20caf7c88738f372d0839b7a3b6`
- Transfer: actor weights only; fresh critic; fresh actor optimizer; fresh critic optimizer
- Source actor output parity error: `0.0`
- Policy, observation, and action: `RivalPolicyV1`, `RivalObsV1`, `RivalActionV1`
- Native control: 120 Hz, no action repeat, one-tick RLBot delay
- Canonical state and existing Stage-1 `RivalBallAcquisitionCurriculumV2`: unchanged
- Opponent: one inert dummy car with exact-zero controller; dummy rows structurally excluded from PPO
- Production Wisp model files: unchanged

The preflight worker sweep selected 56 workers by highest stable active-learner throughput:

| Workers | Active learner steps/s | Stable | Stalls/crashes |
|---:|---:|:---:|---:|
| 32 | 2,322.93 | yes | 0 |
| 40 | 2,908.89 | yes | 0 |
| 48 | 2,938.31 | yes | 0 |
| 56 | 3,165.81 | yes | 0 |
| 64 | 3,138.86 | yes | 0 |

## Exact reward contract

- First genuine physical touch: +10.0 exactly once; all later touches: 0
- Heading: `1.5 * (alignment_now - alignment_previous)` with independent +3.0 and -3.0 episode budgets
- Holding the same heading: 0
- Car-caused distance progress: independent +3.0 approach and -3.0 retreat episode budgets
- Ball-only translation: 0 distance-progress reward
- Idle: first 0.5 seconds exempt, then -1.4 per simulated second while pre-touch speed is below 80 uu/s
- Full eligible idle interval: -16.1
- Maximum positive episode reward: +16.0
- Goals, concedes, generic speed, boost, action magnitude, jump, and named mechanics: 0

The strict truth table passed every requested case, including `abs(-16.1) > +16.0` and first-touch payment exactly once.

## Deterministic capability results

Every row below uses the same frozen 500-episode, five-family corpus and deterministic actor methodology. `No-touch timeout` is the terminal-reason share: its 12-second clock resets after a touch, so an episode can record a first touch and later terminate for another 12-second no-touch interval.

| Actor boundary | First-touch success | No-touch timeout | Successful touch median | Failed initial distance | Failed terminal distance | Failed distance reduction | Failed alignment change |
|---|---:|---:|---:|---:|---:|---:|---:|
| v10.1 +10h source | 9.0% | 97.4% | 1.1583 s | 2,138.2769 uu | 4,184.1958 uu | -95.6807% | -0.558894 |
| M10.5 +1h | 11.0% | 97.6% | 1.0417 s | 2,169.4982 uu | 3,924.0907 uu | -80.8755% | -0.629174 |
| M10.5 +2.5h | 14.2% | 97.4% | 1.0667 s | 2,213.9464 uu | 3,736.7663 uu | -68.7831% | -0.691190 |
| M10.5 +5h | 15.4% | 97.4% | 0.8333 s | 2,229.0149 uu | 4,417.5730 uu | -98.1850% | -0.725554 |

The final core-family aggregate (excluding the natural-kickoff holdout) was 19.25% first-touch success and 96.75% no-touch timeout. Failed core episodes ended much farther away: 1,734.4861 uu initially versus 4,727.5870 uu terminal, a -172.5641% distance reduction.

Final first-touch success by family:

| Family | First-touch success | No-touch timeout |
|---|---:|---:|
| stationary_close | 54.0% | 94.0% |
| stationary_medium | 11.0% | 98.0% |
| moving_chase | 11.0% | 95.0% |
| awkward_heading | 1.0% | 100.0% |
| natural_kickoff_holdout | 0.0% | 100.0% |

The result therefore does not answer the primary question positively. Touches became slightly more frequent and successful touches became faster, but Rival did not reliably turn toward or close on the ball. Failed-episode alignment and terminal distance both worsened materially.

## Terminal checkpoint

- Directory: `training/checkpoints/milestone10_5/stage_1/boundaries/plus-005h/002159909`
- Actor SHA-256: `4ff6af0ad1f2d03383d453d5142fc32a42d28e5548118eef248086ffd0938d55`
- Manifest SHA-256: `62265d60c3e78ad7a8cc0df5e681deee7f154e73f2629efcfff5d94f299bfd92`
- Active learner steps: 2,159,909
- Simulated learner hours: 4.9997893518518515
- PPO iterations / model updates: 24 / 88
- Clean checkpoint and independent reload parity: passed

The generic hard-ceiling scheduler reserved two 56-worker segments to prevent experience overshoot and stopped 91 steps below the nominal 2,160,000-step boundary. The shortfall was inside the explicit 112-step terminal tolerance. It was reclassified without any actor, critic, optimizer, reward, or curriculum update; the exact actual step/hour values remain recorded.

## Verification and disposition

- Production tests: 91 passed
- Training tests: 177 passed
- Ruff: passed
- All deterministic evaluation episodes completed with finite metrics and zero dummy rows
- Stage 2+: not run and not authorized
- Production promotion: not authorized and not performed
- Prior M10.2, M10.3, and M10.4 Stage-1 evidence/checkpoints: preserved
- Historical paused stash: preserved as `stash@{0}: On main: rival-v4-paused-superseded-before-v4.1`

Primary machine-readable closeout: `training/results/milestone10_5/final_summary.json`.
