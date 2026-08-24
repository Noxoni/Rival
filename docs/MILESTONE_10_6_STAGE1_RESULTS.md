# Milestone 10.6 Stage-1 Uncapped Reacquisition Results

## Outcome

M10.6 completed its authorized Stage-1-only experiment and is **not a capability success**. Rival did not reliably learn to turn toward, approach, contact, and repeatedly reacquire the ball. The actor is preserved for diagnosis and is not promoted or continued.

The experiment ended at the clean +5h checkpoint with 2,159,878 active learner steps (4.999718 learner-simulated hours). The 122-step nominal shortfall is within the explicitly implemented 128-step terminal worker-segment tolerance for the selected 64-worker run. All 24 PPO iterations were healthy, all 88 model updates completed, the final checkpoint reloads exactly, and all workers were cleaned up.

## Source and isolation

- Exact source checkpoint: `training/checkpoints/milestone10_1/boundaries/plus-010h/032019870`
- Exact source actor SHA-256: `e6b9fd1a38dbb5c6670711a8b47769a628d2f20caf7c88738f372d0839b7a3b6`
- Actor weights only: yes
- Fresh critic: yes
- Fresh actor optimizer: yes, zero state entries before training
- Fresh critic optimizer: yes, zero state entries before training
- M10.2, M10.3, M10.4, and M10.5 actors used as source: no
- Dummy agent included in PPO: no
- Production modified or promoted: no

The preflight passed all 23 strict reward truth-table/trajectory requirements, RocketSim separated-contact tracing, learner/dummy isolation, frozen curriculum reset audits, fixed source-corpus evaluation, disposable CUDA PPO health, worker cleanup, and checkpoint reload parity. The disposable PPO update was discarded before the real campaign clock began.

## Deterministic capability curve

All rows use the same frozen 500-episode M10.5 Stage-1 gate corpus and deterministic actor methodology.

| Boundary | First contact | Second contact | Third contact | All three |
|---|---:|---:|---:|---:|
| Exact v10.1 source | 45/500 (9.0%) | 2/500 (0.4%) | 0/500 (0.0%) | 0/500 (0.0%) |
| +1h | 80/500 (16.0%) | 17/500 (3.4%) | 12/500 (2.4%) | 12/500 (2.4%) |
| +2.5h | 64/500 (12.8%) | 8/500 (1.6%) | 1/500 (0.2%) | 1/500 (0.2%) |
| +5h | 81/500 (16.2%) | 5/500 (1.0%) | 2/500 (0.4%) | 2/500 (0.4%) |

The contact curve is non-monotonic and repeated-contact behavior regressed sharply after +1h. The final first-touch rate improves by only 7.2 percentage points over the source and leaves 83.8% of episodes without an initial touch. A 1.0% second-touch rate and 0.4% all-three rate are not reliable reacquisition.

## Final +5h capability details

Successful-contact timing is conditioned on the very small successful subsets:

| Metric | Samples | Mean | Median | P95 |
|---|---:|---:|---:|---:|
| Reset to first contact | 81 | 1.7592 s | 1.0583 s | 5.1167 s |
| First to second contact | 5 | 1.0417 s | 1.0250 s | 1.3183 s |
| Second to third contact | 2 | 0.1042 s | 0.1042 s | 0.1454 s |

Failed first-acquisition episodes moved in the wrong direction on average:

- Ball distance: 2,241.356 uu initially to 3,849.798 uu terminal, a +1,608.442 uu increase.
- Alignment: +0.5722 initially to +0.1803 terminal, a -0.3919 decrease.
- Cumulative heading reward over all 500 episodes: -352.6385.
- Cumulative distance reward over all 500 episodes: -222.3543.
- Cumulative acquisition-time penalty over all 500 episodes: -8,063.6617.

The final deterministic action diagnostics are also pathological:

- Mean absolute throttle: 0.35546
- Mean absolute steer: 0.21231
- Jump share: 97.7715%
- Boost share: 97.8416%
- Handbrake share: 0.00787%

This is deterministic controller-output collapse toward nearly continuous jump and boost, not evidence of a useful ball-acquisition policy.

## Reward exploit result

The M10.5 reward-budget exploit was removed from the reward implementation:

- Acquisition pressure is speed-independent and continues at -1.4 per simulated second after each 0.5-second grace period.
- The first three genuinely separated contacts each pay +10; sustained contact pays once and fourth/later contacts pay zero.
- Heading is an uncapped alignment delta; oscillations naturally cancel.
- Distance progress is uncapped across the episode and continues penalizing retreat.
- Stationary-car reward from ball-only motion is zero.
- Every other reward component is zero.

The policy nevertheless continued to fail by moving farther away and degrading alignment. Per the experiment authority, this outcome points next to observation/action learnability, deterministic policy-output behavior, PPO credit assignment, and steering/control mapping. It does **not** justify another reward-magnitude retune.

## Preserved artifacts

- Preflight: `training/results/milestone10_6/preflight.json`
- Campaign state: `training/results/milestone10_6/stage1_campaign_state.json`
- Boundary evidence: `training/results/milestone10_6/stage_1/`
- Compact final summary: `training/results/milestone10_6/final_summary.json`
- Final checkpoint (Git-ignored): `training/checkpoints/milestone10_6/stage_1/boundaries/plus-005h/002159878`
- Final actor SHA-256: `776f3f2fadd1799dfc7dc141840f8b1903550b0e1ae98bd9e2d3a67fcc36c6d8`
- Final checkpoint manifest SHA-256: `d3d76869693b8e6ae5e6d2e7909c4174437b74a9ebb38e27c34171cafb5c7e65`

Frozen Wisp production remains unchanged:

- `bot/models/POLICY.lt`: `1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7`
- `bot/models/SHARED_HEAD.lt`: `3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42`

## Verification

- Production tests: 91 passed.
- Training tests: 186 passed.
- Ruff: passed for production and training code/tests/scripts.
- `compileall`: passed for production and training code/scripts.
- JSON parsing: all 236 tracked JSON documents plus the new final summary parsed successfully.
- Final checkpoint reload parity: actor, critic, both optimizers, held observations, and held outputs exact; all state finite; clean boundary.
- Remaining M10.6 training workers: zero.
- Historical paused stash preserved: `stash@{0}: On main: rival-v4-paused-superseded-before-v4.1`.
