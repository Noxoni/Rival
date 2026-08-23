# Milestone 08 training and evaluation

## Principle

M08 trains only the mechanics/recovery branch after transfer parity is established. The frozen strategic Wisp branch is the protected baseline.

## Initial mechanics prior

The mechanics actor has 69 outputs: PASS + appended actions 90..157.

Do not reuse the M06 `-12/-6/-4` monolithic appended-logit schedule directly because this head has a different normalized action space. Calibrate the initial PASS prior on natural observations.

Desired initial behavior:

- deterministic action is overwhelmingly PASS;
- sampled override rate is small but nonzero in eligible contexts;
- no override outside eligibility;
- appended choices are not all numerically starved.

Record probability mass and sampled rates before training.

## Training distribution

Natural 1v1 remains dominant, preferably >=80% of resets/episodes.

Minority broad reset families may include:

- aerial/wall possession;
- awkward recovery;
- low-resource aerial states;
- other generic physical contexts already supported by the training stack.

Do not build exact named-mechanic drills as the primary source of experience.

Use a modest opponent mixture such as frozen Wisp anchor plus current-policy self-play. The exact mixture can be measured/tuned, but do not create a large opponent-pool project in M08.

## Reward

Retain outcome dominance from Reward V2.

Allowed small shaping:

- useful possession/touch;
- offensive progress;
- boost efficiency;
- recovery completion/quality;
- resource acquisition/follow-up;
- useful aerial touch/wavedash-like signals already independently logged.

Named-mechanic rewards remain disabled. Mechanics signals should encourage useful consequences rather than freestyle identity.

Add override-specific diagnostics, not a large positive override reward:

- whether an override led to a useful touch;
- possession/control change;
- score/concession window;
- recovery-speed effect;
- boost consumed/retained.

## Budget

Maximum authorized M08 learning budget: **5,000,000 agent-steps**.

Recommended checkpoints:

- 0
- 500k
- 1M
- 2M
- 5M if healthy

56 workers is the starting deployment-machine choice based on M06. If the new compositor materially changes throughput, run a short 48/56/64 sanity comparison and choose the fastest stable value. Do not spend another milestone on worker-count tuning.

## Headless evaluation

At every checkpoint run deterministic frozen-Wisp evaluation and report:

- games/wins/losses/goals/goal differential;
- PASS and override rates;
- deterministic appended-action rate;
- override contexts;
- action frequencies;
- mechanics/recovery/resource metrics;
- reward components;
- policy entropy;
- PPO actor/critic metrics;
- rollout and update wall time.

The frozen strategic branch hash and zero-step first-90 parity must be rechecked after training to prove no optimizer leakage.

## RLBot transfer gates

### Zero-step/pass-only

Before training, run balanced full-match tests against installed Nexto and Wisp. Mechanics is disabled or forced PASS. This must resemble the healthy tick-8 control rather than the M07 Z4 collapse.

### 1M

If 1M headless health is acceptable, export the mechanics candidate and run a small balanced RLBot matrix against Nexto and Wisp.

Stop immediately if there is severe real-game regression attributable to the mechanics branch. Do not continue because RocketSim reward looks good.

### Final healthy checkpoint

At 2M or 5M, whichever is the final authorized healthy boundary, repeat RLBot transfer evaluation with enough games to compare against zero-step/pass-only and historical Wisp context.

M08 does not authorize the final promotion battery or production replacement.

## Acceptance outcome

M08 passes if:

1. observation parity and temporal parity gates pass;
2. mechanics-disabled/pass-only dual-rate agent preserves healthy tick-8 transfer;
3. mechanics-head PPO can run, checkpoint, reload and resume without touching strategic weights;
4. sampled/deterministic mechanics overrides actually occur at a measurable but bounded rate;
5. the final bounded candidate does not show severe RLBot regression and preferably shows evidence that at least some overrides are useful;
6. all evidence is compact, reproducible and pushed.

If the architecture transfers cleanly but 5M is insufficient to show meaningful skill gain, that is still a useful pass: M09 can then authorize a large mechanics campaign on a sound transfer foundation.