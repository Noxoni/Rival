# M10.3 Stage 1 repair contract

## Reward V2

Keep the M10.2 car-caused distance-progress reward and physical new-touch reward unchanged, including the existing distance scale, safety clip, and absolute episode budget.

Add exactly one learning component: **pre-touch idle penalty**.

A learner is idle only when all are true:
- no physical learner-ball touch has occurred yet;
- at least 0.5 simulated seconds have elapsed since reset;
- learner linear speed is below 80 uu/s.

Penalty rate: `-0.02 reward / simulated second` while idle. At native 120 Hz this is `-0.02 / 120` per ordinary tick. Compute it from elapsed physics ticks/time so it remains physically scaled if a transition spans multiple ticks.

Stop the idle penalty immediately once speed is >=80 uu/s or after first touch.

Do not add generic speed, action-magnitude, steering, throttle, boost, goal/concede, or named-mechanic rewards. The only intent is to make doing nothing mildly worse without making arbitrary motion profitable.

Before training, truth-table test that idle is penalized, the grace period is not, useful approach retains distance reward, driving away remains negative, ball motion toward a stationary car does not pay positive progress, arbitrary motion without ball progress is not positively rewarded, first touch still pays exactly the existing touch reward, and idle penalty stays off after first touch.

## Curriculum V2

Keep the five M10.2 family names, Phase-A/Phase-B weights, distance ranges, boost ranges, team randomization, mirroring, dummy behavior, termination rules, and kickoff holdout.

Only repair initial orientation for ordinary acquisition families:
- `stationary_close`: face the ball plus uniform ±15° heading error.
- `stationary_medium`: face the ball plus uniform ±30° heading error.
- `moving_chase`: yaw and initial planar velocity approximately toward the current/intercept direction with uniform ±45° error.

`awkward_heading` remains deliberately awkward. `natural_kickoff_holdout` remains unchanged. Keep the reset distributions broad/randomized; do not use exact scripts or controller demonstrations.

## Evaluation V2

Generate fresh frozen and unseen V2 corpora before the first PPO update and evaluate the exact v10.1 source actor on both. The V2 source result becomes the no-learning baseline.

Retain all M10.2 capability metrics and add idle ticks, idle simulated seconds, pre-touch idle share, cumulative idle penalty, and deterministic action diagnostics.

The Stage-1 mastery thresholds and +5h no-learning rule remain unchanged.
