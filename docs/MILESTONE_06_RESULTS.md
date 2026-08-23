# Milestone 06 Results

Milestone 06 is Rival's first serious staged RLGym/RocketSim training campaign. Production remains the frozen Wisp policy unless the explicit final promotion battery is earned.

## Fixed campaign configuration

- Campaign ceiling: 100,000,000 agent-steps.
- Measured worker count: 56.
- Student cadence: 4 physics ticks at 120 Hz.
- PPO iteration/buffer/batch/minibatch: 50,000 / 150,000 / 48,000 / 12,000.
- Actor/critic learning rates: 1e-06 / 0.0001.
- Gamma / GAE lambda: 0.994987437106620 / 0.95.

## Preflight evidence

Status: `passed`. The 24–64 sustained sweep selected **56 workers** at 12039.09 agent-steps/sec. The measured Stage A appended-action offset is `-6`.

The frozen-Wisp headless baseline was 42-58-0 over 100 balanced games. Reward/curriculum audits, a full-size PPO iteration, fresh optimizer reload, exact policy export, and production-runtime loading all passed.

## Training boundaries

### 005m

- Status: `completed_stage_boundary`; cumulative steps/updates: 5,000,010 / 297.
- Aggregate appended-action share: 0.1449%.
- Latest 100-game headless Wisp result: 43-57-0, goal differential -14; health `passed`.
- Production: frozen Wisp unchanged. Resume: `training/.venv/Scripts/python.exe training/scripts/run_m06_campaign.py --stage stage_b --appended-offset -4 --resume training/checkpoints/milestone06/005000010_stage_a_m6p0`

### 020m

- Status: `rejected_at_evaluation_boundary`; cumulative steps/updates: 20,000,016 / 1,194.
- Aggregate appended-action share: 1.0072%.
- Latest 100-game headless Wisp result: 59-41-0, goal differential +18; health `passed`.
- RLBot stage context: 0-8-0, goal differential -29.
- RLBot telemetry integrity: `passed`; transfer verdict: `severe_regression_at_20m`.
- Campaign outcome: **rejected/rollback**.
- Production: frozen Wisp unchanged. Resume: `none authorized; new authority would be required before using training/.venv/Scripts/python.exe training/scripts/run_m06_campaign.py --stage stage_c --appended-offset -4 --resume training/checkpoints/milestone06/020000016_stage_b_m4p0`

## Promotion state

Milestone 06 ended as **rejected/rollback** at the 20M clean boundary. The candidate failed the ordinary eight-game RLBot boundary, so the final 16-game promotion battery was not run and production remains frozen Wisp.
