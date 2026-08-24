# Rival Milestone 10.4 Stage-1-only result

Status: **superseded and stopped at a clean checkpoint**.

Decision: `stop_stage1_reward_experiment_superseded_at_clean_checkpoint`.

The user explicitly replaced this experiment's one-shot `−0.80` idle rule with a new reward hierarchy and prohibited retuning the same actor in place. Training was therefore stopped after completed iteration 19. No +5h boundary or +5h evaluation is claimed.

The preserved clean checkpoint is Git-ignored at `training/checkpoints/milestone10_4/stage_1/rolling/001752092`:

- actor SHA-256: `3f613c2f0fed1f22ad7285ffb403447383c9e9cce75790fd45ffeedb8e2fd4d2`;
- manifest SHA-256: `1d1aef39e076a3caa4db2d6f9cdd262f90fecbb109b0cc16e7576f2b541e995a`;
- 1,752,092 active learner-steps;
- 4.055768519 learner-simulated hours;
- 19 completed PPO iterations and 71 model updates;
- all last-iteration health checks passed;
- independent fresh reload reproduced model, optimizer, held-observation, and held-output state with zero error.

The last valid deterministic capability measurement remains +2.5h: 16.0% overall first-touch success, 98.0% overall no-touch timeouts, 15.5% acquisition-core success, 97.5% acquisition-core no-touch timeouts, and failed episodes ending 148.26% farther from the ball relative to their initial distance. Its exact decision was `continue_ball_acquisition_training`; it was not mastery.

The additional 672,077 steps after +2.5h were not evaluated, so no capability conclusion is attached to them. The run did not reach +5h and no +5h material-learning decision was emitted.

Stages 2–4 were never started, production was not modified or promoted, and no M10.4 worker remains. The exact v10.1 +10h source checkpoint remains the required source for the successor experiment; the stopped M10.4 actor will not be reused.
