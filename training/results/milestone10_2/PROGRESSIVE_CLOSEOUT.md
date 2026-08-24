# Rival v10.2 progressive campaign closeout

Status: **failed and stopped in Stage 1 Phase A**.

Authority decision: `stop_stage_1_boundary_not_reached`.

The +5-hour runner stopped at the clean checkpoint `002136107`, corresponding to 2,136,107 active learner steps and 4.94469212962963 learner-simulated hours. The authorized +5-hour boundary was 2,160,000 steps, leaving a 23,893-step shortfall. All PPO/training health checks and worker cleanup passed, but the required `boundary_reached_or_wall_stop` check failed.

The failure was caused by boundary-planning granularity: the runner rounded a positive final rollout below the normal 48,000-step PPO minibatch down to zero, exited its loop, and then correctly refused to call the boundary complete. It also incorrectly labeled the generated campaign state as wall-clock exhaustion even though 34,489.047 seconds remained. Both implementation defects are corrected for a future authorized campaign, but the failed campaign was not resumed.

No +5-hour deterministic evaluation was run because the +5-hour training boundary was not reached. Therefore this closeout does not claim either the +5-hour no-learning decision or Stage-1 mastery. The last completed evaluation remains +2.5 hours: 4.0% overall first-touch success, 5.0% acquisition-core success, 98.0% core no-touch timeouts, and a negative 206.84% failed-terminal-distance reduction (failures ended much farther from the ball). Its exact decision was `continue_ball_acquisition_training`.

No prerequisite passed. Stages 2–4 were not started. Production was not promoted or modified.

The last clean recovery checkpoint is ignored by Git at `training/checkpoints/milestone10_2/stage_1/boundaries/plus-005h/002136107`. Actor SHA-256 is `f71af340fdff99baa58b8f81821edf24ab5fccf129ee426328e7d71e949f5f02`; manifest SHA-256 is `5dfbb66be61dc4921433f21a40c89f75eeb35e4976711f15e69d0098a8e76703`. Independent reload reproduced actor, critic, optimizer states, reload observations, and held outputs with zero maximum error.

Final verification passed: 91 production tests, 153 training tests, Ruff, compileall, strict JSON parsing, frozen v10.1 source hashes, frozen Wisp hashes, ignored checkpoint status, and worker cleanup. The pre-existing paused strategy stash remains untouched.
