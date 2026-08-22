# Milestone 06 Training Campaign

## 1. Goal

Run Rival's first material learning campaign from the verified Wisp warm start. The goal is not to prove every advanced mechanic in one milestone. The goal is to produce a materially trained policy, preserve core 1v1 competence, open the mechanics-capable action space, and determine whether learned behavior is moving beyond Wisp in useful ways.

## 2. Budget and cadence

Campaign ceiling: **100,000,000 agent-steps**.

Use the measured 24-worker configuration from Milestone 05 unless a new measured stability problem requires a lower count. Keep `mechanics4` (4 physics ticks/action) for the trainable student.

The run must be resumable and split into coherent stages. Do not make the entire campaign one opaque process with no recoverable progress.

Recommended boundaries:

- Stage A: 0–5M agent-steps — open novel actions cautiously and verify learning health.
- Stage B: 5–20M — early mechanics/recovery learning while protecting Wisp competence.
- Stage C: 20–50M — full natural self-play fine-tuning with broader state exposure.
- Stage D: 50–100M — mature fine-tuning only if prior gates remain healthy.

A stage may stop early for evaluation, rollback, or promotion. Do not continue simply to consume the budget.

## 3. PPO scale

Move from the Milestone 05 smoke sizes to real PPO batches. Use the pinned rlgym-ppo implementation and document exact values.

Start from the package's established conventions and the upstream RLGym-PPO scale rather than inventing tiny batches. A sensible initial neighborhood is:

- `ts_per_iteration`: around 50,000 agent-steps;
- experience buffer: around 150,000;
- PPO batch: around 50,000;
- one or a small number of PPO epochs;
- actor LR kept conservative relative to a random-start agent because the actor begins from Wisp;
- critic LR may remain higher than actor LR;
- observation standardization off unless evidence says otherwise because the 432 teacher-compatible observation is already normalized;
- return standardization may be enabled if verified finite/stable.

Do not spend the milestone sweeping dozens of hyperparameter combinations. If the initial serious configuration is numerically healthy, train it. Change one high-impact setting at a time only when evidence requires it.

### Cadence-aware discounting

The serious student acts every 4 physics ticks rather than Wisp's 8. Preserve approximately the same physical discount horizon rather than blindly copying an 8-tick gamma. If using an 8-tick reference gamma of `0.99`, the equivalent per-4-tick discount is approximately `sqrt(0.99) ~= 0.995`. Record the chosen gamma and rationale. Treat GAE lambda separately and keep it in a stable PPO range.

## 4. Opening the 68 appended actions

Milestone 05 correctly initialized actions 90–157 with zero weights and a `-12` bias, producing zero appended selections in the smoke. Serious learning now needs measurable exploration without instantly destroying Wisp behavior.

Do **not** simply zero all 68 biases at once.

Implement a checkpointed novel-action exploration prior or equivalent mechanism that is part of the policy/log-probability path used by PPO. Changes to this prior must happen only at iteration/stage boundaries and must be recorded in checkpoints/reports.

Before the long run, calibrate a small set of candidate appended offsets (for example `-10, -8, -6, -4, -2`) over a large batch of natural Wisp/RocketSim observations. Measure:

- total probability mass on actions 90–157;
- sampled appended-action rate;
- deterministic top-1 legacy retention;
- action entropy.

Choose the safest offset that produces **non-negligible but minority** appended exploration. Do not choose by guess alone.

As training progresses, relax suppression only at stage boundaries when:

- appended actions are being selected without catastrophic outcome regression;
- the policy remains numerically stable;
- frozen-Wisp evaluation is not collapsing;
- appended action use is not just random high-entropy thrashing.

The final mature policy should not require an arbitrary permanent training-only suppression to function. Stage D should target zero external appended-action suppression if evidence supports it.

## 5. Training distribution

Natural 1v1 remains the majority distribution.

### Stage A

Approximately 90% ordinary natural/kickoff states, with at most 10% broad enrichment split between aerial/wall possession and recovery states.

### Stage B/C/D

Natural states must remain at least 75–80% of training. Minority enrichment may increase only enough to expose mechanics/recovery opportunities.

Allowed enrichment is **broad randomized state distribution**, not exact scripted drills. Examples:

- ball/car broadly distributed near sidewalls, backboard, ceiling, or aerial possession heights;
- varied boost levels and approach velocities;
- awkward wall/air recovery orientations;
- offense-to-defense transition states;
- replay-derived states later if a clean replay source is available.

Randomize position, velocity, orientation, boost, opponent pressure and field side within valid ranges. Avoid fixed coordinates or one exact named trick setup.

Use RLGym/RLGym-Tools mutators such as weighted/randomized mutators where useful. Log the reset-source mix.

## 6. Opponent and forgetting control

Start with the simplest stable natural self-play supported by the current trainer. Do not build a large league-management framework before training begins.

However, Wisp competence must be monitored continuously. At minimum perform frozen-Wisp evaluations outside the training rollouts.

If self-play exhibits obvious catastrophic forgetting or cyclic exploitation, use the smallest evidence-backed countermeasure, in this order:

1. rollback to last healthy checkpoint;
2. reduce actor LR / slow novel-action opening;
3. add a small frozen-Wisp anchor or frozen/historical opponent mixture if it can be implemented without destabilizing the trainer.

Do not spend the milestone building a complex opponent league unless simple self-play actually fails.

## 7. Checkpointing and progress persistence

- Save a recoverable full checkpoint at least every 1M agent-steps or the nearest safe interval supported by the trainer.
- Keep the latest healthy checkpoint plus stage-boundary checkpoints.
- Large checkpoint binaries remain Git-ignored.
- Commit compact manifests: cumulative steps, config, model/update counters, hashes, evaluation metrics and reproduction/resume command.
- Push stable code and compact stage reports at each major stage boundary so progress is not trapped in one local session.

If execution must stop before 100M, stop at a checkpoint/stage boundary with an exact resume command and report completed steps. A partial but healthy campaign is acceptable; corrupted or uncheckpointed extra steps are not.

## 8. Health rejection gates

Immediately stop/rollback if any of these persist beyond a brief transient:

- NaN/Inf observations, rewards, logits, losses or optimizer state;
- action distribution collapses to a tiny repeated subset unrelated to outcomes;
- appended actions dominate simply because suppression was removed too quickly;
- frozen-Wisp evaluation collapses sharply while training reward rises (reward farming / self-play pathology);
- goals/concessions or reward components show obvious exploit behavior;
- checkpoint reload/resume parity fails.

Do not rescue a failing run by changing many variables simultaneously.
