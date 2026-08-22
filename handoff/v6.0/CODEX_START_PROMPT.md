# Codex Start Prompt — Rival Milestone 06

You are continuing development of **Rival**, a high-end offline/private Rocket League 1v1 bot.

Milestone 05 completed the RLGym/RocketSim training foundation at:

`4c9aa6f596b3231856107b3a1e59d9a7c4f663db`

Your task is now to run Rival's **first serious training campaign**. Do not answer with another high-level plan. Work directly in `Noxoni/Rival`, implement the required Milestone 06 extensions, train through coherent staged boundaries, evaluate checkpoints, commit compact evidence, and push stable work to `origin/main`.

## First: synchronize and verify

1. Confirm repository `Noxoni/Rival` and fetch `origin/main`.
2. Confirm completed M05 commit `4c9aa6f596b3231856107b3a1e59d9a7c4f663db` remains in history.
3. Read every file under `handoff/v6.0/`.
4. Read `docs/MILESTONE_05_RESULTS.md`, `training/README.md`, `training/ENVIRONMENT.md`, `training/configs/milestone05.json`, current `training/rival_training/*`, and M05 result manifests.
5. Verify the frozen Wisp artifacts and M05 action-table fingerprints before training.
6. Verify the ignored Wisp bootstrap/full smoke artifacts required for resume exist and match their committed hashes. If an ignored artifact is missing, reproduce it from committed instructions rather than guessing.
7. Leave production `.venv`, rejected runtime interventions, unrelated local files, old evidence and `stash@{0}` untouched.

## Central rule

**Train Rival. Do not add another tactical runtime rule around Wisp.**

RLGym/RocketSim is the training environment. Natural 1v1 is the majority training distribution. RLBot/real Rocket League is the external benchmark and promotion environment.

## Required implementation

Follow:

- `TRAINING_CAMPAIGN.md`
- `REWARD_AND_CURRICULUM.md`
- `EVALUATION_AND_PROMOTION.md`
- `MILESTONE_06_SPEC.md`

### 1. Preflight serious-training upgrade

Before spending millions of steps:

- create versioned M06 config;
- scale PPO from smoke-sized batches to a real training configuration;
- implement cadence-aware discount configuration;
- implement checkpointed appended-action exploration control;
- implement Reward V2 / mechanics and recovery instrumentation;
- implement majority-natural weighted broad curriculum mutators;
- implement automated headless frozen-Wisp evaluation;
- implement campaign/checkpoint/stage reporting.

Run the full preflight verification suite before long training.

### 2. Calibrate appended-action exploration

Actions `0..89` remain the exact Wisp-compatible prefix; `90..157` are the mechanics-capable additions.

The current Wisp bootstrap suppresses appended actions with bias `-12`. Do not zero all 68 biases at once.

Evaluate a small bounded set of candidate appended offsets over natural observations and choose the safest setting that gives measurable minority exploration while retaining the Wisp warm start.

All exploration-prior changes must be PPO-consistent, checkpointed, logged, and made only at iteration/stage boundaries.

### 3. Campaign ceiling and stage boundaries

Campaign ceiling: **100,000,000 agent-steps**.

Use these boundaries as a default structure:

- Stage A: through ~5M;
- Stage B: through ~20M;
- Stage C: through ~50M;
- Stage D: through <=100M if still healthy.

This is a ceiling, not a quota. Stop earlier if a checkpoint earns promotion, training becomes unhealthy, or a clean evidence-backed rollback is required.

Use 24 workers unless a measured stability problem requires fewer.

Checkpoint at least every 1M agent-steps or the nearest safe trainer interval. Large binaries remain Git-ignored; commit hashes/manifests and exact resume commands.

### 4. Natural majority curriculum

Natural self-play must remain the majority of experience.

Allow only minority broad randomized state families for:

- aerial/wall/ceiling-adjacent possession;
- recovery/awkward-surface states;
- low-resource aerial opportunities.

Do not make named mechanic drills or fixed coordinates the main environment.

### 5. Reward hierarchy

Goal/concede outcome remains dominant.

Keep independently logged possession, progress, boost-efficiency, recovery and mechanics/resource terms.

Small mechanics aids may include flip/dodge-resource acquisition, wavedash events and bounded aerial usefulness. They must be calibrated so reward farming is visible and cannot silently dominate game outcome.

Do not directly reward musty/breezi/Meeri/zap-dash/wall-dash actions by name.

### 6. Evaluation cadence

At least every 5M steps:

- deterministic headless evaluation against frozen Wisp;
- full action distribution and appended-action share;
- reward contribution audit;
- mechanics/recovery metrics;
- health verdict.

At major healthy boundaries around 20M, 50M, and final candidate:

- export actor;
- run actual RLBot full-game benchmark against installed Wisp and Nexto at validated ~5x effective game speed;
- balanced sides;
- commit compact results.

Do not run RLBot games for checkpoints that have obviously failed the headless health gate.

### 7. Preservation / rollback

Never delete:

- frozen Wisp bootstrap checkpoint;
- best previously verified healthy checkpoint;
- stage-boundary checkpoints until the next stage is verified healthy.

If reward rises while frozen-Wisp gameplay collapses, treat that as a failure signal. Roll back and make the smallest justified correction rather than tuning many variables simultaneously.

### 8. Production promotion

Do not replace production Rival merely because 100M steps completed.

A promotion candidate must pass the final 16-game RLBot battery described in `EVALUATION_AND_PROMOTION.md`, deployment parity, and aggregate health/mechanics review.

If the final evidence is ambiguous, leave production on frozen Wisp and retain the trained model as a candidate.

## Progress persistence

Push stable implementation before the long campaign begins.

At major stage boundaries, push compact reports/manifests so a long local training run cannot disappear with one session failure. Do not commit huge checkpoints or raw rollout data.

If execution limits prevent completing the full campaign in one run, stop only at a verified checkpoint/stage boundary, push all stable progress, and report the exact resume command and remaining authorized step budget. Do not leave an uncheckpointed training process as the only copy of progress.

## Required final report

Return and commit/push:

- final `origin/main` SHA;
- cumulative agent-steps and model updates;
- exact PPO/config values;
- curriculum/reset mix;
- appended-action prior schedule and action-share history;
- reward contribution history;
- headless Wisp results by stage;
- RLBot Nexto/Wisp results by stage where run;
- mechanics/recovery metric trends;
- best checkpoint manifest/hash and exact resume/deploy commands;
- whether outcome is `promoted`, `healthy candidate`, or `rejected/rollback`;
- full test/verification results;
- runtime warnings/errors and remaining limitations.

Prioritize actual training and measured gameplay improvement over additional infrastructure prose.
