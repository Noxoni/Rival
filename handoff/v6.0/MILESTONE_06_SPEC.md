# Milestone 06 Spec — First Serious Rival Training Campaign

## Objective

Use the verified Milestone 05 training foundation to produce and evaluate materially trained Rival checkpoints. This milestone is about learning, not trainer scaffolding.

Starting boundary: `4c9aa6f596b3231856107b3a1e59d9a7c4f663db`.

Production remains frozen Wisp until a trained checkpoint passes the promotion gate.

## Required work

### 1. Serious-training configuration

Create a versioned Milestone 06 config containing:

- 24 workers by default;
- `mechanics4` cadence;
- material PPO iteration/buffer/batch sizes;
- actor/critic learning rates;
- gamma/GAE values with cadence rationale;
- checkpoint interval;
- evaluation interval;
- stage boundaries;
- total campaign ceiling 100M agent-steps;
- random seeds;
- action-exploration-prior schedule/state;
- curriculum reset-source weights;
- reward-v2 weights.

All configuration must be serializable into checkpoints and reports.

### 2. Novel-action exploration control

Implement a PPO-consistent way to expose actions 90–157 gradually.

Requirements:

- calibrate initial exploration from measured natural observations;
- log appended probability mass and sampled action share;
- keep the exploration prior/checkpoint state reproducible;
- modify only at iteration/stage boundaries;
- eventually support zero external suppression;
- never silently change deployment logits during export.

### 3. Reward v2 + metrics

Build from RivalRewardV1 and `REWARD_AND_CURRICULUM.md`.

Every component must be independently logged and finite. Add mechanics/resource terms only at low weight after a preflight contribution audit.

### 4. Broad curriculum mutators

Implement majority-natural weighted reset selection plus broad randomized:

- aerial/wall possession family;
- recovery family;
- low-resource aerial family.

No fixed named-trick drill may dominate the distribution.

Log reset-family counts and percentages.

### 5. Training campaign harness

Create one resumable entry point capable of:

- fresh start from verified Wisp bootstrap;
- resume from full checkpoint;
- stage-aware config/prior changes;
- checkpoint every <=1M agent-steps or nearest safe equivalent;
- local compact metrics logging;
- automated health checks;
- deterministic evaluation invocation;
- clean interruption at checkpoint boundary.

Do not require WandB or an external cloud service. Optional external logging may be supported but local evidence is authoritative.

### 6. Evaluation harness

Implement headless frozen-Wisp evaluation at least every 5M steps.

Implement export + RLBot stage evaluation against installed Nexto/Wisp at major stage boundaries when the checkpoint is healthy enough to justify it.

Reuse the validated 5x RLBot match configuration.

### 7. Mechanics/recovery instrumentation

Record at least:

- legacy vs appended action share;
- appended action family distribution;
- airborne dodge/flip use;
- reset/dodge-resource acquisition and productive follow-up where measurable;
- wavedash-like event rate if reliably detectable;
- aerial boost spend and useful aerial distance/touches;
- boost remaining after aerial commitments;
- recovery time after lost possession/failed offense;
- concessions in recovery windows.

If one named event cannot be reliably detected, document the limitation rather than inventing a noisy label.

### 8. Stage reports

Commit compact stage reports at:

- preflight / 0M;
- ~5M;
- ~20M;
- ~50M if reached;
- final stopping point.

Each must contain:

- cumulative agent-steps;
- cumulative model updates;
- checkpoint hash/manifest;
- PPO metrics;
- reward component contribution audit;
- action distribution;
- curriculum distribution;
- headless Wisp evaluation;
- RLBot evaluation when run;
- health verdict;
- exact resume command.

### 9. Promotion

Follow `EVALUATION_AND_PROMOTION.md`.

Do not replace production unless a checkpoint passes the final promotion battery and deployment parity.

If no checkpoint earns promotion by the campaign ceiling, keep production unchanged and preserve the best healthy candidate for Milestone 07.

## Verification

Before the long campaign:

- existing 70+ Rival tests;
- Milestone 05 training tests;
- new v6 config serialization/checkpoint tests;
- action prefix/fingerprint parity;
- reward-v2 finiteness/contribution audit;
- weighted reset distribution smoke;
- novel-action prior/log-probability consistency test;
- resume-state parity test;
- headless Wisp evaluation smoke;
- deployment export smoke.

During campaign:

- hard fail on NaN/Inf;
- verify every saved checkpoint reloads before old checkpoints are pruned;
- verify compact reports parse;
- verify artifact hashes before committing manifests.

At completion:

- full test suites;
- lint/compile checks;
- `git diff --check`;
- no large checkpoints/datasets/secrets/absolute machine paths staged;
- remote readback after push.

## Completion states

Milestone 06 may complete as one of:

1. **Promoted** — a trained checkpoint passed the final gate and is safely integrated as an optional/new production policy with rollback to frozen Wisp.
2. **Healthy candidate** — material training improved meaningful metrics but final promotion remains ambiguous; production stays Wisp.
3. **Rejected/rollback** — serious training exposed a reward/exploration/self-play problem; best healthy checkpoint and exact evidence are retained.

All three are valid if honestly evidenced.
