# Codex start prompt — Rival Milestone 10

You are executing Milestone 10 in the canonical repository `Noxoni/Rival`.

## Objective

Start the first serious sustained training campaign for the completed Milestone 09 scratch Rival policy.

Milestone 09 already proved the architecture. Do **not** turn M10 back into an architecture/research milestone.

The primary task is:

> Resume the exact final M09 Gate 13 scratch checkpoint and train it for up to 100 **additional** simulated game-hours, using the same policy, observation, native action contract, reward, PPO configuration, worker count, and reset curriculum.

## 1. Establish authority and preserve history

1. Fetch `origin/main` and verify the current history contains completed M09 commit `824e328f6bbf4fe9a47b8e54706b5fcf645fd409` plus this `handoff/v10.0/` package.
2. Read completely:
   - `docs/MILESTONE_09_RESULTS.md`;
   - `handoff/v10.0/README.md`;
   - `handoff/v10.0/MILESTONE_10_SPEC.md`;
   - `handoff/v10.0/EVALUATION_PROTOCOL.md`;
   - `handoff/v10.0/M10_CAMPAIGN.json`;
   - this file.
3. Preserve all M09 evidence, checkpoints, ignored raw artifacts, production Wisp files, and the historical stash.
4. Production Rival remains frozen Wisp. Do not promote a scratch checkpoint.

## 2. Verify the resume checkpoint before changing code

Locate the final M09 Gate 13 checkpoint identified by the M09 report/manifest.

It must match:

- cumulative steps: `1,680,214`;
- simulated hours: `1.9446921296`;
- actor SHA-256: `12770f082c6cbe1fbab8809580dc775d1d78071825eb9481df4a16d9ee85fbe5`;
- actor, critic, both optimizers, trainer counters present;
- fresh reload exact as documented by M09.

If this ignored local checkpoint is missing or corrupt, **stop and report**. Do not substitute a random initialization, a Gate 12 export, or a reconstructed checkpoint without explicit authorization.

## 3. Freeze the proven foundation

M10 does not authorize changing:

- `RivalPolicyV1`;
- `RivalCriticV1`;
- `RivalObsV1` or its schema;
- `RivalActionV1` or its schema;
- canonical train/deploy state semantics;
- native 120-Hz policy cadence;
- one-tick action transport;
- `RivalScratchRewardV1` / its cadence-safe schedule;
- PPO hyperparameters from the M09 pilot;
- the 70/10/8/8/4 reset curriculum;
- observation standardization (`false`);
- 56-worker selection.

No Wisp actor/trunk. No behavior cloning. No action lookup table. No RepeatAction. No state-dependent action mask. No named mechanics macros/rewards.

If you believe one of these is genuinely broken, stop at a clean boundary and produce evidence. Do not silently redesign it inside M10.

## 4. Implement only the minimal long-campaign plumbing

Reuse the completed M09 scratch learner, environment, checkpoint format, evaluator, exporter, and RLViser seam.

Add only what is required to make sustained training recoverable and auditable, such as:

- a committed `training/configs/milestone10.json` derived from `handoff/v10.0/M10_CAMPAIGN.json`;
- a resume-capable M10 campaign entry point if the M09 runner is pilot-ceiling-specific;
- rolling recovery checkpoint retention so a host failure does not lose much work without retaining hundreds of immutable checkpoints;
- immutable boundary checkpoint/report generation;
- compact boundary evaluation orchestration.

Do not rewrite working M09 infrastructure for style.

Before starting the long run, perform a focused preflight only:

- config/version/hash validation;
- exact checkpoint fresh reload;
- one short finite environment/rollout smoke;
- confirm 56 workers initialize cleanly;
- confirm production defaults remain frozen Wisp.

Then **start training**. Do not repeat M09 Gates 0–14.

## 5. Campaign budget

M10 authorizes:

- `100.0` additional simulated game-hours;
- `86,400,000` additional agent-steps;
- nominal cumulative target `88,080,214` agent-steps;
- nominal cumulative simulated time `101.9446921296` hours.

Use complete PPO iterations and record exact achieved values. Do not exceed the authorized budget by more than normal final-iteration alignment.

## 6. Required boundary checkpoints

Evaluate and preserve immutable checkpoints at approximately:

- +5 simulated hours;
- +10 hours;
- +25 hours;
- +50 hours;
- +100 hours.

At every boundary:

1. save the complete checkpoint;
2. fresh-process reload it and verify deterministic parity;
3. run the fixed deterministic M09-compatible behavior evaluation;
4. record action/exploration and PPO health metrics from `EVALUATION_PROTOCOL.md`;
5. generate a compact machine-readable boundary report;
6. commit/push compact code/config/report updates so progress is recoverable remotely.

Large checkpoints/raw rollout data remain Git-ignored and are referenced by path/hash/size in committed manifests.

Do not save every iteration forever. Maintain rolling recovery checkpoints plus immutable milestone boundaries.

## 7. Frozen snapshot comparisons

At +10 h and later, run a bounded headless comparison against frozen prior checkpoints as described in `EVALUATION_PROTOCOL.md`.

This is for measuring real progress despite moving self-play. Keep it small; do not build a league framework during M10.

## 8. Native RLBot checks

At approximately +25 h and +100 h:

- export the candidate;
- run the native-1x scratch RLBot path;
- verify model/schema identity, legal controller outputs, 120-Hz timing health, finite inference, and no sustained missed ticks;
- optionally include a small Wisp/Nexto result set for context.

Winning/losing is not the transfer pass criterion.

## 9. Stop rules

Stop and push the nearest clean recoverable boundary for technical failures listed in `MILESTONE_10_SPEC.md`.

Do **not** stop just because scratch gameplay is weak, ugly, loses to Wisp/Nexto, or lacks advanced mechanics early.

Do not tune the reward/PPO/action/observation stack mid-campaign simply to make a metric improve. If evidence supports a future change, record it as a recommendation for M11.

## 10. Final result

At the final healthy boundary, create:

`docs/MILESTONE_10_RESULTS.md`

and compact machine-readable final evidence under:

`training/results/milestone10/`

The final report must state:

- exact starting/final checkpoint identity;
- additional and cumulative agent-steps/simulated hours;
- PPO iterations/model updates;
- throughput;
- fixed evaluation learning curves across all boundaries;
- analog/button exploration trends;
- jump/dodge/aerial/contact/recovery/goal trends;
- frozen-snapshot comparison results;
- native RLBot transfer health;
- reward-component behavior/exploit audit;
- any emergent mechanic-like behavior observed, without overclaiming named mechanics;
- whether the exact M09 foundation remained unchanged;
- whether continued long training is supported;
- recommended M11 direction.

Run the normal production/scratch verification needed for changed files and final repository hygiene, then push and verify remote readback.

M10 does **not** authorize production promotion.

If execution must end before +100 h, push the latest coherent boundary and report exactly what completed and what remains. Do not leave important progress only in the local workspace.
