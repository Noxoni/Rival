# Transfer Diagnostic Matrix

## Objective

Identify why a policy that improved in RocketSim headless evaluation regressed severely when exported to RLBot v5.

Milestone 06 evidence:

- RocketSim headless Wisp evaluation improved from 42-58 preflight to 59-41 at 20M.
- The 20M RLBot candidate then went 0-8, goals 27-56.
- No appended action (90-157) was selected deterministically in the RLBot battery.
- The candidate ran at roughly twice the historical Wisp decision frequency because candidate deployment forced tick skip 4.

Because appended mechanics actions were not selected in the real-game failure, first isolate the legacy 0-89 policy path and cadence.

## Test modes

Implement explicit diagnostic-only deployment controls. Normal production defaults must remain untouched.

### P0 — Frozen production Wisp

- original `POLICY.lt` + `SHARED_HEAD.lt`
- original 90 actions
- tick skip 8
- action delay 7

Use historical v4.1/M06 baseline evidence where valid, and run fresh controls only where needed for the matrix.

### Z8 — Zero-step reconstructed student, legacy-only, tick 8

- exact trainable reconstruction from frozen Wisp
- no PPO updates
- expanded output head may exist, but indices 90-157 must be hard-masked in this diagnostic
- tick skip 8
- action delay 7

This is the critical deployment parity control.

### Z4 — Zero-step reconstructed student, legacy-only, tick 4

Same actor and mask as Z8, but:

- tick skip 4
- action delay 3

This isolates cadence/action-repeat effects before learning.

### T8 — 20M trained actor, legacy-only, tick 8

- exact rejected 20M actor
- hard-mask indices 90-157
- tick skip 8
- action delay 7

This tests learned legacy-logit drift while removing appended actions and the four-tick deployment change.

### T4 — 20M trained actor, legacy-only, tick 4

- exact rejected 20M actor
- hard-mask indices 90-157
- tick skip 4
- action delay 3

This reproduces the cadence of the rejected M06 deployment while removing any ambiguity about appended action selection.

The existing 20M 158-action RLBot battery may be used as supporting T4 evidence because appended top-1 count was already zero, but still implement the explicit legacy-only mode for causal clarity.

## Same-observation inference parity

Before full matches, prove the model path on identical input tensors.

For a large sample of real live RLBot observations collected without changing gameplay, run in shadow:

1. frozen Wisp model;
2. zero-step reconstructed student;
3. 20M trained student.

Apply the same first-90 legal mask and record compact aggregates:

- max/mean absolute first-90 logit error: frozen vs zero-step;
- top-1 agreement;
- probability-distribution KL/JS where numerically stable;
- confidence and top-1/top-2 margin;
- trained-vs-frozen top-1 agreement;
- trained-vs-frozen action transition matrix/counts;
- any state regions where disagreement concentrates.

The zero-step reconstruction should be exact or extremely close on live observations. If it is not, stop and fix that before match comparisons.

## Live match matrix

Use full five-minute Soccar, accelerated with the already validated runner, normal kickoff countdowns, goal replays skipped, no production promotion.

For the diagnostic matrix, start with a balanced bounded battery per mode:

- 2 games vs installed Nexto (one blue, one orange)
- 2 games vs installed Wisp v2-75B (one blue, one orange)

That is 4 games per mode. P0 may reuse a fresh 4-game control or sufficiently comparable recent frozen-Wisp evidence, but prefer a fresh control if setup cost is small.

If a result is ambiguous, at most double the affected mode to 8 games. Do not turn this into a skill-ranking campaign.

Record:

- record and goal differential;
- touches and possession proxies;
- action frequency/distribution;
- confidence/margin;
- decision cadence;
- boost/recovery metrics already available;
- runtime health.

## Interpretation tree

### If Z8 ~= P0 but Z4 collapses

Primary cause: four-tick cadence/action-delay domain gap before PPO.

Do not continue the current four-tick warm-start strategy unchanged. Next architecture should preserve an 8-tick Wisp-compatible strategic policy while creating a separately trained high-frequency mechanics/recovery control path, or explicitly distill/retrain a 4-tick policy with strong teacher-consistency constraints before RL fine-tuning.

### If Z8 and Z4 are both healthy, but T8 and T4 collapse

Primary cause: learned legacy-logit drift / RL objective drift.

Next training should add teacher regularization, frozen/shared-backbone staging, historical-opponent diversity, and tighter live-boundary evaluations before large step budgets.

### If Z8 collapses despite same-observation parity

Primary suspicion: deployment timing/action-path mismatch outside model logits, including action delay, previous-action semantics, parser/mirroring, or model-output integration.

Instrument runtime state and resolve before training.

### If RocketSim-domain and live-domain policy behavior diverge strongly even for the same actor

Primary suspicion: observation/domain transfer mismatch. Use `OBSERVATION_DOMAIN_AUDIT.md` to identify which feature groups shift.

### If multiple modes fail differently

Report the interaction; do not force a single-cause story.

## Hard rule

No serious PPO continuation is authorized by v7.0. The output is a causal diagnosis and a concrete next architecture/training correction, not another trained candidate.
