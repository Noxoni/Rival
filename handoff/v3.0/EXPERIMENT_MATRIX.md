# Milestone 03 Experiment Matrix

## Purpose

Milestone 03 is accepted only if the challenge-calibration intervention improves the targeted fake-pressure behavior **without creating a meaningful regression against genuine commits**.

Do not judge the experiment from one highlight, one scoreline, or the broad Milestone 02 `release-like` detector alone.

---

# 1. Preserve baseline and treatment modes

Every test/probe that can exercise the intervention should support at least:

- `off`: exact pre-v3 action selection;
- `observe`: compute what v3 would do but return baseline action;
- `intervene`: apply v3 treatment.

The same scenario definitions must be runnable in baseline and treatment mode.

Record the exact commit/config/mode for every evidence session.

---

# 2. Offline fixture tests

Use the existing curated fixture:

`fixtures/evidence/apparent_vs_actual_challenge__appa-1662387304b0.json`

and curate small additional fixtures from the existing local Milestone 02 raw corpus for the highest-ranked natural Nexto challenge candidates, including at minimum:

- `appa-2853d7379f43`
- `appa-1d8d681ee3a9`
- one additional top-ranked natural Nexto event if the raw source remains locally available.

If a raw source is unavailable, do not fabricate it. Use the committed event envelope and controlled fixture instead.

Offline fixture tests should verify:

- estimator outputs are finite and deterministic;
- reset/history handling is deterministic;
- treatment never selects an illegal action;
- treatment action is always an existing Wisp discrete action;
- `off` mode exactly reproduces baseline argmax;
- `observe` returns the baseline action while logging the hypothetical treatment;
- `intervene` changes action only when the documented gate is satisfied;
- one/two-tick deferral budget is enforced;
- high commitment disables deferral promptly.

These tests are structural evidence, not proof of better gameplay.

---

# 3. Controlled A/B challenge suite

Reuse the Milestone 02 controlled opponent and state-setting harness.

Required ground-truth behaviors:

1. `true_commit`
2. `boost_then_brake`
3. `boost_then_veer`
4. `jump_fake`
5. `delayed_challenge`

Run the same initial-state parameterizations under baseline and treatment.

Prefer deterministic paired comparisons. Preserve the original five repetitions/case family and add held-out parameter variants only if needed to prevent tuning to one exact setup.

Do not count startup/reset decisions as challenge outcomes.

## Primary treatment metrics

Add narrower metrics than Milestone 02's broad release-like signal.

At minimum measure:

### `premature_release_jump`

A grounded Rival jump/dodge initiation during a possession/pressure episode while commitment is still below the high threshold and before the opponent reaches the configured unavoidable-intercept boundary.

Document the exact definition.

### `release_delay_ms`

Time between baseline/hypothetical Wisp release selection and the final Rival release when treatment defers.

### `commitment_at_release`

Estimator score/state when Rival actually releases/jumps.

### touch outcome

- next touch self/opponent/none;
- touch sequence in a bounded window;
- time to next self/opponent touch where available.

### possession/separation proxy

- Rival-ball distance growth;
- ETA advantage change;
- time Rival remains the more likely next toucher;
- any existing control proxy that remains stable across baseline/treatment.

### intervention exposure

- eligible decisions;
- interventions applied;
- false/unsafe intervention exclusions;
- average/max consecutive deferrals.

---

# 4. Controlled pass/fail gates

Do not enable v3 by default merely because the code works.

The treatment must satisfy all of the following before natural-match validation:

## Baseline integrity

- `off` mode passes the policy-freeze/selection-equivalence checks.
- model hashes remain unchanged.
- no illegal/controller-synthetic treatment action exists.

## Fake-pressure improvement

Across the four fake/non-immediate-commit behaviors together:

- `premature_release_jump` rate should materially decline versus paired baseline; target at least a **50% relative reduction** unless the refined metric shows the original broad detector substantially overcounted the behavior;
- treatment must not reduce Rival's self-next-touch rate versus paired baseline by more than one case in the original 20-case set;
- treatment should not produce a materially worse mean/median ball-separation or ETA-control outcome.

If the refined metric invalidates the 50% target, document why with paired case data and use an evidence-backed replacement threshold. Do not quietly move the goalposts.

## True-commit protection

For `true_commit` cases:

- no indefinite delay;
- max deferral remains within configured budget;
- no systematic loss of next-touch/control outcome relative to baseline;
- no new obvious open-net/defensive failures in the bounded controlled windows;
- commitment should reach `high` before or by the point where treatment stops deferring in clearly committed trajectories.

A treatment that wins fake probes by simply refusing to jump is a failure.

---

# 5. Parameter selection

Do not hand-tune dozens of thresholds against the same five cases.

Keep the parameter surface small. Suggested tunables:

- low/high commitment threshold;
- pressure distance/ETA boundary;
- projected-miss-distance reference;
- max baseline-vs-continuation logit gap;
- max deferral policy ticks (start at 1; test 2 only if necessary);
- possession/control gate references.

Use coarse candidate values and paired controlled outcomes. Record every attempted accepted/rejected configuration in a compact experiment results file.

Do not run a giant unconstrained search and overfit the probe suite.

---

# 6. Natural-match regression

Only after controlled gates pass.

Run **at most six new natural matches total** for Milestone 03:

- up to 3 Rival-treatment vs installed Nexto;
- up to 3 Rival-treatment vs installed Wisp v2-75B.

Alternate side where practical and keep standard Soccar settings comparable to Milestone 02.

Do not rerun the Milestone 02 baseline matches. Compare against their committed summaries and raw local corpus where available.

Natural matches are a regression/sanity check, not a statistically sufficient skill ranking.

Track at minimum:

- score/outcome for context;
- challenge-candidate rate per 1,000 policy decisions;
- premature-release-jump candidate rate per 1,000 decisions;
- intervention count/rate;
- intervention outcomes by next touch;
- goals conceded shortly after intervention if derivable;
- changes in policy action distribution around challenge windows;
- crashes/NaNs/runtime warnings.

Pay special attention to Nexto because the highest-ranked natural Milestone 02 challenge-abort windows were concentrated there.

---

# 7. Acceptance decision

## Accept

Enable challenge calibration in the normal Rival Dev v3 configuration only if:

- controlled fake-pressure metrics improve;
- true-commit controls are protected;
- structural tests pass;
- natural regression does not expose a clear new failure mode;
- every intervention remains explainable from telemetry.

## Reject / keep experimental

If those conditions fail:

- leave the feature disabled by default;
- preserve the implementation/experiment results if useful;
- report which condition failed;
- keep `off` baseline exact;
- do not compensate by adding unrelated rules in the same milestone.

A clean rejected experiment is more useful than an unmeasured behavior change.