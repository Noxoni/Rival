# Rival Milestone 03 — Challenge-Commitment Calibration

## Goal

Implement and evaluate Rival's first gameplay-policy correction: a conservative, commitment-sensitive delay/re-ranking mechanism that reduces premature possession-releasing jumps under fake/aborted pressure while preserving correct responses to genuine challenges.

This milestone changes gameplay only in this one narrow domain.

## Required starting state

Completed Milestone 02 / current evidence baseline:

`e7b68c6e33faf6fc644a3fc9a07e811d43d2918e`

Frozen original Wisp-equivalent gameplay baseline:

`4f2b21c00e2fcb7108ab1006fd950b066fbd0484`

Before implementation, verify local `main` and `origin/main`, inspect any newer commits, and preserve legitimate handoff commits. Do not reset away `handoff/v3.0/` merely because it is newer than the evidence baseline.

---

# A. Preserve a baseline path

The existing Wisp-derived inference path must remain exactly selectable for comparison.

Provide a challenge-calibration mode with semantics equivalent to:

```text
off
observe
intervene
```

- `off`: baseline Wisp selection, no treatment.
- `observe`: compute all v3 features and hypothetical action but return baseline action.
- `intervene`: apply the accepted narrow re-ranking rule.

Add tests proving `off` is selection-equivalent to the pre-v3 path for existing fixture/smoke observations.

Do not alter model weights, observation-vector semantics, legal action masking, tick skip, or action delay as part of this experiment.

---

# B. Implement challenge commitment tracking

Create a testable strategy/analysis component implementing the design in `CHALLENGE_CALIBRATION_DESIGN.md`.

Minimum responsibilities:

- instantaneous physical/input features;
- projected ball-intersection / closest-approach feature;
- short bounded temporal history;
- commitment score and discrete low/ambiguous/high state;
- pressure-present output;
- clear abort detection;
- reset handling.

The estimator must not mutate controller output itself.

Keep feature calculations cheap enough for the existing 8-tick policy cadence.

---

# C. Implement the possession/release gate

Build a separate gate that answers whether v3 is allowed to consider a treatment.

Minimum gate concepts:

- Rival plausibly possesses/can continue the ball;
- Rival is grounded;
- apparent opponent pressure is present;
- commitment is ambiguous rather than clearly high;
- baseline Wisp selection is a grounded release-sensitive jump/dodge initiation;
- suitable legal non-jump alternative exists;
- no safety exclusion applies.

Telemetry must state why the gate passed or failed.

Do not turn the gate into a full tactical planner.

---

# D. Re-rank existing Wisp actions

When treatment is eligible:

1. retain the original baseline action index and logits;
2. identify the highest-logit legal continuation candidate from Wisp's existing discrete action table;
3. compare baseline vs continuation preference;
4. choose the continuation only if the configured conservative gap/confidence and safety conditions pass;
5. track a strict per-challenge deferral budget;
6. reassess commitment next policy tick.

Start with a one-policy-tick max deferral. Only evaluate two ticks if required by controlled evidence.

Never synthesize a new controller action.

Never permanently mask jump actions globally.

---

# E. Policy-decision representation

Extend the policy/control seam so telemetry and tests can distinguish:

- model/baseline action;
- final Rival action;
- hypothetical action in observe mode;
- whether treatment changed the action.

Do not overwrite the original `PolicyDecision` semantics in a way that loses baseline evidence. A wrapper/result object or explicit baseline/final fields is preferred.

The final control sent to RLBot must always be auditable back to one legal discrete action index.

---

# F. Telemetry schema v3

Extend session telemetry with challenge-treatment fields described in `CHALLENGE_CALIBRATION_DESIGN.md`.

Session lifecycle records must include:

- challenge mode;
- parameter set/version;
- code commit;
- model hashes;
- probe/natural-match metadata;
- whether treatment is actually enabled.

Decision records must include baseline/final action identity and intervention explanation.

Raw telemetry remains Git-ignored. Commit compact results, manifests, and curated fixtures only.

---

# G. Improve the challenge event metric

Milestone 02's controlled `release-like` signal intentionally counted any jump/material pitch/boost. That was appropriate for broad candidate discovery but is too coarse for evaluating treatment.

Add a v3 evaluation detector centered on actual **premature grounded jump/dodge release** under low/ambiguous commitment.

Keep the v2 detector/results intact for historical comparison; do not silently redefine old output.

Version the new detector/schema.

---

# H. Controlled probes

Reuse and extend the existing controlled harness, not a second unrelated test framework.

Required behaviors remain:

- `true_commit`
- `boost_then_brake`
- `boost_then_veer`
- `jump_fake`
- `delayed_challenge`

Run paired baseline/treatment comparisons per `EXPERIMENT_MATRIX.md`.

If useful, add held-out variations in separation, abort timing, challenger speed, lateral offset, and boost, but keep the experiment matrix bounded and deterministic.

Controlled probe count is not subject to the natural-match limit, but avoid meaningless brute-force volume.

---

# I. Natural validation

Only after controlled acceptance gates pass.

Milestone 03 may launch at most six additional natural matches total:

- maximum 3 vs installed Nexto;
- maximum 3 vs installed Wisp v2-75B.

Use the installed references at the hashes already captured in `reference_manifests/v1/MANIFEST.json` and do not modify them.

Do not claim superiority from this small sample.

---

# J. Tests and verification

At minimum add/maintain tests for:

- commitment feature geometry;
- projected closest approach edge cases;
- temporal trend/abort behavior;
- history reset boundaries;
- jump fake is not automatically classified as high commitment from jump input alone;
- boost input alone is not sufficient for high commitment;
- true intersecting trajectory reaches high commitment;
- gate exclusions;
- legal continuation selection;
- max-logit-gap behavior;
- one/two-tick deferral budget;
- `off` mode exact baseline parity;
- `observe` mode no control change;
- `intervene` action always legal and from Wisp action table;
- telemetry v3 serialization;
- existing test suite regression.

Run pytest, targeted/full Ruff as appropriate, compileall, model smoke, policy-freeze/baseline parity checks, and `git diff --check`.

Do not hide inherited untouched Wisp lint debt by editing unrelated code.

---

# K. Deliverables

Commit and push:

- challenge estimator/tracker;
- intervention gate/re-ranker;
- configuration/modes;
- schema-v3 telemetry changes;
- v3 event/evaluation metrics;
- new tests;
- compact paired controlled results;
- curated regression fixtures where useful;
- natural-match regression summary if controlled gates passed;
- `docs/MILESTONE_03_RESULTS.md`;
- `CHANGELOG.md` and README updates;
- exact accepted/rejected parameter set.

Do not commit the hundreds of MiB of raw JSONL telemetry.

---

# L. Acceptance boundary

Milestone 03 is successful in either of two legitimate ways:

## Accepted gameplay change

Evidence supports the treatment, controlled gates pass, natural sanity check exposes no clear regression, and the normal Rival Dev configuration enables the accepted challenge calibration.

If accepted, update the development identity/version metadata consistently, preferably to an agent id such as:

`noxoni/rival/dev-v3`

while preserving an easy baseline-off mode for comparison.

## Rejected experiment

The implementation is technically correct but the treatment does not pass evidence gates. Leave it disabled by default, preserve the results, and report the failure clearly.

Do not add unrelated compensating rules merely to force an accepted result.