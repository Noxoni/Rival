# Codex Start Prompt — Rival Milestone 03

You are continuing implementation of the high-end offline Rocket League 1v1 bot **Rival** for RLBot v5.

Do not answer with another high-level plan. Work directly in `Noxoni/Rival`, implement and evaluate Milestone 03, commit stable work, and push it to `origin/main`.

## Required starting point

Canonical repository:

`https://github.com/Noxoni/Rival`

Completed Milestone 02 evidence baseline:

`e7b68c6e33faf6fc644a3fc9a07e811d43d2918e`

Frozen original Wisp-equivalent gameplay baseline:

`4f2b21c00e2fcb7108ab1006fd950b066fbd0484`

At the beginning:

1. fetch `origin/main`;
2. confirm the working repository is `Noxoni/Rival`;
3. inspect commits after `e7b68c6e33faf6fc644a3fc9a07e811d43d2918e` and preserve legitimate `handoff/v3.0/` documentation commits;
4. read every file under `handoff/v3.0/` before modifying implementation code;
5. read `docs/MILESTONE_02_RESULTS.md`, `evidence/results/v2/candidate_events.md`, and the existing challenge fixture;
6. verify the installed Wisp/Nexto reference hashes against the committed manifests before any new live comparison;
7. leave the user's untracked `bot.7z` and unrelated local files untouched.

Do not reset or squash historical milestone/handoff commits.

---

# Central rule

**Change one behavior only: challenge-commitment calibration.**

Do not fix boost greed, aerial-resource behavior, unrelated lint debt, model architecture, observations, tick skip, action delay, or anything else in this milestone unless a blocking correctness bug prevents the requested experiment.

The intended treatment is a conservative policy re-ranking layer, not a hand-coded replacement bot.

---

# Evidence basis

Milestone 02 produced:

- six natural baseline matches: three vs Nexto, three vs Wisp v2-75B;
- 25 controlled fake-challenge cases, five each of `true_commit`, `boost_then_brake`, `boost_then_veer`, `jump_fake`, and `delayed_challenge`;
- 35,230 policy decisions across 12 primary sessions;
- 870 `apparent_vs_actual_challenge` candidates;
- highest-ranked natural Nexto windows where apparent closing pressure later aborted, Rival jumped, and Nexto took the next touch.

Important caveat:

All 20 fake/non-immediate-commit controlled cases produced the old broad release-like signal, but Rival still took the next touch in 17/20. Therefore the old signal is not itself proof that every release was wrong.

Do not implement `never jump on a fake challenge`, `if closing drops then keep ball`, or another simplistic rule.

---

# Required implementation

Follow `CHALLENGE_CALIBRATION_DESIGN.md` and `MILESTONE_03_SPEC.md`.

The implementation must contain these logical pieces, even if exact file names differ:

## 1. Challenge commitment estimator/tracker

Create a cheap, deterministic, testable estimator using observable RLBot state and short history.

At minimum incorporate useful subsets of:

- opponent/Rival distances and ETAs to ball;
- opponent-ball and opponent-Rival closing speed;
- opponent forward alignment and velocity alignment to the moving ball;
- short-horizon projected closest approach / miss distance;
- opponent last input: throttle, boost, steer, jump, handbrake;
- opponent airborne/jump/dodge state;
- temporal trends and explicit abort evidence.

Return a score plus low/ambiguous/high state, pressure flag, and explainable components.

A jump input alone is not high commitment. Boost alone is not high commitment.

Reset tracker state across goals/kickoffs/time rewinds/demolitions or other state discontinuities.

## 2. Possession/intervention gate

Treatment may only be considered when Rival plausibly has/continues ground possession, apparent opponent pressure exists, commitment is ambiguous, the baseline Wisp choice is a genuinely release-sensitive grounded jump/dodge initiation, and safety exclusions permit a delay.

Do not use Milestone 02's broad `boost|pitch|jump = release-like` detector as the control gate.

## 3. Wisp-action re-ranking

Keep the baseline Wisp inference and legal mask.

When treatment is eligible:

- preserve the baseline action;
- find the highest-logit suitable legal non-jump continuation action already in Wisp's discrete action table;
- compare the policy preference gap;
- only use the continuation when conservative gap/safety conditions pass;
- reassess on the next policy decision.

Start with a maximum **one-policy-tick** deferral (~66.7 ms). Only test two ticks if controlled evidence says one is insufficient.

Never synthesize controller values. Final actions must map to existing legal action indices.

## 4. Modes

Implement modes equivalent to:

```text
off
observe
intervene
```

Suggested surface:

`RIVAL_CHALLENGE_CALIBRATION_MODE=off|observe|intervene`

`off` must reproduce the exact pre-v3 selection path.

`observe` must calculate/log hypothetical treatment but return baseline action.

`intervene` applies treatment.

## 5. Telemetry v3

Record both original Wisp and final Rival action plus challenge estimate, gate, intervention reason, logit gap, continuation candidate, history/trend components, safety exclusion, and deferral budget.

Do not enable full logits by default.

---

# Experiment order

Do not jump straight into natural matches.

## Stage 1 — static/unit verification

Implement tests from `MILESTONE_03_SPEC.md`.

Prove:

- off-mode baseline parity;
- observe-mode no output change;
- treatment selects only legal Wisp actions;
- tracker reset correctness;
- true trajectory can reach high commitment;
- jump/boost alone do not force high commitment;
- deferral budget is bounded.

Run the complete existing test suite as well.

## Stage 2 — offline evidence/fixtures

Use the committed controlled jump-fake fixture and curate compact fixtures for the top natural Nexto challenge candidates from the local Milestone 02 raw corpus if those source files remain available.

Do not fabricate missing raw evidence.

Use these to validate estimator/gate behavior, not to claim causal improvement.

## Stage 3 — paired controlled A/B

Run the fake-challenge suite in baseline and treatment mode using matching initial conditions.

Use the refined v3 metrics in `EXPERIMENT_MATRIX.md`.

Tune only a small named parameter set. Preserve a compact record of attempted parameter sets and outcomes.

The treatment must pass the fake-pressure and true-commit gates before proceeding.

If it fails, stop gameplay validation, leave the feature disabled by default, document the rejected experiment, and still push the technically complete work/results.

## Stage 4 — bounded natural validation

Only if controlled gates pass.

Launch at most **six** new natural matches total:

- maximum 3 vs installed Nexto;
- maximum 3 vs installed Wisp v2-75B.

Do not launch extra matches because results are inconvenient.

Compare challenge/intervention event rates to the committed Milestone 02 baseline. Treat scores as context, not a statistical skill ranking.

---

# Acceptance behavior

If evidence passes:

- enable the accepted challenge calibration in the normal Rival Dev configuration;
- keep a one-switch baseline-off mode;
- update development identity/version metadata consistently, preferably `noxoni/rival/dev-v3`;
- commit `docs/MILESTONE_03_RESULTS.md` with exact parameters and evidence.

If evidence fails:

- default the challenge calibration to off;
- preserve the experiment and results;
- clearly state which acceptance criterion failed;
- do not add unrelated compensating rules.

A rejected experiment is an acceptable Milestone 03 result.

---

# Required verification before final push

At minimum run:

- full pytest suite;
- Ruff over all Rival-authored/modified Python files;
- compileall over relevant runtime/tools/tests/probes;
- model/hash smoke verification;
- baseline off-mode parity/policy-freeze verification;
- controlled A/B result validation;
- telemetry/result/fixture JSON parse checks;
- `git diff --check`;
- remote readback after push.

If a known inherited Wisp lint finding is untouched, record it rather than mixing cleanup into this milestone.

---

# End-of-run report

Return and commit/push a concise but complete result containing:

- commit SHA(s);
- whether challenge calibration was accepted or rejected;
- exact accepted/rejected parameter set(s);
- files changed;
- test results;
- paired controlled baseline/treatment results;
- true-commit regression results;
- natural match results if Stage 4 ran;
- intervention counts and outcomes;
- new curated fixture IDs;
- model/reference hash verification;
- runtime warnings/errors;
- final `origin/main` SHA;
- next smallest evidence-backed target.

Do the work in this run. Prioritize measured behavior over prose.