# Codex Start Prompt — Rival Milestone 04

You are continuing implementation of the high-end offline Rocket League 1v1 bot **Rival** for RLBot v5.

Do not answer with another high-level plan. Work directly in `Noxoni/Rival`, implement Milestone 04, run the required deterministic-pairing evidence, commit stable work, and push it to `origin/main`.

## Required starting point

Canonical repository:

`https://github.com/Noxoni/Rival`

Expected completed Milestone 03 commit:

`e4cc175a4259202d5cc7ee437abef224b731354f`

Frozen original Wisp-equivalent gameplay baseline:

`4f2b21c00e2fcb7108ab1006fd950b066fbd0484`

At the beginning:

1. fetch `origin/main`;
2. inspect commits after `e4cc175a4259202d5cc7ee437abef224b731354f` and preserve legitimate `handoff/v4.0/` documentation commits;
3. read every file under `handoff/v4.0/` before changing implementation code;
4. read `docs/MILESTONE_03_RESULTS.md` and `evidence/results/v3/milestone_03_decision.json`;
5. verify the installed Nexto/Wisp reference hashes and frozen model hashes before new live evidence;
6. keep the challenge calibration default `off` unless this milestone explicitly reaches and passes the treatment gate;
7. leave unrelated local files and ignored evidence untouched.

Do not reset/squash historical milestone commits.

---

# Central rule

**Fix paired-test reproducibility before tuning gameplay.**

Milestone 03 produced different baseline/treatment trajectories even when the treatment applied zero interventions. Therefore those outcome differences were not causal.

Do not loosen challenge thresholds or add tactical rules to compensate.

---

# First investigate/confirm the nondeterminism sources

The inherited Wisp observation builder currently uses process-global `random.shuffle()` for teammate/opponent observation buffers every policy decision. In a 1v1 this can move the one real opponent among padded opponent slots across otherwise equivalent runs.

The observation also includes `player.prev_action`, and Rival carries additional runtime state (`prev_actions`, old/new action, tick-window state, policy tick, RocketSim adapter state, challenge estimator/controller history, etc.) across controlled state-setting cases unless a reset path fires.

Use tests/instrumentation to confirm which of these materially affect paired drift. Do not assume only one source exists.

---

# Required implementation

Follow `DETERMINISTIC_PAIRING_SPEC.md` exactly.

At minimum implement:

## A. Injectable/resettable observation RNG

- Default production/off-mode behavior must preserve the current Wisp shuffle path.
- Controlled deterministic mode gets a dedicated seeded RNG.
- Every case/repetition gets a stable named seed.
- The same case uses the same seed in `off`, `observe`, and any later `intervene` run.
- Different repetitions deliberately use different seeds.

Do not globally replace Wisp's shuffle with fixed ordering.

## B. Full controlled-case reset

Create one explicit internal reset routine covering all decision/history state that can leak between controlled state-setting cases.

This must include previous-action history because previous action is part of the Wisp observation.

Use an explicit test-only boundary if practical. If no clean dynamic case-ID channel exists, implement a controlled-mode state-discontinuity/teleport detector with a safe threshold and anti-repeat/cooldown behavior.

Record reset reason, case epoch, and case seed.

## C. Exact model-input fingerprint

During controlled deterministic testing, hash the **actual float32 observation tensor bytes passed into Wisp** plus the effective legal mask.

Record pair/case identity and enough trace information to compare:

- observation hash;
- legal-mask hash;
- Wisp baseline action index;
- final action index;
- top candidates/logits;
- previous action;
- case-local decision index;
- case seed/epoch;
- relevant packet/state fingerprint.

Do not enable full observation dumps in normal telemetry.

## D. Paired runner

Create explicit matching `off` and `observe` case pairs from identical desired state, opponent behavior, parameters, seed, and model/runtime version.

`observe` does not alter output. Therefore paired traces are expected to remain equivalent.

Before testing treatment, demonstrate for the chosen controlled scenario that corresponding `off` and `observe` runs match on model-input/action traces up to the relevant exposure.

Target 5/5 paired repetitions.

## E. Reproducible release-sensitive exposure

The refined Milestone 03 detector found only one baseline release-sensitive fake-pressure event in 20 broad cases. Use a bounded search around the existing fake-challenge setup to find a controlled scenario/seed where the refined grounded release-sensitive Wisp action itself is repeatable.

Bound the search to roughly 200 state/seed variants maximum. Stop early when a strong candidate is found.

Do not change the detector, model, observation semantics, or legal mask merely to manufacture the event.

Commit the final small reproducible fixture/parameters if found.

---

# Evidence stages

## Stage 1 — static/unit tests

Add tests required by `DETERMINISTIC_PAIRING_SPEC.md`, including:

- default Wisp shuffle path preserved;
- seeded shuffle reproducible;
- same seed/reset -> same observation ordering;
- different seed can produce a different valid permutation;
- full case reset clears previous action/tracker/runtime history;
- state-discontinuity threshold does not fire under plausible normal motion;
- repeated setter packets do not cause endless resets;
- off/observe output parity;
- observation fingerprint hashes the actual model input tensor.

Run the full existing suite.

## Stage 2 — 1x deterministic pairing

Use 1x only to establish the clean reference pairing.

Run paired off-vs-observe controlled cases and prove trace equivalence.

If off and observe still diverge before any treatment point, investigate and fix the harness. Do not tune challenge calibration.

## Stage 3 — exposure search

Search the bounded controlled-state/seed space for a repeatable refined release-sensitive baseline event.

Once found, rerun it under matching off and observe conditions and prove the same exposure occurs at the corresponding decision.

If no reproducible exposure is found within budget, stop gameplay work, document the result, and push the harness improvements.

## Stage 4 — accelerated reproducibility

Follow `SPEED_POLICY.md`.

Test the same synchronized paired fixture at 5x using direct desired match game speed.

Do **not** require `packet.match_info.game_speed` to echo 5.0; Milestone 03 showed that field can stay stale while simulated time genuinely accelerates.

Accept 5x for controlled testing if:

- effective game-sec/wall-sec is near 5x;
- bots remain responsive;
- telemetry is valid;
- action distribution is non-degenerate;
- paired off/observe reproducibility still passes.

If 5x breaks pairing, step down through 4x/3x/2x and select the fastest reproducible speed.

## Stage 5 — optional treatment

Only if deterministic exposure exists.

You may test **one** new prospective challenge-calibration parameter version against the frozen reproducible paired fixture.

Requirements:

- new version name, not either rejected m03 parameter name;
- actual eligible decision(s) must occur;
- at least one actual intervention must occur or the treatment was not tested;
- record the exact first baseline-vs-final action divergence;
- true-commit controls remain protected;
- one-policy-tick deferral bound remains unless a separate evidence gate justifies otherwise.

If treatment fails, leave default mode off and document the rejection.

Do not run natural acceptance matches in Milestone 04 unless a separate explicitly documented gate proves the controlled treatment is valid and you can justify consuming the preserved six-match budget. Default expectation: **zero natural acceptance matches in v4**.

---

# Concurrency

After deterministic pairing is stable, `SPEED_POLICY.md` authorizes one new bounded two-lane capability test because the first Milestone 03 attempt failed specifically from a port race and the harness was changed afterward.

Verify unique listener ports, distinct server processes, two distinct Rocket League processes, correct client/server attachment, and simultaneous independent telemetry.

If Steam/Rocket League still permits only one game process, mark concurrency unsupported on this machine and stop revisiting it during this milestone.

Do not let concurrency delay the deterministic-pairing work.

---

# Required verification

At minimum:

- full pytest;
- Ruff on Rival-authored/modified Python;
- compileall relevant runtime/tools/probes/tests;
- model/hash smoke;
- installed reference hash verification;
- policy-freeze/off-mode regression verification;
- paired observation/action trace identity report;
- deterministic exposure report;
- accelerated-speed reproducibility report;
- JSON/fixture parse checks;
- `git diff --check`;
- credential/local-path staging scan;
- remote readback after push.

Record inherited untouched Wisp lint debt rather than mixing unrelated cleanup into this milestone.

---

# Required deliverables

Commit and push:

- implementation changes;
- tests;
- deterministic pairing report;
- machine-readable v4 evidence summaries;
- curated reproducible exposure fixture if found;
- optional single treatment result only if the exposure gate passed;
- `docs/MILESTONE_04_RESULTS.md`;
- exact commit SHA(s), hashes, warnings, runtime versions, selected test speed, concurrency result, and final `origin/main` SHA.

Prioritize causal evidence over a positive gameplay result. A clean finding that the exposure cannot yet be reproduced is acceptable.