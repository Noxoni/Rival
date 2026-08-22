# Codex Start Prompt — Rival v4.1 Natural-Play Optimization

You are continuing the high-end offline Rocket League 1v1 bot **Rival** for RLBot v5.

Do not respond with another high-level plan. Work directly in `Noxoni/Rival`, execute this natural-play optimization milestone, commit stable work and evidence, and push to `origin/main`.

## Required starting state

Canonical repository:

`https://github.com/Noxoni/Rival`

Completed Milestone 03 implementation/evidence commit:

`e4cc175a4259202d5cc7ee437abef224b731354f`

Later commits may contain handoff documentation only. Fetch `origin/main`, inspect them, preserve legitimate history, and read all of:

- `handoff/v4.1/VERSION.md`
- `handoff/v4.1/NATURAL_PLAY_LOOP.md`
- `docs/MILESTONE_03_RESULTS.md`

`handoff/v4.0/` is historical and superseded before execution. Do **not** spend this milestone implementing deterministic scripted-scenario pairing.

The user's unrelated local files, ignored raw evidence and backups must remain untouched.

---

# Central correction

Rocket League gameplay is inherently trajectory-sensitive. Rival should improve from **natural accelerated matches and current observations**, not from memorizing or tuning against hand-authored fake-challenge scenarios.

The main loop is:

`natural play -> telemetry -> recurring pattern -> state-conditioned adjustment -> natural play -> aggregate comparison`

Do not require bit-identical trajectories between baseline and treatment.

Do not use scenario labels (`jump_fake`, `boost_then_brake`, etc.) as inputs to normal gameplay logic.

Existing controlled probes may remain for smoke/regression inspection, but they are not the primary training set or acceptance gate.

---

# Preserve the current baseline

Milestone 03's challenge-calibration treatment was rejected and must remain **off by default**.

Do not silently enable `m03-conservative-v1` or `m03-candidate-low0-gap1p5`.

The original Wisp-derived policy/model/action space remains the strong baseline. Any new treatment in this run must be easy to switch off and must use observable live state / short history.

---

# Stage 1 — accelerated natural runner

Make natural automated evidence runs use full five-minute Soccar at approximately **5x effective simulation speed** where viable.

Preserve:

- `Stadium_P`;
- normal Soccar physics;
- normal boost / gravity / demolition / scoring;
- normal kickoff countdowns;
- goal replays skipped;
- replay auto-save disabled;
- debug rendering `AlwaysOff`;
- performance monitor `NeverShow`;
- agent readiness waiting;
- clean match restart behavior.

Do not reject working 5x acceleration solely because packet `match_info.game_speed` remains `1.0`.

Validate acceleration by:

- simulated game seconds / wall second;
- policy decisions / simulated second;
- bot process responsiveness;
- valid telemetry;
- non-degenerate action distribution.

Milestone 03 already measured ~4.92x and ~5.00x effective progression at requested 5x. Reuse the mechanism unless current verification shows a real problem.

If 5x materially starves decisions or corrupts play, reduce speed only as much as needed and document why.

Parallel Rocket League instances are optional. Do not spend substantial time engineering them. If the existing isolation changes make two lanes trivially work, use them; otherwise run sequentially at 5x.

---

# Stage 2 — natural baseline batch

Run a natural accelerated baseline batch with the gameplay treatment disabled.

Primary opponents:

- installed Nexto;
- installed Wisp v2-75B.

Alternate blue/orange where practical.

Target roughly **8–12 full matches total**, balanced between the two references. This is a default, not a bureaucracy target: if telemetry volume is already clearly sufficient to identify a repeated high-impact behavior, Codex may stop the baseline batch earlier and state why.

Do not manufacture synthetic scenarios to inflate exposure counts.

Collect existing telemetry plus any low-overhead derived fields needed for natural analysis.

---

# Stage 3 — natural telemetry analysis

Analyze recurring behavior across unrelated natural trajectories.

Useful dimensions include, but are not limited to:

- current opponent distance / ETA / closing speed to Rival and ball;
- opponent forward/velocity alignment;
- opponent live controls and airborne/dodge state;
- Rival ball control / ETA advantage / touch ownership;
- boost resources and boost pickups around possession transitions;
- ball height / distance / trajectory;
- current and recent action history;
- score and clock;
- next-touch and possession outcomes;
- goals conceded after possession loss;
- aerial outcomes under different resource levels.

Rank patterns by **frequency and consequence**, not by similarity to a scripted case.

The known historical hypotheses (challenge overreaction, boost-route greed, resource-stressed aerials) are useful search directions, but do not force the next change to target one of them if natural data shows a stronger recurring issue.

Commit a compact human-readable and machine-readable natural-play summary. Keep huge raw telemetry ignored with hashes/manifests as before.

---

# Stage 4 — make one state-conditioned adjustment

Choose **one** recurring, evidence-backed behavior from the natural batch.

The adjustment must depend only on live observable state and short history available to Rival. It must not depend on scripted-scenario identity.

Prefer modifications such as:

- continuous/graded policy-logit re-ranking;
- a small live opponent-state estimate;
- resource/possession-aware bias over Wisp's existing legal actions;
- another compact state-conditioned correction that preserves the learned Wisp base.

Do not synthesize an entirely separate hard-coded controller.
Do not modify model weights in this milestone unless absolutely necessary; the intended loop is fast policy adjustment, not retraining Wisp.
Do not mix multiple unrelated gameplay fixes in one treatment.

Retain an explicit baseline-off switch and log both baseline Wisp choice and final Rival choice when treatment acts.

The old Milestone 03 challenge infrastructure may be reused if it naturally fits the selected behavior, but rejected thresholds are not accepted parameters.

---

# Stage 5 — natural treatment batch

Run another natural accelerated batch against the same opponent mix and similar side balance.

Do not expect identical trajectories.

Compare aggregate/distributional results, including as applicable:

- goal differential / scores as context;
- next-touch and possession retention/loss rates;
- goals conceded following possession loss;
- favorable ETA/possession share;
- intervention frequency and outcomes;
- boost efficiency around possession changes;
- challenge outcomes under naturally observed pressure;
- aerial success/failure under resource states;
- the specific metric associated with the chosen treatment.

A treatment does not need laboratory-perfect proof. Keep it if natural evidence is directionally and meaningfully better without an obvious damaging regression. Revert/default it off if it is worse or essentially noise.

Do not add more compensating rules merely to rescue a weak treatment.

---

# Test requirements

Keep software correctness tests proportionate to the change.

At minimum:

- full existing pytest suite;
- focused tests for new logic;
- Ruff on modified/Rival-authored Python;
- compileall on touched runtime/tools/tests;
- model/reference hash checks;
- baseline-off path remains available;
- telemetry/result JSON parsing;
- `git diff --check`;
- remote readback after push.

Do **not** require deterministic replay of whole Rocket League trajectories as an acceptance criterion.

---

# Required result

Commit/push a concise result containing:

- commit SHA(s);
- actual effective simulation speed used;
- number of natural baseline and treatment matches;
- opponents and side balance;
- natural telemetry volume;
- highest-frequency/highest-consequence patterns found;
- the one behavior selected for adjustment;
- exact state features used by the treatment;
- how often treatment actually changed Wisp's selected action;
- aggregate baseline vs treatment results;
- whether treatment was accepted or rejected;
- tests/verification;
- final `origin/main` SHA;
- next natural-play target.

Prioritize actual gameplay throughput and useful behavior improvement over experimental ceremony.