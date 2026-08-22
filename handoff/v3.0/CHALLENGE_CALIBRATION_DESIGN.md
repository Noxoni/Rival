# Challenge-Commitment Calibration Design

## Objective

Correct one narrow Rival failure mode without replacing the Wisp policy:

> When Rival has plausible possession and the opponent presents pressure that is not yet physically committed to the ball, avoid an unnecessarily early possession-releasing jump/flick for a very short period, then reassess.

The intervention must be conservative, reversible, measurable, and restricted to Wisp's existing legal action space.

---

## Why a temporal estimator is required

At the first instant of a fake challenge, a real challenge and a fake can look identical. The bot cannot know intent.

The useful distinction appears over time:

- closing velocity persists or collapses;
- ETA to ball continues decreasing or begins increasing;
- the opponent's forward/velocity vector remains ball-intersecting or turns away;
- throttle/boost remains committed or changes to braking/low throttle;
- jump trajectory actually intersects the play or becomes a jump fake;
- projected closest approach to the moving ball tightens or opens.

Therefore do not classify one packet as `fake` or `real`. Maintain a short challenge-history state and estimate **commitment**.

---

# A. Commitment estimator

Implement a pure/testable module, suggested logical path:

`bot/strategy/challenge_commitment.py`

Exact path/type names may differ.

The estimator should return a structured result such as:

```python
ChallengeCommitmentEstimate(
    score=0.0_to_1.0,
    state="low" | "ambiguous" | "high",
    pressure_present=True_or_false,
    components={...},
    history={...},
)
```

Do not hide all logic in one opaque scalar. Telemetry must expose the components.

## Candidate instantaneous features

Use values already present in Rival/RLBot v5 where possible:

- opponent distance to ball;
- Rival distance to ball;
- opponent ETA to ball;
- Rival ETA to ball;
- opponent-ball closing velocity;
- opponent-Rival closing velocity;
- opponent speed;
- opponent forward-vector alignment toward the ball;
- opponent velocity alignment toward the ball;
- opponent last controller input: throttle, boost, steer, jump, handbrake;
- opponent airborne/jump/dodge state;
- ball velocity;
- relative field geometry.

## Geometric commitment feature

Add a short-horizon projected closest-approach / intercept-corridor metric.

A reasonable first approximation is constant relative velocity over a short horizon:

```text
r = ball_position - opponent_position
v = opponent_velocity - ball_velocity

t* = clamp(dot(r, v) / dot(v, v), 0, horizon)
projected_miss_distance = |r - v * t*|
```

Guard zero/near-zero relative velocity.

This is not a full Rocket League intercept solver. It is a feature answering whether the opponent's current trajectory is actually converging on the moving ball.

Codex may improve this feature using existing RocketSim/ETA utilities if the result remains deterministic, cheap, and testable.

## Temporal features

Maintain only a short bounded history sufficient to capture a challenge developing or aborting. Derive at least:

- closing-speed trend;
- opponent ETA trend;
- projected-miss-distance trend;
- forward/velocity alignment trend;
- duration of sustained pressure;
- explicit abort evidence: braking/reverse throttle, hard steer away, rapidly increasing miss distance, collapsing closing speed.

Do not let stale state persist across kickoff, goal/reset, timer rewind, demolition/reset, or missing-player transitions.

## Important interpretation rules

- `jump=True` alone is **not** proof of commitment; the controlled evidence includes a jump fake.
- `boost=True` alone is **not** proof of commitment; boost-then-brake and boost-then-veer are explicit fake cases.
- high closing speed alone is **not** proof of commitment.
- an opponent that is geometrically guaranteed or nearly guaranteed to intersect the ball soon should move strongly toward `high` commitment even if controller inputs are unavailable.
- a clear abort should reduce commitment quickly enough to matter within the 8-tick policy cadence.

Use thresholds/config values that are named and documented, not magic numbers scattered through the runtime.

---

# B. Possession/intervention gate

The estimator should run broadly, but the gameplay intervention must be narrow.

Only consider intervention when all required safety/gating conditions hold. Suggested categories:

## Rival plausibly controls or can continue controlling the ball

Use evidence-supported fields rather than requiring a perfect dribble classifier. Candidate signals include:

- Rival is grounded;
- ball is within a configurable control distance;
- Rival ETA is competitive/favorable relative to opponent ETA;
- recent Rival touch when available;
- ball height/relative geometry is compatible with ground possession;
- no reset/goal/kickoff phase.

Do not require every signal if that excludes the documented fixtures. Tune the gate against v2 evidence and controlled probes.

## Opponent pressure exists

The intervention is not for an unpressured dribble. Require an apparent challenge/pressure window using distance/ETA/closing geometry.

## Pressure is ambiguous rather than clearly committed

- `high` commitment -> do not delay the Wisp release response.
- `low` / no meaningful pressure -> generally let Wisp act normally unless the baseline selected action is specifically being considered under the continuation experiment.
- `ambiguous` is the primary intervention region.

## Selected Wisp action is release-sensitive

Milestone 02's broad detector counted boost/pitch as release-like for ranking; that is too broad for control.

For the first intervention, focus on **grounded jump/dodge initiation** or another clearly justified discrete action category that actually releases the possession decision.

Do not suppress ordinary boost, steering, or throttle merely because the broad v2 detector called them release-like.

## Safety exclusions

Do not intervene when delaying would obviously be unsafe, including cases such as:

- Rival already airborne;
- defensive emergency / imminent own-goal threat;
- opponent is already within a very short unavoidable intercept window;
- no legal non-jump continuation exists;
- reset/kickoff/goal phase;
- any state where the estimator is invalid/uninitialized.

Codex should define these from existing state/ETA/geometry rather than invent an enormous tactical state machine.

---

# C. Policy re-ranking intervention

The intervention must operate on the **existing Wisp logits and legal action mask**.

Do not synthesize a controller action like `throttle=1, steer=...`.

## Baseline

```text
baseline_action = argmax(masked_wisp_logits)
```

## Treatment candidate

If the intervention gate is active and the baseline action is a release-sensitive jump/dodge:

1. find legal non-jump continuation candidates in the existing discrete action table;
2. select the highest-logit suitable continuation candidate;
3. compare its logit/probability to the baseline selected action;
4. intervene only if the model preference gap is within a configurable conservative limit and all safety checks pass.

This means the Wisp model still decides *how* Rival continues driving/boosting/steering; v3 only gives it a tiny amount of time to gather more evidence about opponent commitment.

## Maximum deferral

Start with **one policy decision tick** of deferral (approximately 8 Rocket League physics ticks / 66.7 ms at 120 Hz).

Only test a two-policy-tick maximum (~133 ms) if one tick does not produce enough separation in controlled evidence.

Never allow indefinite suppression of a jump. Track the deferral budget per challenge episode.

Once the opponent becomes `high` commitment, immediately stop deferring and allow the baseline Wisp selection on the next decision.

## Strong model preference

If Wisp strongly prefers the release action over the best continuation candidate, do not casually override it. Use a named max-logit-gap / confidence condition and evaluate it in the experiment matrix.

The point is to correct marginal/ambiguous panic releases first, not overrule confident mechanics everywhere.

---

# D. Modes

Provide explicit modes for A/B testing:

```text
off       -> exact pre-v3 baseline selection
observe   -> compute estimator + hypothetical intervention, but return baseline action
intervene -> apply the accepted re-ranking rule
```

Suggested environment/config surface:

```text
RIVAL_CHALLENGE_CALIBRATION_MODE=off|observe|intervene
```

Use whatever config structure best fits the codebase, but preserve a one-switch exact baseline path.

---

# E. Telemetry schema v3

Bump telemetry/schema metadata when adding gameplay intervention data.

Every decision should make it possible to reconstruct:

- original Wisp action index/controller action;
- final Rival action index/controller action;
- whether intervention was eligible;
- whether intervention was applied;
- reason applied/not applied;
- commitment score/state;
- pressure-present flag;
- estimator components;
- short-history trend values;
- possession-gate components;
- safety-exclusion reason if any;
- chosen continuation candidate;
- baseline-vs-continuation logit/probability gap;
- remaining deferral budget;
- challenge episode identifier if useful.

Do not duplicate full raw logits unless verbose telemetry is explicitly enabled.

---

# F. Non-goals

Milestone 03 is not permission to:

- hard-code a complete dribbling controller;
- create a rule such as `if opponent fake then never jump`;
- disable flicks globally;
- change Wisp model weights;
- change observation semantics to improve gameplay;
- change tick skip/action delay;
- fix boost greed or resource-stressed aerials;
- claim the estimator reads opponent intent.

It estimates physical commitment from observable state and briefly delays a marginal release decision while uncertainty resolves.