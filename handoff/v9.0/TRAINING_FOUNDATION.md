# Rival v9 scratch training foundation

## Principle

Rival v9 learns the whole 1v1 policy from scratch. There is no frozen Wisp trunk, no mechanics PASS path, and no teacher-compatible action head.

Wisp and Nexto are external opponents/evaluation anchors. They may shape the distribution of experience, but their weights/actions are not copied into `RivalPolicyV1`.

## 1. Time base

Physics and policy cadence are 120 Hz.

For a 1v1 environment with two learning-agent decisions per physics tick:

- 1 simulated game second = 240 agent-steps;
- 1 simulated game minute = 14,400 agent-steps;
- 1 simulated game hour = 864,000 agent-steps.

Every training/evaluation report must include simulated game seconds/hours in addition to raw agent-steps.

Do not compare a 120-Hz v9 step count directly with the old 30-Hz M06/M08 step count.

## 2. PPO physical-time defaults

Use time-aware defaults rather than blindly copying 4/8-tick hyperparameters.

### Discount

To preserve approximately the same physical discount horizon as `gamma=0.99` at an eight-physics-tick decision cadence:

`gamma_120hz = 0.99 ** (1/8) = 0.9987444968227265`

Initial v9 default: `0.9987444968227265`.

### GAE lambda

To preserve approximately the same physical trace decay as the M08 `lambda=0.95` at four ticks:

`lambda_120hz = 0.95 ** (1/4) = 0.9872585449014338`

Initial v9 default: `0.9872585449014338`.

These are justified starting values, not immutable laws. Change them only through a versioned measured experiment.

### Batch scale

A 50k-agent-step iteration at 120 Hz covers only one quarter of the physical experience that 50k steps covered at 30 Hz.

Initial PPO scale should therefore start near four times the old step counts:

- rollout target / iteration: ~200,000 agent-steps;
- experience buffer: ~600,000;
- PPO batch: ~192,000;
- minibatch: ~48,000;
- epochs: 1 initially;
- clip range: 0.2.

Benchmark memory/update time on the 5090. Preserve roughly comparable **simulated game time per update** if sizes change.

### Learning rates

Scratch default to evaluate in the smoke/pilot:

- actor: `1e-4`;
- critic: `1e-4`.

If the hybrid policy or large batch requires a different measured value, record the change before serious training.

## 3. Hybrid exploration

The action distribution has two distinct exploration problems and must log them separately.

### Analog exploration

Track:

- mean/std for each of five axes;
- action quantiles;
- fraction with absolute value >0.95;
- per-axis pre/post-tanh entropy or an equivalent stable diagnostic;
- correlation of analog action with physical state over time.

Do not allow log-std to silently collapse to near-deterministic values during early scratch learning.

### Button exploration

Track:

- all eight jump/boost/handbrake combo probabilities and sampled shares;
- marginal jump, boost and handbrake shares;
- categorical entropy;
- physical-effect masks and masked counts.

Use a separate entropy term/coefficient if necessary so the categorical branch does not collapse merely because analog entropy dominates the scalar entropy statistic.

Exploration coefficients may anneal by simulated game-hours, never by a hard-coded raw step number without converting the cadence.

## 4. Reward design — outcome dominant, cadence safe

Goals remain the true objective.

### Terminal/event outcome

- score: `+10`;
- concede: `-10`.

Do not divide event rewards by cadence. A goal is one event regardless of policy frequency.

### Dense shaping rule

Any reward that can fire every step must be explicitly cadence-safe.

Preferred forms:

1. **potential-based shaping** `weight * (gamma * Phi(s') - Phi(s))`; or
2. a bounded physical reward rate multiplied by `dt = 1/120`.

Never copy a 30-Hz per-step shaping magnitude directly into a 120-Hz environment; that would quadruple its influence per simulated second.

### Initial shaping families

Keep each component independently logged and bounded.

#### Ball/attack progress

A small potential based on ball progress toward the opponent goal and useful ball velocity. It should be antisymmetric by team and subordinate to goal outcome.

#### Approach/control

Early scratch learning may use modest shaping for useful approach to the ball and meaningful touches so a random agent does not wait only for goals to obtain signal.

Avoid a permanent raw `distance-to-ball` reward that teaches mindless ball chasing. Prefer change/potential and reduce/anneal this component as basic competence develops.

#### Touch quality

Event shaping can score a touch by its immediate useful consequence, e.g. change in ball velocity toward the attacking goal/control direction, rather than paying a large constant simply for touching.

#### Recovery

Use a small potential measuring useful recovery state, including surface alignment, retained speed, goal-side relevance after lost control, and recovery completion. This supports wavedashes/wall recoveries/zap-dash-like behavior because those can improve recovery quality, not because their names are rewarded.

#### Boost efficiency

Do not globally punish using boost. Penalizing boost consumption alone teaches hoarding.

Small shaping may reward retaining/obtaining boost *conditional on comparable useful progress/recovery* or penalize clearly wasteful states such as sustained boost input while already unable to gain useful speed. Keep this term low-weight and audit it for farming.

#### Dodge/resource utility

Track airborne dodge/reset acquisition and productive follow-up. A tiny event reward for acquiring a recoverable resource is allowed only if necessary for learning signal and should anneal; the larger signal should come from productive follow-up, control and outcome.

### Named mechanics

No named mechanic receives a large identity reward.

Do not create `MustyReward`, `BreeziReward`, `ZapDashReward`, etc. as the main reason to perform those actions.

Named/broad mechanic detectors are diagnostic. A mechanic should survive because it improves control, recovery, resource use, scoring or defense.

## 5. Reward schedule

Scratch learning needs more assistance at the beginning than a mature policy.

Use a versioned schedule expressed in **simulated game-hours**:

### Foundation phase

Higher-but-still-subordinate approach/touch/progress/recovery shaping to bootstrap movement and ball interaction.

### Competence phase

Reduce approach/chase shaping. Retain outcome, touch quality, progress, recovery and resource efficiency.

### Mature phase

Outcome and directly useful control dominate. Anneal any bootstrap-only shaping toward zero or a very small floor.

Do not schedule purely by elapsed steps if objective metrics show that competence has not reached the intended stage. Stage transitions may be gated by touch rate, scoring, movement/recovery health and fixed-opponent evaluation.

## 6. Training-state curriculum

Natural 1v1 is always the majority distribution.

### Initial reset mixture

Start approximately:

- 70% ordinary natural kickoff / continuation distribution;
- 10% broad ground-possession / awkward challenge states;
- 8% broad wall/aerial/ceiling possession states;
- 8% broad awkward recovery / landing states;
- 4% broad low-resource / low-boost states.

These are distributions, not exact drills. Randomize position, orientation, velocity, boost, ball state and opponent relationship over physically plausible ranges.

As the policy becomes competent, move natural play toward 80–90%+ and reduce bootstrap reset mass.

### Why minority resets remain useful

Pure natural self-play can take a long time to encounter rare states while the policy is bad. Broad state randomization increases state-space coverage without teaching a canned named sequence.

The goal remains that mechanics work in chaotic natural play.

## 7. Opponent/self-play curriculum

### Start

The scratch policy primarily learns through self-play/current-policy play. At initialization both sides are bad, which is acceptable because they are similarly matched and can discover movement/touches together.

Use a small fixed-opponent anchor only after basic movement/contact metrics are nonzero. A completely random Rival repeatedly being destroyed by Wisp provides poor credit assignment.

### Historical pool

Once the policy develops meaningful competence:

- periodically snapshot frozen historical Rival opponents;
- sample current/lagged/history opponents to reduce catastrophic forgetting and self-play cycling;
- keep the pool intentionally small at first.

### Wisp/Nexto

- frozen Wisp is a useful headless/fixed anchor after basic competence;
- Nexto and installed Wisp remain required real-RLBot benchmarks;
- they do not become the majority training distribution merely because they are strong.

## 8. Mechanics diagnostics

Implement robust event/sequence metrics where feasible. These are primarily measurements, not reward identities.

Track at least:

- first-jump / double-jump / dodge use;
- directional dodge angles;
- flip cancel timing;
- airborne dodge/reset acquisition;
- productive touch after acquired dodge;
- ceiling contact + retained dodge;
- stall-like controller/state events;
- wavedash-like landing/dodge events;
- repeated wall-contact acceleration / wall-dash-like events;
- zap-dash-like recovery acceleration signatures;
- sidewall/ceiling recovery events;
- air-roll usage and rotation control;
- aerial touches and aerial possession duration;
- aerial distance/control per boost spent;
- boost remaining after aerial commitments;
- recovery time after miss/lost possession;
- landing orientation/speed retention;
- time until goal-side defensive relevance after lost possession;
- concessions during recovery windows.

Do not overclaim a named mechanic from a weak detector. Reports may label uncertain signatures as `*-like`.

## 9. Serious training budget units

Do not authorize the first large campaign in raw steps alone.

Use simulated game-hours as the primary experience budget and derive agent-steps from the fixed 120-Hz/two-agent contract.

Examples:

- 10 simulated game-hours = 8.64M agent-steps;
- 100 hours = 86.4M;
- 1,000 hours = 864M;
- 10,000 hours = 8.64B.

This makes the experience amount interpretable even if a future policy cadence or environment implementation changes.

## 10. v9 foundation/pilot limit

v9 itself is an **implementation and correctness milestone**, not the 150k-hour campaign.

After every gate in `VALIDATION_GATES.md` passes, it may run a maximum pilot of **2 simulated game-hours** (~1.728M agent-steps for two agents at 120 Hz) solely to prove:

- genuine hybrid PPO learning;
- checkpoint/reload/resume;
- exploration remains healthy;
- reward components remain finite/bounded;
- the policy begins changing measurable behavior without parser/domain failures.

Do not use the 2-hour pilot to judge the scratch architecture's eventual skill ceiling.

The next milestone should set the first serious game-hour campaign after this foundation is reviewed.

## 11. RLViser during scratch training

Retain the independent spectator process.

It may load the current/latest checkpoint and render one environment at ~1x human-viewable speed while training workers remain headless.

Watching random-to-competent behavior is useful, but do not adjust rewards or abort a healthy run solely from a few visually memorable mistakes without aggregate evidence.
