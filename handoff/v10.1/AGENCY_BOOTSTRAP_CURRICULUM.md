# RivalAgencyBootstrapCurriculumV1

## Purpose

The normal 70/10/8/8/4 curriculum assumes a policy that can already interact with Rocket League. The current scratch policy has not reached that point reliably.

This temporary curriculum therefore increases **interaction density**. It does not teach exact mechanics or expose named-action macros. Every family is a broad randomized distribution whose only purpose is to put Rival close enough to a solvable motor/ball-control problem that PPO can receive useful feedback.

Recovery-specific resets are deliberately removed during this bootstrap because recovery was the clearest capability already learned by +10h.

## General rules

- Both teams still use the same trainable Rival policy unless an explicitly separate future opponent curriculum is authorized.
- Randomly assign the advantaged/active role to blue or orange on every non-natural reset.
- Randomize left/right field side and apply the existing episode symmetry augmentation.
- Never place the car at one exact mechanic setup repeatedly.
- Preserve legal Soccar state and the existing canonical observation/action contracts.
- Default boost-pad state remains standard unless the family explicitly changes car boost.
- Goal and reset semantics remain ordinary Soccar.

## Phase A — wake up and touch the ball

Use from bootstrap start until its readiness gate passes, with a **10 simulated-hour hard review ceiling**.

| Family | Weight |
| --- | ---: |
| `ground_acquisition` | 30% |
| `moving_ball_chase` | 20% |
| `touch_chain` | 20% |
| `easy_aerial_contact` | 15% |
| `easy_finish` | 10% |
| `natural` | 5% |

### Ground acquisition — 30%

Goal: learn throttle/steer/boost enough to reach and touch a simple ground ball.

Randomize broadly:

- ball on ground, stationary or moving slowly (<~700 uu/s);
- active car 600–2200 uu from ball;
- initial heading error approximately ±45 degrees, not perfectly lined up every time;
- active-car speed 0–800 uu/s;
- active-car boost 20–80;
- opponent placed 2200–4500 uu from ball, usually lateral or goal-side rather than directly contesting immediately;
- ball and roles may be anywhere in a broad playable midfield/half-field region.

This should make `drive to the thing and hit it` one of the easiest profitable discoveries in training.

### Moving-ball chase — 20%

Goal: learn steering, acceleration and interception against a ball that will not wait.

Randomize:

- rolling/bouncing-low ball moving ~300–1400 uu/s;
- active car 800–3000 uu away;
- broad intercept angle rather than simply spawning directly behind the ball;
- active-car speed 100–1100 uu/s and boost 15–80;
- opponent generally farther from the immediate intercept than the active car.

Do not supply an intercept target or scripted controller. Rival must use `RivalObsV1` to solve it.

### Touch chain — 20%

Goal: turn one touch into another instead of colliding with the ball and leaving.

Randomize:

- ball 150–750 uu ahead/offset from active car;
- ball and car have broadly similar attacking motion;
- active-car speed ~400–1500 uu/s;
- ball speed ~250–1300 uu/s;
- boost 10–60;
- opponent starts 1800–4200 uu away so there is pressure eventually but enough time for a second touch to be possible.

This is a broad possession-follow-up distribution, not a dribble macro.

### Easy aerial contact — 15%

Goal: make jumping/boosting/air-control discoverable because an elevated ball is actually reachable.

Randomize:

- ball height approximately 250–900 uu;
- moderate horizontal ball velocity, roughly 0–900 uu/s;
- active car starts on or very near the ground, 500–1700 uu behind/below the likely contact point;
- heading error approximately ±30 degrees;
- boost 30–80;
- opponent 2000–4500 uu away or otherwise unable to instantly steal the setup;
- include a minority (~20%) of near-sidewall variants once the basic setup is valid.

Do **not** spawn the car already in a completed aerial orientation. The policy must discover jump/boost/pitch/yaw/roll use itself.

### Easy finish — 10%

Goal: make the +10 goal signal reachable often enough for PPO to connect ball interaction with scoring.

Randomize:

- ball between active car and opponent goal;
- ball roughly 700–2500 uu from the goal line;
- active car 300–1200 uu behind/offset from ball;
- ball may be stationary or moving moderately;
- active-car boost 20–80;
- defender exists but begins off-center, recovering, or far enough away that a simple useful touch can score.

Do not place the ball already crossing the goal line. Rival must physically create the scoring contact/trajectory.

### Natural — 5%

Standard randomized legal 1v1 kickoff using the existing v9 implementation. This retains some whole-game context but no longer dominates the early motor bootstrap.

## Phase B — interact reliably

Enter only after the Phase-A readiness gate in `EVALUATION_AND_EXIT_GATES.md` passes. Hard review ceiling: **15 additional bootstrap hours total**.

Weights:

| Family | Weight |
| --- | ---: |
| natural | 20% |
| ground_acquisition | 20% |
| moving_ball_chase | 15% |
| touch_chain | 20% |
| easy_aerial_contact | 15% |
| easy_finish | 10% |

Compared with Phase A, natural play becomes substantial while all high-interaction families remain common.

## Phase C — consolidation

Enter only after the Phase-B readiness gate passes. Hard review ceiling: **25 additional bootstrap hours total**.

Weights:

| Family | Weight |
| --- | ---: |
| natural | 40% |
| ground_acquisition | 15% |
| moving_ball_chase | 10% |
| touch_chain | 15% |
| easy_aerial_contact | 10% |
| easy_finish | 10% |

Phase C asks the policy to carry its learned motor skills into increasingly normal games.

The bootstrap should end rather than continue indefinitely once the exit gate passes.

## Dead-play recycling

During v10.1 only:

- no-touch timeout: **10 seconds**;
- episode timeout: **120 seconds**;
- goal terminates normally.

The current 30-second no-touch timeout is too expensive when both policies are capable of spending most of an episode not interacting with the ball.

A 10-second no-touch reset is not a punishment. It simply allocates the next samples to a new solvable interaction instead of collecting more empty driving.

Record truncation reason and no-touch-timeout share for every training/evaluation boundary.

## Why no dedicated recovery family

Recovery improved sharply during the original M10 run while ball interaction regressed. The bootstrap should not keep paying curriculum bandwidth to the one behavior already showing clear progress.

Recovery still occurs naturally after misses, aerials, collisions and failed touches. It is simply no longer over-sampled.

## Distribution validation

Before PPO:

1. sample at least 10,000 resets from each phase;
2. verify empirical family shares within sampling tolerance;
3. verify every position/velocity/orientation/boost is finite and physically bounded;
4. report car-ball distance, ball height, car speed and boost distributions by family;
5. visually inspect a small RLViser sample from every family;
6. prove episode symmetry mirrors each family correctly;
7. ensure neither team/field side receives systematic advantage after random role assignment.
