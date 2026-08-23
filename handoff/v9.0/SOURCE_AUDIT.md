# Rival v9 source audit

This audit exists to prevent `RivalActionV1` and `RivalObsV1` from being designed in a vacuum. It compares the strongest ideas available in the current Rival/Wisp stack, Nexto/Necto, current RLGym/RLGym Tools, and RLBot v5's actual runtime schema.

The rule is **adopt useful information and invariants, not compatibility baggage**.

## 1. RLBot v5 — authoritative live-runtime boundary

Primary references:

- https://wiki.rlbot.org/v5/botmaking/tick-rate/
- https://wiki.rlbot.org/v5/botmaking/game-data/
- https://github.com/RLBot/flatbuffers-schema/blob/main/schema/gamedata.fbs

### Adopt

RLBot v5 is the authority for what a deployed bot can actually receive and control.

Useful live data includes:

- GamePackets delivered at up to 120 Hz, matching Rocket League's 120-Hz physics simulation;
- five continuous controller axes: throttle, steer, pitch, yaw, roll;
- jump, boost, handbrake button states;
- player physics and hitbox information;
- `air_state` distinguishing grounded / first-jump / double-jump / dodge force states;
- `dodge_timeout`, `has_jumped`, `has_double_jumped`, `has_dodged`, `dodge_elapsed`, `dodge_dir`;
- current boost, supersonic state and demolition timeout;
- each player's `last_input`, including opponents;
- latest ball touch information;
- boost-pad active state and timer;
- score, match phase, time remaining/overtime and gravity;
- static goal and boost-pad geometry through FieldInfo.

These fields allow a new policy to observe substantially more explicit mechanics state than older RLBot/RLGym policies often exposed.

### Consequence

Do not force Rival to infer a dodge window, current dodge phase, opponent last input, or boost respawn timing from motion history when RLBot already exposes the relevant state and RocketSim can represent the matching concept.

### Reject

- no v4 compatibility fields merely for historical reasons;
- no Rumble `use_item` in the initial Soccar-only action contract;
- no actor feature that exists only in RLBot unless an exactly equivalent canonical representation exists in training.

## 2. Current RLGym / RocketSim — authoritative training-state substrate

Primary references:

- https://github.com/RLGym/rlgym/blob/main/rlgym/rocket_league/obs_builders/default_obs.py
- https://github.com/RLGym/rlgym/blob/main/rlgym/rocket_league/sim/rocketsim_engine.py

### Adopt

Current RLGym already demonstrates the value of directly exposing partially observable mechanics state such as:

- holding jump;
- handbrake;
- has jumped;
- currently jumping;
- has flipped;
- currently flipping;
- has double jumped;
- can flip;
- air time since jump.

RocketSimEngine exposes even richer car state internally, including wheel contacts, boost-active time, jump time, flip time/torque and autoflip state.

Use RocketSim as the physics/training source, but map it into `RivalCanonicalStateV1` rather than letting the actor consume RLGym-only object semantics directly.

### Reject

RLGym's stock `DefaultObs` is intentionally simple and generic. It omits many features useful for a mechanics-first 1v1 policy: opponent input history, explicit match strategy context, rich goal geometry, shared predictions, surface/recovery geometry, full boost-pad entities, richer jump/dodge timing and temporal controller history.

It is a useful minimum reference, not the final Rival observation.

## 3. Nexto — entity representation and relative geometry

Pinned upstream reference used by this project:

`Rolv-Arild/Necto@2e6ed7d6ed2b352e8ff529d4a12a0c9c70c28cca`

Relevant source:

- `rlbot-support/Nexto/nexto_obs.py`

### Adopt

Nexto's observation design treats the world as typed entities rather than only one monolithic hand-engineered vector:

- self / teammate / opponent / ball / boost type markers;
- position, linear velocity, forward/up orientation and angular velocity;
- boost, demo, ground and flip state;
- previous actions;
- relative transforms into the observing player's frame;
- all standard boost pads represented as entities.

For Rival, preserve the *idea* of structured entities and relative representations, especially for all 34 boost pads.

### Reject

- no need to preserve Nexto's exact tensor shapes, attention architecture, padding rules or normalization;
- no 8-tick cadence inherited because Nexto used it;
- no omission of newer RLBot v5 mechanics fields just because they were unavailable to the old model.

## 4. Necto — match state, timers and richer relative features

Pinned source:

- `training/obs.py` under the same Necto commit.

### Adopt

Useful additions over the older Nexto observation include:

- separate jump availability/state;
- demo timers;
- boost respawn timers;
- goal differential;
- time remaining;
- overtime;
- additional player-relative and flip-relative coordinate features.

Rival should explicitly know strategic match context and exact resource timers instead of relearning those clocks from sparse events.

### Reject

Do not reproduce Necto's older timer emulation when current RLBot/RLGym can provide or canonically reconstruct the timer directly.

## 5. Wisp / current Rival baseline — useful geometry plus a transfer warning

Relevant local source:

- `bot/obs_builder.py`
- `bot/eta.py`
- `docs/MILESTONE_07_RESULTS.md`

### Adopt

Wisp contains several valuable ideas:

- future ball context;
- previous controller input;
- boost-map context;
- wall/corner distances;
- landing/surface context;
- match score context;
- local ball/car/goal geometry;
- explicit touch and handbrake information;
- a notion of intercept/arrival time.

Those concepts belong in a rich Rival observation when they can be implemented identically in both domains.

### Critical warning from M07

M07 proved that "same shape" is not observation parity. The old training reconstruction changed frozen Wisp's action on more than half of held live states despite having 432 values in the expected order. Cached ETA and analog touch/handbrake/car-state semantics were major contributors.

Therefore v9 does **not** copy Wisp's process-global cached ETA or use separate live/training implementations of any actor feature.

If Rival keeps a deterministic intercept-time feature, it is a new pure function/state machine shared by both adapters and independently parity-tested.

## 6. RLGym lookup-table actions — useful history, wrong final ceiling

Primary references:

- RLGym `LookupTableAction`
- RLGym Tools `AdvancedLookupTableAction`

### Adopt

The lookup tables demonstrate which broad controller patterns are often useful and provide excellent regression/capability traces for testing the new native action path.

In particular, the advanced table is useful as a source of known aerial/dodge/stall controller rows for parser tests.

### Reject

The 90- and 158-row tables are finite subsets of a controller whose five analog axes are continuous. They cannot be the final mechanical ceiling if the objective is to allow any physically valid controller blend.

`RivalActionV1` therefore emits the native controller distribution directly.

## 7. rlgym-ppo — proven pipeline, custom hybrid policy required

M05–M08 already proved the existing Python `rlgym-ppo` stack can be integrated with custom policies, CUDA training, checkpoints, reload/resume and centralized inference.

### Adopt

- existing rollout/checkpoint/evaluation infrastructure where it remains generic;
- PPO as the first scratch learner because the project already has a working operational path.

### Modify

The stock continuous-policy path is not sufficient for the exact v9 action contract. Rival needs a custom hybrid distribution:

- tanh-squashed continuous Gaussian for five analog axes with correct transformed log probability;
- one 8-way categorical for the three correlated button states.

The mixed log probability and entropy calculations must be tested analytically/numerically before training.

## 8. rlgym-learn — candidate throughput/trainer backend, not an architectural dependency

Reference:

- https://github.com/JPK314/rlgym-learn

`rlgym-learn` is a newer RLGym v2 learning framework with Rust/shared-memory multiprocess infrastructure and PPO support. It may materially reduce Python/process overhead, which matters more at 120-Hz policy cadence.

### Decision

Keep `RivalObsV1` and `RivalActionV1` trainer-neutral. During the v9 foundation milestone, run a bounded apples-to-apples throughput/health comparison between:

1. the existing proven `rlgym-ppo` path with the new hybrid policy; and
2. `rlgym-learn`, **only if** it can express the exact same hybrid action/log-probability/checkpoint contract without a large detour.

Choose based on measured stable simulated game-time throughput plus update wall time, not novelty.

Do not rewrite the project around a new trainer merely because it is newer.

## 9. Final adopted principles

The source audit supports these frozen design principles:

1. Native 120-Hz controller output is the real mechanical interface.
2. Full controller expressivity is more important than preserving a discrete legacy action set.
3. Explicit mechanical state should be observed when available rather than inferred unnecessarily.
4. Relative/entity representations reduce wasted learning without hard-coding strategy.
5. All boost pads and meaningful match-resource timers belong in the world model.
6. Temporal controller history matters more at 120 Hz.
7. Derived geometry/prediction can be valuable only when one shared implementation serves training and deployment.
8. No old model's tensor contract is sacred in a scratch policy.
9. Training throughput must be optimized *after* the native action/observation contract is correct.
10. Wisp/Nexto remain valuable opponents and benchmarks, not components of Rival's actor.
