# Natural 1v1 environment contract

`RivalNatural1v1RocketSimV1` uses `RocketSimEngine` with one blue and one orange Octane, kickoff resets, goal termination, a 30-second no-touch truncation, a 300-second episode cap, RLBot-like one-tick action delay, and no renderer.

The opt-in `RivalRLViserSpectatorV1` is a separate process with exactly one independently
constructed RocketSim environment. It is paced at `tick_skip / 120` wall seconds per
decision and never changes the renderer-free training builders, worker count, rollout
configuration, diagnostics, or checkpoint state.

Milestone 05 used ordinary play only. Milestone 06 keeps natural 1v1 as the majority and
adds seeded minority families for broadly randomized aerial/wall possession, recovery,
and low-resource aerial states. No fixed named-mechanic drill defines the distribution.
Configured natural shares are 90%, 80%, 78%, and 76% across Stages A through D.

The serious campaign uses 56 RocketSim environments. This is not a guessed maximum: a
sustained 24/32/40/48/56/64 sweep selected the highest stable agent-step throughput at
56, while 64 remained stable but was 1.25% slower. CPU peaks at 100% were accepted when
the run remained stable, per the throughput objective.

Milestone 09 keeps the Gate 11 diagnostic environment/version reproducible and adds a
prospective Gate 13 pilot version with the authoritative 70/10/8/8/4 mixture: natural
kickoffs, broad ground possession/challenges, wall/aerial/ceiling possession, awkward
recovery/landing states, and low-resource states. The config migration changes only
the reset/metric environment contract. `RivalPolicyV1`, `RivalObsV1`,
`RivalActionV1`, cadence-safe reward, PPO settings, 56-worker selection, and native
one-tick timing remain unchanged. Pilot mechanic-like detectors are diagnostics only
and cannot affect reward or actions.

## Actions

`RivalExpandedActionV1` has 158 unique rows in controller order:

`throttle, steer, pitch, yaw, roll, jump, boost, handbrake`

Indices 0 through 89 are byte-exact Wisp actions in production order. The remaining 68 are unique rows appended from `AdvancedLookupTableAction(torque_subdivisions=3, flip_bins=16, include_stalls=True)`. The parser mirrors steer, yaw, and roll using the same team/field-X rule as live `XMirroredActionParser`, then repeats the world action for four or eight physics ticks.

## Observation

`WispCompatible432RLGymV1` preserves the 432-value count, normalization constants, feature categories, and ordering of `bot/obs_builder.py`. It includes four RocketSim predictions at 22, 66, 198, and 594 ticks and preserves previous-action semantics after X mirroring.

The live-to-training differences are explicit rather than hidden:

- RocketSim's `BallPredictor` replaces RLBot flatbuffer predictions at the same horizons.
- A bounded box-surface landing normal approximates the production arena-SDF query.
- A finite kinematic ETA replaces Wisp's process-global cached ETA helper.
- Episodic training score differential is zero unless a later curriculum wrapper supplies it.
- Previous inputs come from the training action parser for both self-play agents.

These differences retain the teacher's expected interface without making a false bit-parity claim about live RLBot observations.

## Reward

`RivalOutcomeRewardV1` remains the Milestone 05 reference. The serious campaign uses
`RivalRewardV2`, which emits and accumulates six independent components:

- signed goal/concede outcome at +10/-10;
- useful-touch/possession proxy;
- small signed offensive progress;
- small boost-efficiency accounting;
- small stateful recovery value.
- separately bounded low-weight mechanics/resource shaping.

Reward V2 tracks airborne dodge use, generic dodge-resource acquisition and productive
follow-up, wavedash-like and wall-speed recovery events, aerial boost/usefulness, and
recovery windows. There are no named-musty/breezi/Meeri/zap-dash/wall-dash rewards.
Advanced controls are structurally possible, while winning, possession quality,
resource use, and recovery determine value.
