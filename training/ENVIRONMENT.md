# Natural 1v1 environment contract

`RivalNatural1v1RocketSimV1` uses `RocketSimEngine` with one blue and one orange Octane, kickoff resets, goal termination, a 30-second no-touch truncation, a 300-second episode cap, RLBot-like one-tick action delay, and no renderer.

The default state distribution is ordinary play. Broad wall, aerial, recovery, replay, and opponent-diversity curricula are deliberately deferred until the natural self-play foundation earns a longer run.

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

`RivalOutcomeRewardV1` emits and accumulates five independent components:

- signed goal/concede outcome at +10/-10;
- useful-touch/possession proxy;
- small signed offensive progress;
- small boost-efficiency accounting;
- small stateful recovery value.

There are no named-mechanic bonuses. Advanced controls are structurally possible, while winning, possession quality, resource use, and recovery determine value.
