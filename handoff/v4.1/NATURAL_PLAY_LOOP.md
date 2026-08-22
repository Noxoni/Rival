# Rival v4.1 — Natural-play optimization loop

## Core principle

Rival should improve from **naturally occurring Rocket League gameplay**, not from hand-authored challenge geometries or attempts to force the same exact trajectory twice.

The bot should react to continuous current observations and short rolling history:

- opponent and Rival positions, velocities, orientation, angular velocity;
- ball position, velocity, height and trajectory;
- relative distance / ETA / closing speed;
- opponent throttle, boost, steer, jump, handbrake and airborne/dodge state when available;
- current boost and nearby boost opportunities;
- recent touch ownership and touch timing;
- score and clock;
- previous actions / short temporal trends already available to the runtime.

Do not introduce scenario IDs such as `jump_fake`, `boost_then_brake`, or `true_commit` into normal gameplay logic. Those are test labels, not things Rival gets to know during a real match.

## Development loop

1. Run full five-minute natural 1v1 matches at approximately 5x effective simulation speed.
2. Collect schema-v3-or-later telemetry from every Rival decision.
3. Aggregate what repeatedly happens across many unrelated trajectories.
4. Rank recurring failure patterns by frequency and consequence.
5. Pick one narrow behavior to change.
6. Make the change depend only on observable live state / short history.
7. Run another natural accelerated batch against the same opponent mix.
8. Compare aggregate results. Keep the change only if the evidence is directionally better without an obvious regression.

This is an engineering optimization loop, not a requirement for perfectly controlled scientific trials.

## Natural match batch

Default development batch:

- full five-minute Soccar;
- approximately 5x effective game speed;
- goal replays skipped;
- replay auto-save disabled;
- rendering and performance overlay disabled;
- normal boost, gravity, demolition, scoring and kickoff behavior;
- alternate Rival blue/orange where practical;
- use installed Nexto and Wisp v2-75B as the primary reference opponents.

A reasonable first batch is 8–12 total games, balanced between Nexto and Wisp, but Codex may stop earlier if telemetry volume is already sufficient to identify a clear recurring pattern. Do not create artificial scenario cases merely to reach a sample count.

## What to compare

Do not expect identical scores or trajectories. Compare distributions / aggregates such as:

- wins/losses as context;
- goal differential;
- goals conceded following possession loss;
- touch ownership / next-touch rates;
- time or decision share with favorable ETA/possession proxy;
- possession-loss rate after Rival-controlled-ball states;
- challenge outcomes under different observed opponent pressure levels;
- boost spent / boost acquired around possession transitions;
- aerial attempts and outcomes when resources are low;
- intervention frequency and outcomes for whatever treatment is being tested;
- any other natural recurring pattern strongly supported by telemetry.

Individual anecdotes are useful for inspection but should not define the training/evaluation universe.

## Gameplay adjustment rule

A behavior adjustment must be state-conditioned. It may alter Wisp action selection/re-ranking based on continuous live features, but it must not hard-code a response to a named scripted scenario.

Prefer smooth or graded behavior over brittle thresholds when practical. For example, an opponent-commitment estimate should arise from live geometry, motion, input and recent trend, not `if scenario == jump_fake`.

Keep Wisp's existing legal action table and model as the strong baseline unless a later milestone explicitly authorizes broader retraining.

## Existing challenge calibration

Milestone 03's challenge estimator/re-ranker may be reused as code infrastructure if useful, but its rejected parameters are not accepted gameplay. It must remain disabled until a new naturally derived adjustment demonstrates improvement.

Do not spend this milestone making the previous 25 scripted fake-challenge cases deterministic. They can remain optional regression probes after a natural-play change exists.

## Fast simulation

Milestone 03 measured approximately:

- requested 5x -> ~4.92x effective simulated seconds / wall second vs Nexto;
- requested 5x -> ~5.00x effective simulated seconds / wall second vs Wisp.

Therefore effective simulation progression is the primary speed validation metric. Packet `game_speed` remaining at `1.0` is treated as a stale/insufficient echo, not proof that acceleration failed.

Monitor bot responsiveness and decision cadence. If a faster regime materially starves the bots of decisions or corrupts telemetry, reduce speed. Otherwise keep 5x for development throughput.

## Parallel games

Parallel Rocket League instances are optional. Do not spend substantial engineering time on them. The first multi-lane attempt showed platform/server-port limitations. If a simple isolated-port launch now works, use it; otherwise run sequentially at 5x and move on.

The major speedup comes from simulation acceleration, not launcher engineering.
