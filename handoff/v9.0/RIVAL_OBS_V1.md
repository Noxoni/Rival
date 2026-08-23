# RivalObsV1 — rich train/deploy-identical 1v1 observation contract

## Decision

`RivalObsV1` is a new observation for a scratch policy. It is not constrained to Wisp's 432-value layout and does not maintain compatibility with any old checkpoint.

The primary rule is stronger than feature richness:

> **There is one feature implementation. RocketSim and RLBot are only adapters into the same canonical state.**

No separate "training approximation" and "live implementation" may exist for actor features.

## Canonical state layer

Implement a versioned `RivalCanonicalStateV1` that contains only values available or exactly reconstructible in both environments.

Two thin adapters populate it:

- `RocketSim -> RivalCanonicalStateV1`
- `RLBot v5 GamePacket/FieldInfo -> RivalCanonicalStateV1`

All normalization, coordinate transforms, relative features, surface geometry, histories, prediction features, timers and masks run *after* this canonicalization in shared code.

A canonical snapshot serialized to disk must produce bit-identical `RivalObsV1` whether invoked from the training or deployment path.

## Coordinate convention

- Normalize team direction so Rival always attacks toward positive Y and defends negative Y.
- Apply the ordinary 180-degree team inversion for orange.
- Do **not** dynamically X-mirror based on the car's current field position; that creates a discontinuity when crossing the center line.
- Optional left/right symmetry augmentation is episode-stable: choose a mirror bit at reset and apply the same reflection to every observation and controller action for that entire episode.

Actor observations contain both team-frame and car-local quantities where each is useful. Redundancy is intentional when it saves the network from relearning common coordinate transforms.

## Logical observation blocks

The initial schema is 1v1-specific and fixed-order. The implementation must machine-generate a field/index manifest and final float count from this specification; do not maintain hand-written offsets in multiple files.

### A. Match and control context

Include:

- score differential from Rival's perspective;
- normalized game time remaining;
- overtime flag;
- kickoff flag;
- active-play flag;
- normalized gravity;
- seconds since Rival's latest ball touch;
- seconds since opponent's latest ball touch;
- one-hot last toucher: Rival / opponent / none;
- age of the cached shared ball-prediction context;
- Rival deterministic intercept-time proxy;
- opponent deterministic intercept-time proxy;
- intercept-time advantage.

Touch ages are maintained from actual touch events and clipped/normalized. The intercept proxy must use one shared pure implementation in training and deployment; it must not reproduce Wisp's process-global cached ETA semantics.

### B. Rival car block

Physics/state:

- team-frame position XYZ;
- forward XYZ;
- up XYZ;
- team-frame linear velocity XYZ;
- team-frame angular velocity XYZ;
- car-local linear velocity XYZ;
- car-local angular velocity XYZ;
- normalized speed;
- signed forward speed;
- boost amount;
- demolition time remaining;
- wheel/surface-contact flag;
- boosting flag;
- supersonic flag;
- current handbrake state.

Jump/dodge state:

- jump currently held;
- has jumped;
- has double-jumped;
- has dodged/flipped;
- dodge currently available;
- currently in first-jump force phase;
- currently in dodge/flip phase;
- air time since first jump/takeoff where defined;
- first-jump hold elapsed;
- dodge window remaining;
- dodge elapsed;
- dodge direction XY in the canonical car/flip frame.

Surface/recovery geometry from the shared standard-Soccar geometry helper:

- distance to floor;
- distance to ceiling;
- distance to nearest side wall;
- distance to nearest back wall;
- distance to corner surface;
- nearest-surface normal in car-local coordinates XYZ;
- car-up alignment with that surface normal;
- signed velocity toward/away from the nearest surface.

Goal geometry:

- own-goal center in car-local coordinates XYZ;
- opponent-goal center in car-local coordinates XYZ.

### C. Opponent car block

Include the same deployable mechanical state where useful, plus relational context:

- team-frame position, forward, up, linear velocity, angular velocity;
- Rival-local relative position and relative velocity;
- normalized speed and signed forward speed;
- boost, demo time, surface contact, boosting, supersonic, handbrake;
- jump-held, has-jumped, has-double-jumped, has-dodged, can-dodge, jumping, dodging;
- air/jump/dodge timing fields and dodge direction when canonical parity supports them;
- opponent's latest RL controller input (8 fields);
- ball position and relative velocity in the opponent's local frame;
- the same shared surface-distance/nearest-normal/recovery geometry used for Rival;
- goal-side-of-ball indicator from Rival's perspective.

Do not provide simulator-private opponent information that RLBot cannot provide or reconstruct at runtime.

### D. Ball and goal block

Raw/team-frame ball state:

- position XYZ;
- linear velocity XYZ;
- angular velocity XYZ.

Relative state:

- ball position and velocity in Rival-local coordinates;
- ball position and velocity in opponent-local coordinates;
- ball speed;
- Rival-ball distance and closing speed;
- opponent-ball distance and closing speed.

Goal geometry:

- ball-to-own-goal-center vector XYZ;
- ball-to-opponent-goal-center vector XYZ;
- ball-to-own-left-post and own-right-post vectors;
- ball-to-opponent-left-post and opponent-right-post vectors.

The post vectors expose shooting/opening geometry without baking in a hand-authored shot decision.

### E. Shared ball-prediction block

Use the **same RocketSim ball predictor implementation in both training and deployment**, initialized from the current canonical ball state. Do not feed RLBot's native prediction to the deployed actor while feeding RocketSim prediction during training.

Initial horizons:

- 0.125 s;
- 0.25 s;
- 0.5 s;
- 1.0 s;
- 2.0 s;
- 4.0 s.

For each horizon include:

- predicted team-frame position XYZ;
- predicted team-frame linear velocity XYZ;
- predicted position relative to Rival in Rival-local coordinates XYZ;
- predicted velocity relative to Rival in Rival-local coordinates XYZ.

To control CPU cost at 120-Hz policy cadence, this expensive context may be recomputed every 4 physics ticks and held between updates. The observation includes its exact age. Raw car/ball state still updates every physics tick.

Before freezing implementation, benchmark prediction period 1 vs 2 vs 4 ticks. Period 4 is the default target only if it materially improves throughput without hurting short evaluation.

### F. Boost-pad entity block

Represent all 34 standard Soccar boost pads. Do not reduce the map to only the nearest few pads.

Each pad entity contains:

- team-frame absolute X/Y location;
- Rival-local relative XYZ location;
- distance from the current ball;
- big/small flag;
- active flag;
- normalized time until active.

Canonical timer meaning is **time until the pad can be collected**. RLBot's active/timer representation and RLGym's remaining-timer representation must be converted into that same meaning in the adapter.

Pads are kept as 34 fixed-order entities and processed by a shared pad encoder/pooling or attention module in the actor. The flattened transport representation retains the fixed entity boundaries in its schema metadata.

### G. Controller-history block

At 120 Hz, controller history is valuable mechanical context.

Maintain ring buffers with identical reset/update semantics in training and deployment:

- Rival's previous 8 physical controller rows: `8 x 8` values;
- opponent's previous 8 physical controller rows: `8 x 8` values.

The opponent history is observable in RLBot v5 because each packet exposes the opponent's last input. RocketSim uses the exact controls applied by that agent.

Do not substitute policy logits or action-table indices. Store the actual physical controller values.

### H. One-tick motion-delta block

For Rival, opponent and ball include:

- change in linear velocity XYZ since the previous physics tick;
- change in angular velocity XYZ since the previous physics tick.

Use normalized per-tick deltas rather than a noisy wall-clock acceleration estimate. Reset these to zero on episode/reset discontinuities.

This gives a feed-forward actor immediate contact/bounce/rotation-change information without requiring an RNN simply to infer the previous frame.

## Estimated size

The intended contract is on the order of ~700 normalized floats, including the flattened pad/prediction/history entities. Exact dimension is generated by the schema implementation and then frozen in `RivalObsV1` metadata.

A ~700-float observation is small relative to the policy network and preferable to silently omitting high-value mechanical state. The network should process entity sub-blocks structurally rather than treating all fields as an undifferentiated MLP input.

## Normalization

Normalize by stable physical scales, not running training statistics:

- X/Y/Z positions by appropriate field dimensions;
- linear velocities by 2300 uu/s unless a more specific stable scale is documented;
- angular velocities by 5.5 rad/s;
- boost by 100;
- game time by 300 seconds;
- timers by their physical maximum and clip when necessary;
- controller values are already bounded;
- distances use field/car/ball scales appropriate to the quantity.

No runtime observation standardization is allowed unless the exact running statistics are checkpointed, exported, and applied identically in deployment. Default for v9 is **off**.

## Derived feature policy

Derived features are allowed when they satisfy all three rules:

1. materially useful or difficult for a small network to infer cheaply;
2. computed by the same source code from canonical state in training and deployment;
3. independently parity-tested on held RLBot states.

This is why v9 keeps prediction, surface, local-frame and intercept context but rejects the old Wisp cached-ETA implementation.

## What is deliberately not copied from older bots

- no dynamic X mirror tied to current car X;
- no training-only approximation of live fields;
- no process-global history cache whose reset/update semantics differ between domains;
- no hidden RocketSim-only actor feature;
- no opponent information unavailable through RLBot v5;
- no feature solely because an old checkpoint expected its index.

## Actor input architecture

Transport may remain one contiguous float32 vector for RLGym/trainer compatibility, but the actor must slice it by generated schema metadata:

- core self/ball/opponent/global encoder;
- shared boost-pad entity encoder plus attention/pooling;
- shared prediction-horizon encoder;
- compact controller-history encoder;
- fusion trunk.

Do not manually duplicate index constants between observation builder and model.

## Parity gate

Before serious PPO:

1. collect a large natural RLBot corpus spanning ground, wall, aerial, dodge, kickoff, boost-starved, contact and recovery states;
2. serialize canonical snapshots, including history state;
3. run the exact same `RivalObsV1` builder used by RocketSim training and RLBot deployment on those snapshots;
4. require bit-identical results where all inputs are identical;
5. separately audit canonical adapter fields against their source values;
6. verify prediction, boost timer, touch history, dodge timing and surface features explicitly;
7. no actor training is authorized while a material unexplained train/deploy feature mismatch remains.

## Schema artifact

Commit a machine-readable observation schema containing for every block/field:

- name;
- start/end index or entity shape;
- dtype;
- normalization;
- coordinate frame;
- canonical source;
- update cadence;
- reset semantics.

Hash the serialized schema and include its hash in every checkpoint/export. Any breaking field/order/normalization change creates a new observation version.
