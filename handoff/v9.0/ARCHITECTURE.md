# Rival v9 architecture — scratch native-control policy

## Objective

Build Rival as an independent high-end 1v1 Rocket League policy whose strategy and mechanics are learned together from scratch.

The policy is not constrained by Wisp's network, observation layout, cadence, or action lookup table.

End-to-end path:

`RLBot/RocketSim state -> RivalCanonicalStateV1 -> RivalObsV1 -> RivalPolicyV1 -> RivalActionV1 -> controller -> next 1/120-s physics tick`

Wisp and Nexto exist outside this graph as opponents/evaluation anchors.

## 1. Environment and cadence

- ordinary Soccar physics at 120 Hz;
- actor runs once per physics tick;
- no `RepeatAction` in the scratch path;
- one selected controller row corresponds to one physics tick;
- model the live RLBot one-tick input-delay relationship exactly;
- natural 1v1 is the dominant training distribution.

Track both **agent decisions** and **simulated game seconds/hours**. At 120 Hz, raw step counts are four times larger than the old 30-Hz M06/M08 steps for the same game time, so future budgets must not compare raw agent-step totals without converting to physical time.

## 2. Canonical state and observation

`RivalCanonicalStateV1` is the domain boundary. RocketSim and RLBot have separate thin source adapters, followed by one shared observation implementation.

The actor uses `RivalObsV1` from `RIVAL_OBS_V1.md`.

No trainer or deployment path may independently recreate feature logic after this point.

## 3. Actor architecture

`RivalPolicyV1` is feed-forward but structurally aware of the observation schema.

### Why not recurrent in v1

The policy already receives:

- explicit RL jump/dodge state and timers;
- latest physical state;
- one-tick motion deltas;
- eight ticks of Rival controller history;
- eight ticks of opponent controller history;
- touch/event context.

This covers the short hidden history most relevant to frame-level mechanics while preserving simple, fast PPO batching and stateless 120-Hz deployment.

A GRU/LSTM/transformer memory may be evaluated later only if held-state analysis demonstrates material perceptual aliasing or a clear performance ceiling. Adding recurrence would create `RivalPolicyV2`, not silently alter v1.

### Logical encoders

Exact dimensions come from the generated `RivalObsV1` schema rather than duplicated constants.

#### Core encoder

Input:

- match/control context;
- Rival car block;
- opponent block;
- ball/goal block;
- one-tick motion deltas.

Default network:

`LayerNorm -> Linear(512) -> SiLU -> Linear(512) -> SiLU -> Linear(384) -> SiLU`

#### Boost-pad entity encoder

Each of 34 pads is encoded by one shared pad MLP:

`pad_features -> Linear(64) -> SiLU -> Linear(64) -> SiLU`

Pool the pad entities with learned query attention conditioned on the core embedding, plus a simple mean/max summary as a robust fallback signal.

Target pooled width: 128.

Do not assign unique neural weights to pad index merely because the order is fixed; geometry is already part of each entity.

#### Prediction encoder

Each of the six prediction horizons is encoded through one shared MLP with a normalized horizon/time embedding:

`prediction_features + horizon -> Linear(64) -> SiLU -> Linear(64) -> SiLU`

Preserve chronological order and pool with a small attention/weighted temporal encoder. Target width: 128.

#### Controller-history encoder

Combine self and opponent controller rows per historical physics tick, yielding an ordered 8-step sequence.

Default encoder:

- 1D temporal convolution `16 -> 64`, kernel 3;
- SiLU;
- 1D temporal convolution `64 -> 96`, kernel 3;
- SiLU;
- concatenate newest-step, mean and max summaries;
- project to 128.

All history is past/current-known data; no causal leakage exists because future ticks are not present.

### Fusion trunk

Concatenate:

- core embedding ~384;
- pad embedding ~128;
- prediction embedding ~128;
- history embedding ~128.

Default fusion:

`Linear(768 -> 768) -> SiLU -> Linear(768 -> 512) -> SiLU -> Linear(512 -> 512) -> SiLU`

Use LayerNorm at logical boundaries where measured training stability benefits; do not add BatchNorm because rollout/inference batch statistics differ between training and live deployment.

### Actor heads

From the 512-wide fused representation:

- analog mean head: 5 outputs;
- analog log-standard deviations: five trainable bounded parameters initially, independent of state;
- button-combo head: 8 logits.

The resulting hybrid action distribution is defined in `RIVAL_ACTION_V1.md`.

Initial target actor size is approximately 1–4M trainable parameters. Exact size is a result, not a requirement. The deployment latency gate is more important than parameter count.

## 4. Critic architecture

Use a separate critic network with no actor parameter sharing in v1.

Reasons:

- actor representations are highly sensitive to mechanics/control gradients;
- value loss can be much larger and noisier;
- separate networks avoid critic optimization silently altering the deployment actor;
- critic is training-only, so it does not affect live inference latency.

The first implementation should consume the **same deployable RivalObsV1** as the actor. Do not give the critic simulator-private privileged information in v1; doing so adds another domain-specific code path before the basic scratch architecture is validated.

The critic may reuse the same encoder topology with independent weights and a scalar value head.

## 5. Policy initialization

Scratch does not mean pathological random control.

Use initialization that leaves every control reachable while keeping the initial car approximately stable:

- analog mean head centered near zero;
- Kaiming/orthogonal initialization in hidden layers;
- initial analog standard deviation calibrated by a short random-policy rollout rather than guessed permanently;
- button categorical gives modest extra prior to no-buttons, but every one of the 8 button combinations has finite, measurable probability;
- no action dimension is initialized to an effectively unreachable probability.

Before PPO, log:

- each analog axis mean/std/quantiles;
- fraction near ±1 saturation;
- all 8 button-combo probabilities and sampled frequencies;
- total mixed entropy;
- jump/boost/handbrake physical effect rates in random rollouts.

## 6. Symmetry augmentation

Team inversion is deterministic: Rival always attacks +Y in its actor frame.

Left/right reflection augmentation is allowed and desirable, but the mirror choice is **episode-stable**.

When reflected:

- canonical X coordinates/sign-sensitive orientation and angular values are reflected consistently;
- controller steer/yaw/roll are transformed with the matching physical reflection;
- the same mirror bit persists through the entire episode.

Do not flip the coordinate convention when Rival crosses X=0. M07 showed that avoiding unnecessary domain transforms is valuable, and a state-dependent mirror introduces an artificial discontinuity.

## 7. Deployment architecture

The scratch candidate is opt-in until promotion.

Deployment pipeline:

1. RLBot v5 GamePacket + FieldInfo are converted to `RivalCanonicalStateV1`;
2. history/event state machines update once per packet/physics tick;
3. shared `RivalObsV1` builder emits the actor observation;
4. exported `RivalPolicyV1` runs on CPU by default;
5. deterministic controller is produced by tanh(mean) + categorical argmax;
6. controller is sent for that tick.

### Latency budget

At native 120 Hz the total frame budget is ~8.33 ms.

Gate targets on the deployment machine:

- actor-only CPU p99: **<2.0 ms target**, <4.0 ms hard maximum;
- complete observation + actor + controller p99: **<6.0 ms**;
- no sustained missed RLBot ticks in native-rate matches.

If the initial structured actor misses these gates, optimize implementation/export/feature caching before reducing control cadence.

### Export format

PyTorch 2.13 deprecates the old TorchScript path. v9 should benchmark a modern portable export path (`torch.export`/AOT/ONNX Runtime where practical) against the known TorchScript seam.

The requirement is not a fashionable format. The chosen export must provide:

- exact or tightly bounded numerical parity;
- reliable Windows CPU loading;
- p99 latency under the 120-Hz budget;
- versioned artifact hashes;
- no dependency on the training virtual environment.

## 8. Visualizer

Retain the optional RLViser spectator from M07.

For scratch training, the spectator runs an independent environment/process at human-viewable speed and periodically loads a chosen/current checkpoint. It must not render the rollout workers.

The viewer should optionally display debug text for:

- checkpoint/game-hours;
- five analog outputs;
- jump/boost/handbrake combo;
- boost/dodge state;
- policy entropy/confidence summaries;
- most recent reward/event diagnostics.

Watching the policy is diagnostic and motivational; it is not a replacement for quantitative evaluation.

## 9. Policy versioning

The following are jointly frozen for `RivalPolicyV1` serious training:

- `RivalActionV1` schema/hash;
- `RivalObsV1` schema/hash;
- actor architecture/config;
- symmetry convention;
- canonical adapter version;
- prediction/update cadence;
- history length/reset semantics.

Breaking any of these creates a new policy contract and should be treated as requiring fresh training unless an explicit migration/distillation experiment proves otherwise.
