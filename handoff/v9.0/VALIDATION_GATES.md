# Rival v9 validation gates

No serious scratch training is authorized until the interfaces that previously caused transfer failures are proven first.

The gates below are ordered. A later gate does not excuse failure of an earlier one.

## Gate 0 — preserve project history

Before implementation:

- fetch the final M08 `origin/main` boundary;
- preserve all M08 reports, checkpoints and rejected/accepted conclusions;
- preserve the existing superseded stash unless explicitly removed by the user;
- bring the `handoff/v9.0/` package from `rival-v9-scratch-design` onto the final M08 history by normal merge/rebase/cherry-pick as appropriate;
- never reset `main` backward to the branch's old design-base SHA;
- production Rival remains frozen Wisp until a future scratch checkpoint earns promotion.

## Gate 1 — action-contract implementation

Implement `RivalActionV1` exactly.

Required tests:

1. one policy action corresponds to one physics tick;
2. action transport/controller shape is exactly 8 physical fields;
3. all five analog axes independently cover `[-1,1]` without lookup quantization;
4. all 8 jump/boost/handbrake combos encode/decode exactly;
5. steer and yaw remain independent;
6. the tanh-squashed Gaussian log probability matches an independent numerical/reference calculation, including Jacobian correction;
7. categorical button log probability and entropy are correct;
8. mixed log probability equals analog sum + categorical log probability;
9. PPO backprop reproduces rollout log probabilities for stored actions within tight tolerance;
10. physical-effect masks never remove an input that can still matter and are identical in training/deployment;
11. known representative mechanics controller traces pass through the parser byte/float-identically without quantization/synthesis.

### Gradient test

For a seeded observation/action batch:

- hybrid PPO objective is finite;
- each analog mean head receives nonzero gradients when appropriate;
- each analog log-std parameter receives finite gradients;
- button logits receive finite/nonzero gradients;
- no NaN/Inf in distribution math at near-saturation actions.

## Gate 2 — canonical observation schema

Implement:

- `RivalCanonicalStateV1`;
- RocketSim adapter;
- RLBot v5 adapter;
- one shared `RivalObsV1` builder;
- generated schema/index manifest.

Hard requirements:

- no duplicated train/live feature math after canonicalization;
- every observation field has one documented source, coordinate frame, normalization, update cadence and reset rule;
- generated observation size is stable and automatically checked;
- schema hash is included in checkpoints/exports.

## Gate 3 — observation parity corpus

Collect a natural RLBot v5 corpus spanning at least:

- kickoff;
- normal ground play;
- low/high boost;
- ball contact;
- first jump hold/release;
- double jump;
- directional dodge and flip cancel;
- airborne reset/dodge-resource state where naturally available;
- wall/ceiling contact;
- awkward recovery;
- demolition/respawn if observed;
- overtime/late-clock states.

Prefer several thousand sampled ticks, not one hand-picked scenario.

For every captured RLBot tick:

1. RLBot adapter -> canonical state;
2. serialize canonical + required history/event state;
3. reload it independently;
4. shared observation builder must reproduce the same float32 observation bit-identically.

Then validate source-adapter fields against the packet itself.

### Special audits

Explicitly report:

- boost-pad timer conversion;
- jump/air/dodge state mapping;
- dodge timeout/elapsed/direction;
- current/previous controller history;
- opponent last-input history;
- touch ages/last toucher;
- match phase/time/score;
- surface geometry;
- prediction cache age and values;
- intercept proxy;
- normalization extrema and non-finite counts.

No unexplained actor-input domain mismatch is accepted.

## Gate 4 — prediction/update-cadence benchmark

Because ball prediction is relatively expensive, benchmark shared predictor refresh periods of:

- every 1 physics tick;
- every 2 ticks;
- every 4 ticks.

For each, report:

- environment throughput;
- observation-build CPU time;
- predictor CPU time;
- prediction-age distribution;
- short fixed-policy evaluation / behavioral differences where a meaningful policy exists.

Default to the fastest cadence that retains acceptable prediction freshness. v9's current expected choice is 4 ticks with an explicit age input, but it is not assumed without measurement.

## Gate 5 — one-tick timing parity

Training must reproduce the live relationship between observation, selected action, applied controller and next physics state.

Build tick-indexed traces in RocketSim and RLBot at native 1x rate.

Require:

- policy decision index increments every physics tick;
- no action repeat is hidden in the scratch parser;
- a selected action is applied according to the same RLBot one-tick delay/previous-input semantics in both domains;
- missed live packets preserve previous controller exactly as RLBot specifies;
- history ring buffers record *applied physical controllers*, not merely desired outputs.

The technical timing gate is independent of winning/losing matches.

## Gate 6 — short-horizon physics transfer audit

Use broad natural RLBot snapshots and observed controller traces. Initialize RocketSim from the closest canonical state and replay the same applied controllers.

Separate contact-free and contact windows.

Report divergence at:

- 1 tick;
- 2 ticks;
- 4 ticks;
- 8 ticks;
- 16 ticks;
- 32 ticks.

At minimum, contact-free first-four-tick behavior must be close enough that the policy is not learning an immediately different control law. Suggested initial materiality flags:

- self position p95 >5 uu by 4 ticks;
- self velocity p95 >100 uu/s by 4 ticks;
- self orientation p95 >7.5 degrees by 4 ticks;
- ball position p95 >2 uu by 4 ticks without contact.

These are diagnostic thresholds, not permission to hide known divergence just below a number. If a threshold trips, investigate before training.

Longer-horizon divergence is expected in chaotic physics and is reported rather than requiring trajectory identity.

## Gate 7 — reward cadence audit

For synthetic and natural trajectories, verify:

- goal/concede event rewards are cadence invariant;
- potential/rate shaping has comparable integrated reward per simulated second across equivalent 1/2/4-tick sampling tests;
- no 120-Hz dense term accidentally becomes 4× stronger than its 30-Hz predecessor;
- all components finite and independently logged;
- no shaping family dominates outcome-scale reward by silent accumulation;
- mirrored/inverted states produce appropriately symmetric reward.

## Gate 8 — native 120-Hz environment stress

Run a large no-learning stress test with the complete canonical observation, predictor caching, hybrid action parser and rewards.

Minimum:

- >=100,000 policy ticks across many episodes;
- every observation/action/reward finite;
- all button combos and analog axes exercised under stochastic exploration;
- no worker leaks/stalls;
- reset/history state does not bleed between episodes;
- left/right episode mirror produces valid symmetric controls/states.

## Gate 9 — worker/throughput sweep

The 56-worker M06 result is not transferable to a 120-Hz scratch environment.

Run a new sweep using the **actual v9 workload**. Start broadly around:

`16, 24, 32, 40, 48, 56`

and continue upward/downward only while useful.

Report for each:

- stable agent-steps/s;
- simulated game-seconds/s;
- simulated game-hours per wall hour;
- CPU/GPU utilization;
- memory/commit usage;
- inference batch sizes/latency;
- observation/prediction CPU share;
- PPO iteration wall time on at least the leading candidates;
- worker crashes/stalls/restart reliability.

Choose highest **reliably restartable sustained simulated game-time throughput**, not largest worker count and not a one-off peak.

## Gate 10 — trainer/backend decision

Implement the exact hybrid distribution on the proven `rlgym-ppo` path first.

Optionally spike `rlgym-learn` if it can preserve:

- exact `RivalObsV1`;
- exact `RivalActionV1` hybrid distribution and log probabilities;
- checkpoint/reload/resume semantics;
- metrics/evaluation hooks;
- Windows stability.

Bound this comparison. If `rlgym-learn` requires a large framework rewrite just to express the hybrid policy, do not block v9 on it.

If both work, select using measured total throughput including PPO updates and operational stability.

## Gate 11 — real hybrid PPO smoke

Run multiple real PPO iterations on CUDA.

Required evidence:

- rollout sampling uses the exact hybrid actor;
- GAE finite;
- actor/critic losses finite;
- analog and button policy updates nonzero;
- analog stds remain finite;
- button entropy remains finite;
- no action branch starved;
- checkpoint saves actor, critic, optimizers, trainer counters, action/obs hashes and reward/config version;
- fresh-process reload reproduces deterministic outputs exactly/tightly;
- resumed learner completes another nonzero update.

Reject zero-update or load-only smokes.

## Gate 12 — export and 120-Hz live inference

Export the actor through the selected deployment format.

On a held observation corpus require:

- finite actions/logits/distribution parameters;
- deterministic export outputs match training actor within documented numerical tolerance;
- every emitted controller field is legal;
- full observation -> inference -> controller CPU p99 <6 ms;
- actor-only CPU p99 target <2 ms and hard maximum <4 ms.

Run an opt-in RLBot native-rate process smoke and verify no sustained missed ticks.

Do not use 5x acceleration to prove native timing. 5x may be used separately for outcome throughput after the native timing gate is already clean.

## Gate 13 — scratch pilot

Only after Gates 0–12 pass, run the bounded v9 pilot authorized in `TRAINING_FOUNDATION.md`:

- maximum 2 simulated game-hours;
- checkpoints during the pilot;
- RLViser-compatible checkpoint output;
- fixed metrics for movement, touches, scores, action exploration, reward, recovery and mechanic-like events.

The pilot passes if learning is technically healthy and behavior measurably changes without domain/parser failures. It does not need to beat Wisp or Nexto.

## Gate 14 — repository/final verification

Before declaring v9 foundation complete:

- existing production Rival suite passes;
- new scratch-training tests pass;
- Ruff/compile checks pass for changed code;
- production Wisp hashes unchanged;
- normal production config still loads Wisp unless explicitly opted into scratch candidate;
- generated large checkpoints/raw rollouts remain ignored;
- committed compact evidence parses;
- no absolute machine paths or secrets in committed artifacts;
- exact hashes/sizes/reproduction commands recorded for ignored candidate artifacts;
- remote `origin/main` readback matches the reported final SHA.

No production promotion is authorized by v9.
