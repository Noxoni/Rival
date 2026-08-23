# RivalActionV1 — native mechanics-complete controller contract

## Decision

`RivalActionV1` is **not a lookup table**.

The actor outputs the native Rocket League controller once per physics tick at 120 Hz. The action representation has complete support for every RLBot controller state relevant to ordinary Soccar:

1. throttle: continuous `[-1, +1]`
2. steer: continuous `[-1, +1]`
3. pitch: continuous `[-1, +1]`
4. yaw: continuous `[-1, +1]`
5. roll: continuous `[-1, +1]`
6. jump: binary
7. boost: binary
8. handbrake: binary

The production/controller order is exactly:

`throttle, steer, pitch, yaw, roll, jump, boost, handbrake`

No `RepeatAction` is used. One policy action maps to exactly one physics tick.

## Why native 120 Hz

Rocket League simulates physics at 120 Hz and RLBot v5 GamePackets are delivered at that rate when the game/client keeps up. A 4-tick repeat reduces policy frequency to 30 Hz only for computational/sample-efficiency reasons; it is not a game limitation.

For a mechanics-first scratch agent, 30 Hz throws away three of every four possible control transitions. The lost ~25 ms between decisions can matter for flip cancels, wavedashes, wall/zap dashes, stalls, reset manipulation, recovery contacts, and fine aerial orientation.

v9 therefore treats 120 Hz as the primary action contract. Throughput must be optimized around that contract rather than lowering the policy rate first.

## Hybrid PPO action distribution

The actor has one shared representation and two output branches.

### Analog branch

Five continuous dimensions represent throttle/steer/pitch/yaw/roll.

Use a tanh-squashed diagonal Gaussian:

- actor predicts five unconstrained means;
- actor owns five trainable log-standard-deviation parameters, or a carefully bounded state-dependent equivalent if later evidence justifies it;
- sample pre-squash values from `Normal(mean, std)`;
- apply `tanh` to map every analog axis to `[-1, 1]`;
- PPO log probability includes the tanh Jacobian correction;
- deterministic inference uses `tanh(mean)`.

Do not clip an ordinary Gaussian after sampling and then pretend the clipped value has the original Gaussian log probability.

### Button branch

Jump, boost and handbrake form an exact 8-state joint categorical distribution rather than three independent Bernoulli samples.

Canonical encoding:

`button_combo = jump + 2*boost + 4*handbrake`

Thus every simultaneous button combination is represented directly and can acquire correlated probability mass.

The policy outputs 8 categorical logits. Sampling chooses one combo; deterministic inference chooses argmax. The categorical log probability is added to the five analog log probabilities for PPO.

### Stored action

Experience storage and the environment use the actual 8-value controller row:

`[five analog floats, jump_bit, boost_bit, handbrake_bit]`

During PPO backprop, the three bits are converted back to the canonical combo index so the exact categorical log probability is recovered.

## Physical-effect masks

Do not shrink the controller vocabulary because a mechanic looks uncommon. A small state-dependent categorical mask is allowed only where an input is physically ineffective:

- if boost is exactly empty, mask button combos that press boost;
- jump remains legal while grounded, during the first-jump hold window, or while a dodge/double jump is available;
- when jump can have no physical effect, mask jump-pressed combos;
- do **not** mask handbrake merely because the car is airborne, because holding it through a landing can be useful for recoveries.

The mask implementation must be shared by training and deployment and covered by parity tests. If there is uncertainty about whether an input can matter, keep it legal.

## One-tick temporal semantics

RLBot v5 consumes a controller state each physics tick. If a bot misses a tick, Rocket League continues using the previous controller state.

Training must use the RocketSim/RLGym RLBot-delay path so the observation/action relationship matches deployment. With one-row actions, the selected controller from observation `t` is scheduled according to the same one-tick input-delay semantics as RLBot, not applied retroactively to the state that produced the observation.

The deployment wrapper must use the matching RLGym-RLBot step offset/configuration. The exact mapping is verified by tick-indexed traces before training.

## No artificial controller coupling

Do not carry lookup-table assumptions into the native action path.

Specifically:

- `steer` and `yaw` are separate outputs;
- throttle is not forced to equal boost;
- boost is not forbidden with reverse/neutral throttle;
- pitch/yaw/roll can take any simultaneous continuous values;
- handbrake is independent of jump;
- stalls do not require a special named action because the controller can emit the required simultaneous yaw/roll/jump values directly.

## Mechanical coverage

Because the action space is the native controller itself at native physics cadence, it can express the primitive inputs required for all of the target mechanic families without adding named macros:

- directional air-roll aerials and arbitrary tornado-spin blends;
- front/back/side/diagonal and continuously directed dodges;
- flip cancels, half-flips and speedflip input transitions;
- wavedashes and landing powerslide combinations;
- wall dashes, chain/wall recovery inputs and zap-dash-like timing;
- stalls;
- flip-reset acquisition and arbitrary reset follow-ups;
- ceiling resets and ceiling-to-air transitions;
- musty/Breezi/Meeri-pop-like rotational/flick sequences;
- air-dribble corrections and fine aerial possession;
- sidewall/ceiling recoveries;
- aerial flips used for momentum conservation and boost saving.

These mechanics remain learned sequences. There is no `musty`, `reset`, `zapdash`, or `wavedash` button.

## Coverage tests

Before PPO, implement tests proving:

1. random analog inputs throughout `[-1,1]^5` round-trip through training parser and RLBot deployment conversion within float tolerance;
2. all eight binary button combinations round-trip exactly;
3. steer/yaw independence is preserved;
4. action parser emits shape `(1,8)` for one policy step;
5. one-tick delayed RocketSim action traces match the live RLBot timing contract;
6. known controller traces representative of speedflip/flip-cancel, wavedash, stall, reset follow-up, wall dash and aerial air-roll sequences are representable without quantization or parser alteration;
7. no action-row lookup, nearest-neighbor quantization or hidden controller synthesis remains in the scratch path.

The representative traces are capability tests only. They are not macros exposed to the policy and are not scripted training scenarios.

## Initial exploration prior

A random scratch policy should not be initialized into permanent button spam, but no controller dimension may be starved.

- analog means initialize near zero with nontrivial variance;
- all 8 button combinations receive finite probability;
- `no buttons` may receive a modest initial bias for stability;
- jump, boost, handbrake and their useful combinations must still be sampled at measurable rates;
- report sampled analog distributions, saturation rates, button-combo frequencies and action entropy before training;
- do not choose a strong prior solely by intuition: run a short random-policy RocketSim calibration and record its actual effect rates.

## Version/fingerprint

The action contract must have a machine-readable metadata file containing:

- version `RivalActionV1`;
- controller field order;
- 120-Hz cadence;
- analog distribution type and bounds;
- button-combo encoding;
- physical-effect mask rules/version;
- timing/delay version;
- parser source hash;
- policy action-head source hash.

Changing any of these after serious training begins creates a new action-contract version and normally requires a new policy.
