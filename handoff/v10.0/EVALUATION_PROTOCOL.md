# Milestone 10 evaluation protocol

## Principle

M10 evaluates **learning progress**, not architecture validity. M09 already proved the interfaces and native transfer contract.

Use fixed protocols so changes across checkpoints mean something. Do not alter the evaluation states/seeds after seeing a result.

## Boundary evaluations

Run the complete evaluation package at approximately +5, +10, +25, +50, and +100 simulated game-hours beyond the M09 final checkpoint.

### A. Fixed deterministic behavior protocol

Reuse the exact M09 Gate 13 deterministic evaluation protocol/seeds where possible so M09-final is the baseline.

At minimum report:

- mean 3D speed;
- mean planar speed;
- mean distance to ball;
- mean boost;
- touches per 100k agent-steps;
- first jumps per 100k;
- dodges per 100k;
- aerial touches per 100k;
- recovery-like landings per 100k;
- goals/concessions when present;
- possession/control proxies already available;
- all reward components;
- the existing fixed behavior-signature vector and distribution fingerprint.

Preserve the same observation/action determinism and evaluation length as M09 unless a documented bug makes that impossible.

### B. Action/exploration health

Report from both stochastic training rollouts and deterministic evaluation:

- each analog action mean/std;
- p01/p50/p99 for throttle, steer, pitch, yaw, roll;
- fraction near analog saturation (`abs(x) > 0.95`);
- learned analog log-standard-deviations;
- all eight jump/boost/handbrake combination frequencies;
- button categorical entropy;
- standalone jump/boost/handbrake activation rates;
- consecutive held/released jump lengths where available;
- dodge/flip acquisition and use metrics;
- controller-change rate per second.

Do not force every action to remain common. The purpose is to detect numerical/exploration collapse, not impose a human mechanic distribution.

### C. PPO/training health

Record:

- cumulative agent-steps and simulated game-hours;
- rollout throughput and simulated game-hours/wall-hour metric;
- actor loss;
- critic loss;
- approximate KL if available;
- clip fraction if available;
- analog entropy;
- button entropy;
- explained variance/value diagnostics if available;
- gradient norms;
- actor/critic parameter-change magnitude;
- reward component totals and absolute totals;
- episode lengths, goals and reset-family counts.

### D. Frozen-snapshot comparison

The moving self-play opponent makes raw self-play win rate uninformative. Add a bounded frozen-checkpoint comparison protocol.

At +10 h and later, compare the current deterministic policy against:

1. the final M09 checkpoint;
2. the most recent earlier immutable M10 checkpoint;
3. optionally one older M10 checkpoint if it is cheap and useful.

Use balanced sides and fixed seeds/states. Prefer a compact headless match set rather than a large league implementation. Record wins/losses/goals/goal differential and behavior metrics, but do not overinterpret small samples.

Do not let opponent-pool engineering become the milestone.

## RLBot transfer checks

The technical train/deploy contract already passed M09. M10 therefore uses bounded live checks rather than repeating the entire M09 gate suite.

At approximately +25 h and +100 h:

- export the current checkpoint;
- run a native-1x RLBot smoke through the same `RivalObsV1`/`RivalActionV1` deployment path;
- verify 120-Hz cadence health, finite outputs, legal controller bounds, model/schema hashes, and no sustained missed ticks;
- play a small balanced context set against installed Wisp and Nexto if convenient.

Wins/losses are context only. Technical cadence/parity decides transfer health.

If the +25 h native smoke exposes a genuine deployment regression, stop rather than accumulating another 75 simulated hours on a policy that cannot transfer.

## RLViser

Keep the isolated native-120-Hz spectator compatible with every immutable boundary checkpoint.

Do not render training workers. A spectator failure is not a reason to stop PPO unless it exposes a checkpoint/runtime compatibility defect.

## What counts as progress

No single gameplay metric is required to improve monotonically.

Healthy M10 progress is evidence such as:

- contact/touch frequency rising from the M09 baseline;
- distance-to-ball/control behavior becoming more purposeful;
- jumps/dodges beginning to appear when useful;
- recovery improving;
- goals beginning to occur;
- current snapshots outperforming earlier frozen snapshots;
- new aerial/mechanic-like behaviors appearing without reward exploitation.

At +100 h, report the result honestly even if advanced mechanics have not emerged. Do not alter the final checkpoint after evaluation to make it look better.
