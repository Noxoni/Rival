# Milestone 10 specification — sustained scratch learning

## Purpose

Milestone 09 proved that the scratch Rival interfaces, PPO implementation, checkpointing, export, native 120-Hz deployment, and RocketSim/RLBot parity are sound. Milestone 10 therefore does **not** reopen those design questions.

M10 asks one primary question:

> What does the exact proven scratch policy learn when given a meaningful amount of experience?

## Starting checkpoint

Resume the final M09 Gate 13 checkpoint described by `docs/MILESTONE_09_RESULTS.md` and its manifest.

Expected identity:

- format: `rival-v9-hybrid-ppo-checkpoint-v1`;
- cumulative steps: `1,680,214`;
- simulated hours: `1.9446921296`;
- actor SHA-256: `12770f082c6cbe1fbab8809580dc775d1d78071825eb9481df4a16d9ee85fbe5`;
- second-reload maximum parameter error: `0`;
- actor, critic, both optimizer states, and trainer counters present.

Before collecting a new step, verify the local ignored checkpoint exists and matches the committed M09 manifest. If it is genuinely unavailable or corrupt, stop and report rather than silently reconstructing or restarting from random weights.

## Frozen interfaces

The following are immutable throughout M10 unless a hard technical failure proves one is broken:

- `RivalPolicyV1` architecture;
- `RivalCriticV1` architecture;
- `RivalObsV1` and its schema hash;
- `RivalActionV1` and its schema hash;
- `RivalCanonicalStateV1` / adapter contract;
- native 120-Hz policy cadence and one-tick transport semantics;
- `RivalScratchRewardV1` / reward schedule;
- no observation standardization;
- no action lookup table;
- no RepeatAction;
- no state-dependent action mask;
- no Wisp actor or strategic branch;
- no named-mechanic macros or named-mechanic reward.

Do not tune these mid-campaign merely because early scratch behavior is weak.

## PPO configuration

Start from the exact M09 pilot values:

- gamma: `0.9987444968227265`;
- GAE lambda: `0.9872585449014338`;
- rollout agent-steps/iteration: `192000`;
- experience buffer: `600000`;
- PPO batch: `192000`;
- minibatch: `48000`;
- epochs: `1`;
- clip range: `0.2`;
- actor learning rate: `1e-4`;
- critic learning rate: `1e-4`;
- analog entropy coefficient: `0.0002`;
- button entropy coefficient: `0.001`;
- max gradient norm: `1.0`;
- observation standardization: off.

M10 is not a hyperparameter sweep. If training becomes technically unhealthy, stop at a recoverable boundary and report the evidence rather than tuning multiple knobs inside the same campaign.

## Environment and workers

Use the M09-proven native workload:

- RocketSim standard Soccar;
- 1v1;
- 120-Hz physics;
- 120-Hz policy;
- one-tick RLBot-compatible action transport;
- 56 workers.

Do not repeat the worker sweep unless the actual M10 implementation materially changes the hot path or 56 is no longer reliably launchable. M09 measured 56 as the best stable workload at 5,072.97 agent-steps/s and 64 slightly lower.

## Curriculum

Keep the exact M09 reset distribution for the full M10 campaign so learning changes can be attributed to experience rather than a moving curriculum:

- natural 1v1: 70%;
- broad ground possession/challenge: 10%;
- broad wall/aerial/ceiling possession: 8%;
- broad awkward recovery/landing: 8%;
- broad low-resource states: 4%.

The minority reset families remain randomized distributions, not exact mechanic drills. Natural play remains the majority.

Do not add Wisp imitation, behavior cloning, or a Wisp policy branch. Wisp and Nexto are benchmarks only in M10.

## Training budget

Authorize 100 **additional** simulated game-hours from the M09 final checkpoint.

At 864,000 agent-steps per simulated game-hour this is:

`86,400,000` additional agent-steps.

Nominal cumulative target:

`88,080,214` total agent-steps.

Use complete PPO iterations. A boundary may land slightly above/below its nominal step target due to worker/iteration alignment; record the exact achieved steps and simulated hours.

## Checkpoint retention

Do not retain hundreds of full immutable checkpoints unnecessarily.

- maintain a rolling recovery checkpoint frequently enough that a host/process failure loses at most a small number of PPO iterations;
- keep at least the latest two rolling recovery states;
- retain immutable full checkpoints at the evaluation boundaries;
- every immutable boundary checkpoint must include actor, critic, optimizer states, trainer counters, config/action/obs/reward versions and hashes;
- verify a fresh-process reload at every immutable boundary before continuing.

Nominal M10 added-hour boundaries:

- +5 h;
- +10 h;
- +25 h;
- +50 h;
- +100 h.

## Stop conditions

Stop training and preserve the nearest clean boundary for:

- NaN/Inf in observations, actions, rewards, advantages, losses, parameters, or optimizer state;
- checkpoint/reload mismatch;
- observation/action/reward version or hash drift;
- worker crashes/stalls that cannot be cleanly recovered;
- sustained native-controller output outside legal bounds;
- clear reward exploitation where fixed evaluation behavior improves the shaping score while materially degenerating contact/play behavior;
- unexplained loss of train/deploy parity;
- unrecoverable storage or host-resource failure.

Do **not** stop merely because:

- Rival loses games;
- deterministic jumps/dodges/aerial touches remain rare early;
- the policy looks clumsy in RLViser;
- Wisp/Nexto are much stronger;
- one evaluation metric temporarily regresses.

This is scratch learning. Weak early gameplay is expected.

## Promotion

M10 does not authorize production replacement. Production remains frozen Wisp regardless of M10 outcome.
