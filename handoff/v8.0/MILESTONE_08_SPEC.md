# Milestone 08 — transfer-safe dual-rate Rival

## Objective

Correct the failure modes proven by Milestone 07 before another serious training campaign.

M07 established three interacting causes:

1. the zero-step Wisp-compatible policy collapses when its strategic policy is queried/executed at the four-tick cadence;
2. the M06 20M actor drifted materially inside legacy actions 0..89;
3. the training observation approximation materially changes frozen-Wisp decisions relative to the live RLBot observation.

M08 must remove those causes structurally rather than compensating with more monolithic PPO.

## Non-negotiable starting state

- Start from completed M07 commit `10c41f708d6e8145bf719f8f322041e7753f6c3f` plus any legitimate newer commits.
- Preserve the frozen production `POLICY.lt` and `SHARED_HEAD.lt` hashes.
- Do not resume the rejected M06 20M actor as the base policy.
- Recreate/load the exact zero-step direct Wisp reconstruction as the strategic teacher/base.
- Production remains frozen Wisp at tick skip 8.
- Existing rejected M03/M04/M06 paths remain off by default.
- Preserve ignored checkpoints, telemetry, the paused v4 stash, and local spectator artifacts.

## Workstream A — observation contract v2

Create one explicit versioned Wisp strategic observation contract shared conceptually by training and deployment.

At minimum address the M07 policy-sensitive mismatches:

- self ETA;
- opponent ETA;
- touch/ball-touched-step semantics;
- handbrake analog/input semantics;
- jump/flip/car-state flags included in the 51-value player blocks;
- landing normal where practical;
- previous-action semantics;
- score/kickoff/pad fields without regressing already-exact features.

Port/reuse the production Wisp ETA algorithm rather than retaining the simplified training `_rough_eta`. Training may use RocketSim ball prediction as its prediction source, but the iterative ETA algorithm, cache/update semantics, boost duration, and linear ETA math must match the live contract.

Prefer a shared pure-Python/common feature kernel or canonical input adapters so the two paths cannot silently diverge again. Do not destabilize the production runtime merely to remove code duplication.

### Observation gate

Before PPO:

- build a held-live corpus of at least 1,000 natural RLBot decision states, reusing M07 data where valid;
- compare exact live strategic observations to the training/RocketSim-style reconstruction feature-group by feature-group;
- feed both tensors to frozen Wisp and measure masked first-90 policy agreement;
- target >=99% top-1 agreement;
- hard minimum to proceed: >=97% top-1 agreement, mean JS divergence <=0.002, and no known single feature group still causing >5% top-1 changes in the substitution audit;
- all directly representable/non-approximated fields must be exact within float tolerance;
- if the hard minimum is not reached, stop M08 before PPO and report the remaining mismatch rather than training through it.

## Workstream B — exact temporal action contracts

Implement explicit action schedulers rather than relying on one generic `rlbot_delay` interpretation.

Required strategic 8-tick execution window:

`[previous, previous, previous, previous, previous, new, new, new]`

Required mechanics 4-tick execution window:

`[previous, new, new, new]`

Unit-test steady-state and transition windows over multiple consecutive decisions. The strategic RocketSim path must reproduce the live production Wisp temporal schedule exactly.

## Workstream C — dual-rate architecture

Build a dual-rate agent with two distinct policy roles.

### Strategic branch

- frozen zero-step Wisp reconstruction / frozen Wisp-compatible actor;
- first 90 legacy actions only;
- queried on an 8-tick strategic clock;
- uses observation contract v2;
- executes through the exact strategic 8-tick scheduler;
- no strategic weights are trainable in M08.

### Mechanics/recovery branch

- separate trainable actor head and critic;
- queried on a 4-tick mechanics clock;
- action space initially exactly 69 choices: `PASS` plus appended action indices `90..157`;
- `PASS` means do not override the strategic executor;
- an appended choice overrides controller output for the defined four-tick mechanics window, then relinquishes control unless selected again;
- the underlying strategic schedule continues advancing even while an override is active so returning to PASS is well-defined;
- no hidden modification of strategic logits.

A broad physical eligibility mask/gate may be used to prevent clearly irrelevant overrides, but it must describe generic state classes (airborne, wall/recovery, dodge-resource, nearby-ball interaction, etc.), not named scripted mechanics. The gate must be independently disableable.

### Exact fallback invariant

With mechanics disabled, or with the mechanics actor forced to PASS at every decision:

- policy outputs, selected legacy actions, controller schedule, and live behavior must reduce to the verified strategic Z8 path;
- this is a hard pre-training gate.

## Workstream D — zero-step transfer gate

Before any PPO update:

1. prove first-90 zero-step logit parity on randomized and held live observations;
2. prove observation-contract gate;
3. prove strategic 8-tick temporal schedule parity;
4. prove mechanics-disabled/pass-only action-sequence equivalence;
5. run a bounded RLBot zero-step/pass-only battery against installed Wisp and Nexto with balanced sides.

Do not demand precise win-rate equality from four-game cells, but reject any Z4-like collapse or obvious severe regression. Record goals, goal differential, decisions, cadence, runtime health and telemetry invariants.

## Workstream E — bounded mechanics-head learning

Only after A-D pass, train the mechanics actor/critic while the strategic Wisp branch remains frozen.

M08 is not another 100M-step campaign. It is a proof that useful learning can occur without damaging the strategic baseline.

Authorized ceiling: **5 million agent-steps**.

Suggested boundaries:

- 0 / zero-step;
- 0.5M diagnostic checkpoint;
- 1M transfer checkpoint;
- 2M checkpoint if healthy;
- 5M maximum if earlier gates remain healthy.

Use the measured M06 worker optimum of 56 as the starting worker count. A short local recheck is allowed if the new dual-rate pipeline changes throughput materially; do not rerun a broad worker sweep unless needed.

Natural 1v1 must remain the majority distribution. Minority broad aerial/wall/recovery states are allowed. Named mechanic scripts/macros remain prohibited.

Keep outcome dominance. Mechanics/resource/recovery shaping must remain small, independently logged, and unable to dominate goal/concede outcome.

## Evaluation during learning

At every checkpoint record:

- headless record and goal differential versus frozen Wisp;
- PASS rate and override rate;
- appended action distribution;
- override context distribution;
- goals/concessions within short windows after overrides;
- possession/touch/recovery/boost metrics;
- policy entropy and PPO losses;
- NaN/stability checks;
- strategic branch hash/parity proof.

At 1M, and again at the final healthy checkpoint, run the bounded RLBot transfer battery against installed Wisp and Nexto with balanced sides. Do not continue through a severe real-game regression merely because RocketSim reward improves.

## Promotion

Production promotion is **not authorized in M08**. The milestone may conclude that the architecture is healthy and ready for a longer M09 mechanics campaign, but normal Rival must remain frozen production Wisp.

## RLViser

Preserve the optional M07 RLViser spectator. If low-cost, support the new dual-rate checkpoint and expose whether each displayed control segment came from strategic PASS-through or mechanics override. Viewer work must remain off the training hot path.

## Verification

At completion require:

- existing production tests;
- training tests;
- frozen policy/shared-head hash checks;
- 90-row action-prefix proof;
- observation parity report;
- temporal scheduler parity tests;
- exact fallback/pass-only proof;
- zero-step RLBot battery;
- bounded PPO save/reload/resume if training is reached;
- post-training strategic branch unchanged;
- RLBot transfer evaluation at required boundaries;
- deploy/export smoke for the dual-rate candidate without changing production defaults;
- Ruff/compileall/diff checks;
- compact committed evidence and hashes for ignored large artifacts;
- clean remote readback.

If a hard gate fails, stop at that coherent boundary, commit the diagnosis, and do not spend the remaining training budget.