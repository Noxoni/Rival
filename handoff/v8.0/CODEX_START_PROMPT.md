# Codex start prompt — Rival Milestone 08

Work in the canonical repository `Noxoni/Rival` and execute Milestone 08 completely.

## Authority and starting boundary

Start from completed Milestone 07 commit:

`10c41f708d6e8145bf719f8f322041e7753f6c3f`

Preserve all legitimate newer commits. Read every file under `handoff/v8.0/` before implementation. Also read:

- `docs/MILESTONE_07_RESULTS.md`
- M07 machine-readable evidence under `training/results/milestone07/`
- `docs/MILESTONE_06_RESULTS.md`
- current production `bot/obs_builder.py`, `bot/eta.py`, action parser/runtime cadence code
- current training observation, environment, policy, checkpoint and reward code

The M06 20M actor is rejected and diagnostic-only. **Do not resume it as the base policy.** Start from the verified zero-step direct Wisp reconstruction.

Do not modify or promote the frozen production Wisp default. Preserve its model/shared-head hashes and all existing rejected intervention defaults.

## Objective

Implement and validate the transfer-safe dual-rate architecture specified by v8.0:

1. repair the strategic live/training observation contract;
2. reproduce production Wisp's real 8-tick temporal action schedule in RocketSim;
3. keep the strategic Wisp branch frozen at 8 ticks;
4. add a separate trainable four-tick mechanics/recovery actor with exactly `PASS + actions 90..157`;
5. make disabled/forced-PASS mechanics reduce exactly to the healthy zero-step tick-8 strategic path;
6. only after those gates pass, run bounded mechanics-head PPO up to 5M agent-steps with RLBot transfer checks.

## Observation work comes first

Milestone 07 showed only 46.7085% frozen-Wisp top-1 agreement between live observations and the old training-style reconstruction. The dominant errors were ETA and touch/handbrake/car-state fields.

Create a versioned observation contract and fix those semantics before PPO.

In particular, replace the simplified training `_rough_eta` with a reusable implementation matching production `rough_eta`/`linear_eta`, including:

- two-pass predicted-ball lookup;
- prior cached ETA initialization/update;
- 120-Hz prediction index selection;
- projected initial velocity;
- boost duration semantics;
- identical linear ETA math.

Reconcile `ball_touched_step`, handbrake analog/input state, flip/jump flags and previous-action timing. Preserve already-good ball/pad/coordinate behavior.

Use >=1,000 held natural live observations, preferably reusing M07's 1,276-state corpus. Do not train unless:

- frozen-Wisp masked top-1 live-vs-training agreement >=97% (target >=99%);
- mean JS divergence <=0.002;
- no known single feature group still changes >5% of top-1 decisions in the substitution audit;
- directly representable fields meet exact/tolerance parity.

If this gate cannot be met, stop, commit the diagnosis and do not train.

## Temporal schedulers

Implement explicit tested schedulers.

Strategic tick-8 production semantics:

`[previous, previous, previous, previous, previous, new, new, new]`

Mechanics tick-4 semantics:

`[previous, new, new, new]`

Do not use the generic one-previous/seven-new RocketSim delay as the strategic legacy8 path.

Test consecutive decision transitions and long action traces, not just isolated windows.

## Dual-rate policy

### Frozen strategic branch

- exact zero-step reconstructed Wisp actor;
- actions 0..89 only;
- 8-tick observation/decision clock;
- exact strategic scheduler;
- no trainable strategic parameters;
- exclude strategic parameters from all optimizers and prove hashes/weights remain unchanged.

### Trainable mechanics branch

- separate actor and critic;
- 4-tick decision clock;
- exactly 69 outputs: PASS + global action indices 90..157;
- PASS leaves the strategic scheduler's controller output untouched;
- appended selection temporarily overrides the controller for the defined mechanics window;
- strategic scheduler continues advancing underneath an override;
- always keep PASS legal;
- generic physical eligibility masking is allowed, but no named mechanic macros or exact scenario scripts.

Calibrate the PASS prior from natural states so sampled overrides are small but nonzero. Do not blindly reuse the old monolithic appended bias schedule.

## Zero-step transfer gate

Before PPO:

- randomized and held-live first-90 zero-step logit parity;
- observation contract gate;
- strategic temporal parity;
- mechanics-disabled and forced-PASS equivalence;
- bounded balanced RLBot full-game control against installed Nexto and Wisp.

Reject/stop on a Z4-like cadence collapse or other severe regression. The disabled mechanics architecture must preserve the healthy tick-8 strategic baseline.

## Bounded PPO

Only if all zero-step gates pass.

Authorized M08 ceiling: **5,000,000 agent-steps**.

Boundaries:

- 500k diagnostic;
- 1M transfer boundary;
- 2M if healthy;
- 5M maximum if healthy.

Start with 56 environments, the measured M06 optimum. If the new architecture changes throughput materially, a short 48/56/64 sanity recheck is allowed. Do not run another broad worker sweep without evidence it is necessary.

Natural 1v1 remains the majority training distribution. Frozen-Wisp anchoring plus current-policy self-play is sufficient for this milestone. Broad aerial/wall/recovery reset families may remain a minority.

Keep outcome-dominant Reward V2 principles. No named-mechanic reward. Independently log mechanics/recovery/resource shaping.

## Evaluation

At every training boundary record headless frozen-Wisp results and full policy/training metrics, including:

- PASS/override probability and sampled/deterministic rate;
- appended action distribution;
- override contexts;
- short-window outcomes after overrides;
- possession/touch/boost/recovery metrics;
- policy entropy/loss, critic metrics and PPO health;
- throughput/iteration timing;
- strategic branch unchanged proof.

At 1M, if healthy, run a balanced RLBot transfer matrix against installed Nexto and Wisp. Repeat at the final healthy M08 checkpoint. Stop on severe RLBot regression even if RocketSim improves.

No production promotion is authorized.

## RLViser

Preserve the optional M07 spectator. If low-cost, make it load the dual-rate candidate and visually/logically identify PASS-through versus mechanics override. Do not put rendering in the training hot path.

## Repository/evidence discipline

Keep large checkpoints, raw rollouts and raw telemetry ignored. Commit compact JSON/Markdown evidence with paths, sizes, hashes and reproduction commands. Push stable coherent work as you progress so it is recoverable.

At completion produce `docs/MILESTONE_08_RESULTS.md`, machine-readable reports under `training/results/milestone08/`, final verification, and a clear verdict:

- `passed_ready_for_m09_long_mechanics_training`,
- `partial_architecture_pass_learning_inconclusive`, or
- `rejected/rollback` with the failed gate.

Run all existing production/training tests, new M08 tests, frozen-hash checks, action-prefix proof, Ruff, compileall, diff checks, checkpoint/export reload if training occurs, and exact remote readback.

Push all stable implementation and compact results to `origin/main`.