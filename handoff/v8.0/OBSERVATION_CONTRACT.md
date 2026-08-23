# Observation contract v2

## Purpose

Milestone 07 showed that matching the 432-value shape is not sufficient. The frozen Wisp policy changed its masked top-1 action on 680/1,276 held live states when the same RLBot packet was reconstructed through the M05/M06 training observation approximation.

M08 must make the training strategic observation behave like the live strategic observation before PPO resumes.

## Contract design

Create an explicit version such as `Wisp432ContractV2` with:

- one canonical feature ordering/schema;
- shared constants and normalization;
- shared ETA math;
- explicit adapters from live RLBot state and RocketSim/RLGym state;
- feature-group metadata mapping every index range to its semantic source;
- deterministic parity tooling that can compare two 432 tensors and attribute differences by feature group.

The live production Wisp path may remain operationally separate if necessary, but its semantics must be represented in tests and the training adapter must target those semantics.

## ETA

This is the highest-priority fix.

The current training `_rough_eta` is not equivalent to production `bot/eta.py::rough_eta`.

Port the production behavior into a reusable implementation:

1. start from the prior cached ETA for that player;
2. twice select the predicted-ball slice at `min(int(time * 120), 599)`;
3. calculate car-to-predicted-ball direction and projected initial velocity;
4. call the same `linear_eta` calculation;
5. use the same boost-duration interpretation (`boost / 33.3`);
6. preserve/reset the per-player cache with explicit episode/session semantics.

RocketSim may provide the prediction slices, but prediction sampling must line up with the live 120-Hz indices used by the ETA loop.

Validate `linear_eta` independently over broad randomized inputs against the production implementation.

## Touch and handbrake semantics

Do not use coarse state surrogates when the live Wisp observes event/input values.

Reconcile:

- live `player.ball_touched_step` versus training touch-event state;
- live analog `player.handbrake_val` versus the appropriate applied/previous controller input;
- any stateful timing needed so the value refers to the same decision moment.

These fields appear both in the top-level self-car fields and inside player blocks; verify all occurrences.

## Player flags

Audit exact semantic equivalence for:

- on-ground;
- has flip or jump;
- demoed;
- jumping;
- flip-reset/resource state;
- demo respawn timer;
- boost amount;
- wall/back-wall flags.

Where RLGym does not expose the exact live notion directly, maintain minimal adapter state rather than silently substituting a different boolean.

## Previous action

Preserve the exact controller vector observed by Wisp at that decision point, including X mirroring. The strategic branch must see the same previous-action semantics under the explicit eight-tick scheduler.

## Ball prediction and landing normal

M07 found ball prediction relatively low impact and landing normal modest impact. Preserve the existing RocketSim prediction source if parity remains good after higher-priority fixes. Improve landing-normal parity if practical, but do not block the architecture on an expensive arena-SDF rewrite unless it remains policy-material after ETA/input fixes.

## Parity corpus and reports

Use at least 1,000 held natural live states. Reuse M07's 1,276-state corpus when possible to allow before/after comparison.

Report:

- whole-vector MAE/max error;
- per-feature-group MAE/p95/max/correlation where meaningful;
- frozen-Wisp first-90 logit MAE/max;
- masked top-1 agreement;
- JS/KL divergence;
- confidence/margin differences;
- single-group substitution materiality.

The important metric is policy equivalence, not only vector MAE.

## Gate

Target:

- >=99% frozen-Wisp masked top-1 agreement.

Minimum allowed for M08 PPO:

- >=97% masked top-1 agreement;
- mean JS <=0.002;
- no known single feature group causing >5% top-1 changes in substitution analysis;
- exact/tolerance parity for all fields that are directly representable.

If this cannot be reached, stop and report the remaining domain mismatch. Do not compensate by training the policy to tolerate a known broken observation contract.