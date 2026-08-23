# Rival v9.0 — Native 120-Hz Scratch Policy

This package defines the next-generation Rival architecture after the M06–M08 experiments exposed the limitations of retaining Wisp as the policy skeleton.

## Bottom line

Rival v9 is a new 1v1 policy trained from scratch.

It operates directly on Rocket League's native controller every physics tick:

`RivalObsV1 -> actor -> native controller -> one 1/120-second physics tick`

There is no Wisp strategic branch, no mechanics overlay, no PASS action, no 4-tick action repeat, and no 90/158-row lookup-table ceiling.

Wisp and Nexto remain useful as opponents and fixed evaluation anchors.

## Why this package exists

The prior milestones established several facts that should not be carried into a scratch architecture:

- four-tick control changes Wisp behavior substantially even before learning;
- a frozen competent Wisp branch gives a mechanics overlay a strong incentive to defer rather than explore long mechanical sequences;
- the 158-row expanded lookup table is richer than Wisp but still samples only selected points from the controller's actual continuous space;
- train/deploy observation mismatch can completely invalidate otherwise healthy RocketSim learning.

v9 freezes the two interfaces that are most expensive to change after serious training begins: actions and observations.

## Package contents

- `VERSION.md` — design provenance and non-negotiables.
- `ARCHITECTURE.md` — overall policy/training/deployment architecture.
- `RIVAL_ACTION_V1.md` — full native controller action distribution and 120-Hz timing contract.
- `RIVAL_OBS_V1.md` — rich actor observation schema and canonical-state rules.
- `SOURCE_AUDIT.md` — feature audit across Wisp, Nexto/Necto, RLGym/RLBot and lessons retained from M07.
- `TRAINING_FOUNDATION.md` — scratch PPO/self-play/reward/curriculum plan.
- `VALIDATION_GATES.md` — pre-training, throughput, transfer, and promotion gates.
- `CODEX_START_PROMPT.md` — implementation authority to use after M08 closes.
- `PACKAGE_MANIFEST.json` — package index.

## Execution status

This branch is intentionally separate from `main` while Codex is actively completing M08. Do not merge or execute the v9 package until M08 has a clean final boundary. At that point, bring this package onto the final `main`, reconcile any legitimate newer infrastructure, and execute v9 as the new prospective direction.
