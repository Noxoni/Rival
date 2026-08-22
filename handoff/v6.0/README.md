# Rival Handoff v6.0 — Serious Training Campaign

Milestone 05 proved that Rival can train headlessly, resume PPO exactly, reconstruct Wisp into a trainable actor, expose 68 additional mechanics-capable actions, and export a deployment policy. Milestone 06 uses that system for a material training campaign.

## Central rule

**Train Rival; do not add another Wisp runtime heuristic.**

Natural 1v1 RocketSim self-play is the majority training distribution. Broad wall/aerial/recovery state enrichment is allowed as a minority distribution to expose useful parts of the state space, but exact hand-authored trick scenarios are not the main training set or acceptance gate.

## Campaign shape

- ceiling: 100,000,000 agent-steps;
- 24 workers unless a new measured instability requires fewer;
- `mechanics4` cadence;
- Wisp-derived actor initialization;
- staged opening of appended actions rather than instantly removing the `-12` prior;
- checkpoint every 1M agent-steps or safer equivalent supported by the trainer;
- compact evaluation/report boundary at least every 5M;
- headless frozen-Wisp evaluation throughout;
- RLBot Nexto/Wisp deployment evaluation at major stage boundaries;
- production promotion only after an explicit final gate.

## What we want to learn

The policy is allowed to discover whatever actions win, but the training environment and metrics must make room for:

- useful flip/ceiling resets and reset follow-ups;
- controlled air-dribble and ceiling possession rather than possession dumps;
- aerial flips used for momentum/boost conservation;
- wavedash/zap-dash/wall-dash-like acceleration and recovery;
- sidewall/awkward-surface recovery;
- fast defensive recovery after failed offense or possession loss;
- retaining enough boost to recover after aerial commitments;
- mechanically creative outplays only when they improve 1v1 outcomes.

Read `TRAINING_CAMPAIGN.md`, `REWARD_AND_CURRICULUM.md`, `EVALUATION_AND_PROMOTION.md`, then execute `CODEX_START_PROMPT.md`.
