# Codex Start Prompt — Rival Milestone 05 v5.1

Continue development of `Noxoni/Rival` from the completed Milestone 04 boundary.

Do not return another high-level plan. Work directly in the repository, build and verify the Milestone 05 training foundation, commit stable work, and push it to `origin/main`.

## Required starting point

1. Confirm the repository is `Noxoni/Rival`.
2. Fetch `origin/main`.
3. Confirm completed v4.1 commit `80f4a24e60c9c9613322b1f46612a30ebf5b2bb4` is present in history.
4. Preserve all legitimate commits after it, including the v5.1 handoff itself.
5. Read **every file under `handoff/v5.0/`** and `handoff/v5.1/` before modifying implementation code.
6. Read `docs/MILESTONE_04_RESULTS.md` and `evidence/results/v4.1/milestone_04_decision.json` so the rejected natural adjustment is understood and not accidentally enabled or retuned.
7. Preserve the existing paused/superseded v4.0 stash and unrelated ignored evidence/backups. Do not reset, clean, squash, or rewrite history.
8. Verify frozen Wisp teacher artifacts/hashes before using them.

## Milestone boundary

Milestone 04 is complete. Its low-resource-aerial intervention was rejected after natural 5x evaluation. Both experimental live adjustment modes remain off by default. Do not rescue, loosen, retune, or extend that heuristic in this milestone.

The recurring apparent-pressure-release detector may remain as benchmark/analysis history, but it is **not** the next implementation target.

## Central objective

Execute the complete Milestone 05 design in `handoff/v5.0/`:

**Build Rival's first complete RLGym/RocketSim training foundation.**

RLGym/RocketSim becomes the primary training environment. RLBot/Rocket League remains the deployment, telemetry, and benchmark environment.

Do not add another tactical rule to the Wisp runtime.

## Required technical direction

Follow `handoff/v5.0/ARCHITECTURE.md`, `TEACHER_BOOTSTRAP.md`, `REWARD_AND_CURRICULUM.md`, and `MILESTONE_05_SPEC.md` exactly unless a verified dependency/API incompatibility requires a documented adaptation.

Key invariants:

- isolated `training/` dependency environment; do not destabilize RLBot runtime;
- Python RLGym v2 + RocketSim + `rlgym-tools` + `rlgym-ppo` first;
- natural headless 1v1 self-play as default training distribution;
- exact Wisp action rows/order/controller semantics at student indices `0..89`;
- append richer mechanics-capable actions after index 89 rather than replacing the Wisp prefix;
- support an eventual mechanics-oriented 4-physics-tick student cadence;
- use frozen Wisp as teacher/warm start when feasible;
- bound direct TorchScript reconstruction; fall back to behavior distillation rather than prolonged reverse engineering;
- winning/game outcome dominates reward; mechanics rewards are low-weight shaping for useful outcomes;
- build checkpoint save/reload/resume from the beginning;
- benchmark worker throughput rather than guessing;
- run only a bounded PPO smoke in this milestone, not the first long training campaign;
- preserve a clean inference/export seam back to RLBot.

## Mechanical capability requirement

The architecture must not structurally prevent learning:

- flip resets and useful reset follow-ups;
- ceiling resets/control;
- controlled aerial possession and air-dribble outplays;
- musty/breezi/Meeri-pop-like sequences when useful;
- wavedash/zap-dash/wall-dash-style recovery and acceleration;
- sidewall skims/recoveries;
- using flips to maintain aerial momentum and reduce boost dependence;
- retaining boost for recovery;
- rapid defensive recovery after a miss or possession loss.

Do not implement these as hard-coded macros in the RLBot runtime.

## Use v4.1 as benchmark data

The completed v4.1 run produced 16 full 5-minute natural matches at approximately 5x effective simulation speed, with 86,950 policy decisions and stable benchmark evidence against Nexto/Wisp. Preserve those results as the deployment baseline for future trained checkpoints.

Do not consume time rerunning the same baseline merely to begin Milestone 05 unless a specific deployment-parity check requires a small bounded sample.

## End-of-run requirement

Push a coherent Milestone 05 implementation and result containing at least:

- dependency/runtime versions;
- working headless RLGym/RocketSim 1v1 environment;
- expanded action count/fingerprint and exact first-90 parity proof;
- observation path and shape/finiteness proof;
- Wisp teacher-bootstrap outcome;
- throughput benchmark and selected worker configuration;
- bounded PPO update with finite metrics;
- checkpoint save/reload/resume proof;
- deployment inference smoke;
- full relevant tests/lint/compile/diff checks;
- `docs/MILESTONE_05_RESULTS.md`;
- final `origin/main` SHA;
- what remains before the first serious mechanics/self-play training run.

Do the work now. Prefer a working training system over more prose, more Wisp heuristics, or more scenario-specific test infrastructure.
