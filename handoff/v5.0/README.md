# Rival Handoff v5.0 — RLGym Training Foundation

This handoff is the architectural pivot from **patching Wisp behavior inside RLBot** to **training Rival as its own policy in RLGym/RocketSim**, while preserving the existing RLBot/Wisp implementation as the deployment baseline and teacher.

## Status

- Milestone 03 challenge calibration remains rejected and disabled.
- v4.1 natural accelerated gameplay remains valuable as a real-match benchmark/telemetry source.
- If a v4.1 Codex run is still active, allow it to finish or stop at a coherent pushed checkpoint before starting v5.0.
- Do not reset, rewrite, or discard v4.1 results.

## v5.0 objective

Build a **working, resumable, headless 1v1 training stack** under `training/` using:

- RLGym v2
- RocketSim
- `rlgym-tools`
- Python `rlgym-ppo`
- the frozen Wisp v2-75B TorchScript models as the initial teacher/knowledge source
- the existing Rival RLBot code as the deployment/evaluation target

The first milestone is infrastructure plus a bounded teacher/bootstrap and PPO smoke run. It is **not** an open-ended billion-timestep training run.

## Core design

```text
existing Wisp/Rival RLBot baseline
             │
             ├── frozen teacher / benchmark
             │
             ▼
       RLGym + RocketSim
             │
     mechanics-capable policy
             │
    natural 1v1 self-play PPO
             │
             ▼
        Rival checkpoint
             │
             ▼
       RLBot deployment
             │
             ▼
     Nexto/Wisp/live benchmark
```

RLGym/RocketSim is the training environment. RLBot/Rocket League is the proving ground.

## Non-negotiable principles

1. **Natural play is primary.** Do not make exact hand-authored situations the main training set or acceptance gate.
2. **Winning and useful possession remain dominant.** Do not create a freestyle reward farm.
3. **Mechanics are means, not ends.** Flip resets, ceiling resets, wavedash/zap-dash/wall-dash behavior, musty/breezi-style movement, etc. should become useful because they improve scoring, possession, aerial efficiency, or recovery.
4. **Preserve Wisp knowledge.** Do not train a replacement from random weights without first attempting the bounded Wisp bootstrap path in `TEACHER_BOOTSTRAP.md`.
5. **Expand action capability.** Rival must not remain permanently constrained to Wisp's 90 coarse actions or eight-tick decision cadence if that prevents high-level mechanics.
6. **Keep training isolated from production.** Do not downgrade or destabilize the existing RLBot environment to satisfy training dependencies.
7. **Checkpoint everything important.** Long training must be resumable; model artifacts may remain Git-ignored, but configs, manifests, metrics, hashes, and promotion decisions belong in Git.

## Read order

1. `VERSION.md`
2. `ARCHITECTURE.md`
3. `TEACHER_BOOTSTRAP.md`
4. `REWARD_AND_CURRICULUM.md`
5. `MILESTONE_05_SPEC.md`
6. `CODEX_START_PROMPT.md`

## Research basis

Verified 2026-08-22:

- RLGym overview: https://rlgym.org/Getting%20Started/overview/
- RLGym training guide: https://rlgym.org/Rocket%20League/training_an_agent/
- RLGym reward functions: https://rlgym.org/Rocket%20League/Configuration%20Objects/reward_functions/
- RLGym state mutators: https://rlgym.org/Rocket%20League/Configuration%20Objects/state_mutators/
- RLGym action parsers: https://rlgym.org/Rocket%20League/Configuration%20Objects/action_parsers/
- rlgym-tools: https://github.com/RLGym/rlgym-tools
- rlgym-ppo: https://github.com/AechPro/rlgym-ppo
- RLGym/RLBot v5 wrapper: https://github.com/RLGym/rlgym-rlbot

`rlgymppo_rs` was also reviewed as a promising later high-throughput/transfer-learning option, but **do not port Milestone 05 to Rust** unless Python RLGym-PPO is proven to be the limiting factor: https://github.com/VirxEC/rlgymppo_rs
