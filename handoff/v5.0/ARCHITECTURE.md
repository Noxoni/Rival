# Rival v5.0 Training Architecture

## 1. Separation of concerns

Keep the existing RLBot bot and the new trainer separate.

Recommended tree:

```text
bot/                         # existing RLBot deployment baseline
training/
  README.md
  pyproject.toml or requirements files
  configs/
  env/
    build_env.py
    observations.py
    actions.py
    rewards.py
    mutators.py
    metrics.py
  teacher/
    wisp_teacher.py
    bootstrap.py
    dataset.py
  ppo/
    train.py
    checkpoint.py
  deploy/
    export.py
    inference.py
  tests/
```

Do not force training-only packages into the existing bot runtime environment. Prefer a dedicated `training/.venv` or equivalent documented environment.

`rlgym-rlbot` currently pins a specific RLBot beta version in its package metadata. Do not downgrade Rival's working RLBot runtime merely to install that wrapper. Evaluate deployment compatibility later in isolation; the existing Rival RLBot runtime can also host a trained policy directly.

---

## 2. Environment

Primary environment:

- RLGym v2
- `RocketSimEngine`
- Soccar 1v1
- one shared trainable policy for both sides during ordinary self-play
- natural goals and normal Rocket League physics
- no renderer during training

RLGym's environment is intentionally decomposed into transition engine, state mutator, observation builder, action parser, reward function, and done conditions. Keep Rival's implementations modular along the same boundaries.

### Episode/reset distribution

The default environment should be natural 1v1 play. Do **not** make exact scripted positions the primary training distribution.

Initial Milestone 05 smoke training can use ordinary 1v1 kickoff/reset flow only. The code should support a later weighted broad reset distribution without making it mandatory for the first smoke run.

Later curriculum resets may include broad randomized classes such as:

- normal/kickoff states;
- naturally sampled replay states;
- randomized wall/ceiling possession;
- randomized awkward recovery/low-boost states;
- randomized aerial possession.

These are distributions, not exact test scripts.

---

## 3. Observation strategy

### Teacher observation

Implement a Wisp-compatible teacher observation path based on the existing Rival `CustomObs` semantics and the current 432-value model input. Reuse existing normalization/constants where possible instead of re-deriving them independently.

The live Wisp observation includes ball-prediction information. In RLGym, use RocketSim/RLGym shared-info support such as `BallPredictionProvider`, or an equivalent deterministic prediction path, to supply the same semantic information.

The teacher path exists to preserve Wisp competence; it does not have to become Rival's permanent observation design.

### Student observation v1

Start Milestone 05 with a student observation that is close enough to Wisp's current semantics to make bootstrap easy. If Codex can cleanly use the same 432-value layout, prefer that for v1.

Architect the student builder so a later `RivalObsV2` can add or replace features without rewriting the trainer. Candidate future features include short state history, possession/recovery context, and richer opponent-relative geometry.

Do not add features merely because they are available; every extra observation should have a clear gameplay purpose.

---

## 4. Mechanics-capable action space

Wisp currently has 90 unique discrete actions. Rival needs to preserve those actions for teacher compatibility while adding finer control.

Create a versioned `RivalExpandedActionV1` with these invariants:

1. **Indices 0-89 are byte/value-equivalent to the existing Wisp action rows in `bot/action_parser.py`.**
2. Preserve Wisp's existing X-mirroring semantics where required so teacher action indices map to equivalent controller inputs.
3. Append only unique new actions after index 89.
4. Use `rlgym-tools` `AdvancedLookupTableAction` as the source of candidate additional actions, initially with a moderate configuration such as:
   - `torque_subdivisions=3`
   - `flip_bins=16`
   - `include_stalls=True`
5. De-duplicate against the original 90 actions and record the final table count and SHA-256/serialized fingerprint.

Against the `rlgym-tools` implementation reviewed on 2026-08-22, the expected union is approximately 158 unique actions, but Codex must compute and verify the actual table from the installed/pinned version rather than blindly asserting that count.

The expanded table is intended to provide:

- finer aerial torque choices;
- more flip directions;
- stall-capable inputs;
- the control resolution needed for advanced recoveries and reset manipulation.

Do not manually encode 'musty', 'breezi', 'zap dash', etc. as macros in this milestone.

---

## 5. Decision cadence

Wisp acts every 8 physics ticks. That is acceptable for the teacher but may be too coarse for higher-level mechanics.

Implement two explicit modes:

- `legacy8`: 8 physics ticks/action, teacher compatibility/reference;
- `mechanics4`: 4 physics ticks/action, intended Rival training cadence.

For Wisp imitation under `mechanics4`, the teacher may choose an action at its native 8-tick cadence and that action can be used as the target for two consecutive 4-tick student decisions. This gives a student the ability to reproduce Wisp behavior initially while later PPO can change control halfway through what used to be one Wisp action window.

Do not move below 4 ticks in Milestone 05 unless measurements show a concrete need. Higher action frequency increases inference/training cost substantially.

---

## 6. Trainer

Use Python `rlgym-ppo` for Milestone 05.

Required properties:

- CUDA when available;
- multi-process/headless rollout collection;
- resumable checkpoints;
- explicit config file or committed config object;
- metrics for steps/sec, cumulative timesteps, reward components, and policy/action statistics.

Do not hard-code a huge `n_proc` count. Add a short throughput benchmark across a few reasonable worker counts and select the best stable configuration on the actual machine.

Suggested bounded benchmark candidates: 8, 12, 16, 24 processes. Stop increasing when throughput stops improving or memory/CPU pressure becomes counterproductive.

Long training must be resumable and checkpointed. A crashed process should not destroy the run.

---

## 7. Deployment

Milestone 05 should produce an inference/export seam but does not need to replace the live Rival bot yet.

Preferred deployment sequence later:

```text
RLGym checkpoint
      ↓
export/loadable actor
      ↓
Rival RLBot inference adapter
      ↓
RLBot v5 match
      ↓
Nexto/Wisp/human evaluation
```

Keep model selection switchable so the original Wisp baseline remains available for A/B benchmarking.

---

## 8. Research references

- RLGym overview: https://rlgym.org/Getting%20Started/overview/
- Training guide: https://rlgym.org/Rocket%20League/training_an_agent/
- Observation builders: https://rlgym.org/Rocket%20League/Configuration%20Objects/observation_builders/
- Action parsers: https://rlgym.org/Rocket%20League/Configuration%20Objects/action_parsers/
- State mutators: https://rlgym.org/Rocket%20League/Configuration%20Objects/state_mutators/
- RLGym tools: https://github.com/RLGym/rlgym-tools
- Advanced action table: https://github.com/RLGym/rlgym-tools/blob/main/rlgym_tools/rocket_league/action_parsers/advanced_lookup_table_action.py
- RLGym PPO: https://github.com/AechPro/rlgym-ppo
- RLGym RLBot wrapper: https://github.com/RLGym/rlgym-rlbot
