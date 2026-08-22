# Rival training foundation

This subtree is Rival's isolated RLGym v2 + RocketSim training system. It does not modify the production RLBot virtual environment or replace the deployed Wisp baseline.

The default environment is renderer-free natural 1v1 self-play at Rocket League's 120 Hz physics rate. `mechanics4` makes a student decision every four physics ticks. `legacy8` preserves Wisp's native eight-tick decision cadence.

## Install

From a PowerShell prompt at the repository root:

```powershell
./training/install_training_env.ps1
```

The installer requires Python 3.12, creates `training/.venv`, installs the resolved lock, and verifies CUDA. It reuses the repository's production Python executable only as the interpreter used to create the separate environment; package installation never targets `.venv`.

The exact dependency closure is in `requirements-lock.txt`. `rlgym-ppo` is pinned to commit `4ffd2e924198bf4b2d59f4bf280b29919d7c07ea` rather than a moving branch.

## Verify

Fast deterministic checks:

```powershell
./training/run_verification.ps1
```

Include the bounded worker sweep and three-iteration PPO save/reload/resume run:

```powershell
./training/run_verification.ps1 -IncludeMeasuredRuns
```

The production `.venv` is used only for the direct `bot/action_parser.py` prefix comparison. All RLGym work uses `training/.venv`.

## Important entry points

- `rival_training/environment.py`: natural headless 1v1 construction.
- `rival_training/actions.py`: exact Wisp prefix, 68 appended advanced actions, X mirroring, and cadence.
- `rival_training/observations.py`: finite Wisp-ordered 432-value observation path.
- `rival_training/teacher.py`: frozen-artifact hash gate and direct trainable reconstruction.
- `rival_training/rewards.py`: outcome-dominant modular reward and component accounting.
- `rival_training/ppo_smoke.py`: real `rlgym-ppo` rollout/update/checkpoint integration.
- `rival_training/deploy.py`: single-observation inference and TorchScript export seam.
- `configs/milestone05.json`: committed initial configuration.
- `results/`: compact measured evidence.

## Generated artifacts

Large files remain intentionally ignored:

- `training/artifacts/bootstrap/`: direct-reconstruction actor checkpoint;
- `training/artifacts/ppo/`: post-smoke actor checkpoint;
- `training/artifacts/deploy/`: TorchScript export;
- `training/checkpoints/`: full policy, critic, and optimizer resume state;
- `training/datasets/`: reserved for later distillation/replay data.

Their relative paths, sizes, SHA-256 hashes, formats, and reproduction commands are recorded in committed result manifests. The direct reconstruction passed, so Milestone 05 generated no behavior-distillation dataset.

This milestone is a training-system proof, not a trained gameplay release. The live bot remains frozen Wisp with both rejected Milestone 04 intervention modes off.
