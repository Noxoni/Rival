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

## Optional RLViser spectator

The viewer is a separate, single-environment process. It is disabled by default and is
never imported by rollout workers, verification, the RLBot transfer matrix, or PPO. It
uses one CPU inference thread, below-normal Windows process priority, and real-time
pacing; it does not render any training worker.

Install the optional dependency and pinned RLViser v0.8.2 executable without changing
the locked NumPy/RLGym closure:

```powershell
./training/install_rlviser_spectator.ps1
```

Check the viewer path without opening a window, then watch the latest local campaign
checkpoint against frozen Wisp at the checkpoint's normal four-tick cadence:

```powershell
training/.venv/Scripts/python.exe training/scripts/run_m07_rlviser_spectator.py `
  --checkpoint current --check

training/.venv/Scripts/python.exe training/scripts/run_m07_rlviser_spectator.py `
  --checkpoint current --opponent frozen-wisp --tick-skip 4
```

Use `--checkpoint frozen-wisp` for production Wisp, or pass an actor `.pt`, campaign
checkpoint directory, or TorchScript `.ts` export. `--opponent selected` renders
self-play, `--legacy-only` masks actions 90 through 157, and `--playback-speed` adjusts
wall-clock pacing. Stop an unbounded viewer with Ctrl+C.

`rlviser-py==0.6.13` is pinned intentionally: 0.6.14 requires NumPy 2.x, while the
RLGym 2.0.1 Rocket League package requires NumPy below 2. The optional installer uses
`--no-deps`, verifies that the headless environment remains on NumPy 1.26.4, and
downloads the matching RLViser v0.8.2 Windows executable only after enforcing its
pinned SHA-256. The generated 48 MB executable is ignored by Git.

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
- `configs/milestone06.json`: serious staged 100M-ceiling campaign configuration.
- `rival_training/campaign.py`: <=1M checkpointed campaign loop with a deterministic
  100-game frozen-Wisp health gate every 5M agent-steps.
- `rival_training/curriculum.py`: seeded majority-natural broad reset families.
- `rival_training/evaluation.py`: balanced deterministic headless Wisp evaluation.
- `rival_training/deployment_candidate.py`: exact opt-in candidate export; it never
  replaces the frozen production default.
- `results/`: compact measured evidence.

## Milestone 06 campaign

The required preflight order is throughput sweep, action-prior calibration, reward and
curriculum audit, headless baseline, one full-size PPO iteration, and export/runtime
parity. The committed preflight report records the measured 56-worker optimum and the
Stage A `-6` appended-action offset.

Start Stage A only after the preflight report passes:

```powershell
training/.venv/Scripts/python.exe training/scripts/run_m06_campaign.py `
  --stage stage_a --appended-offset -6
```

Resume commands are written into each compact stage report. Prior changes are accepted
only at stage boundaries. Candidate export and RLBot evaluation remain opt-in:

```powershell
training/.venv/Scripts/python.exe training/scripts/export_m06_candidate.py `
  --checkpoint <checkpoint-directory> --label <boundary-label> `
  --output training/results/milestone06/candidate_export_<boundary-label>.json

.venv/Scripts/python.exe training/scripts/run_m06_rlbot_stage_eval.py `
  --export-report training/results/milestone06/candidate_export_<boundary-label>.json `
  --games 8 --output training/results/milestone06/rlbot_<boundary-label>.json
```

## Generated artifacts

Large files remain intentionally ignored:

- `training/artifacts/bootstrap/`: direct-reconstruction actor checkpoint;
- `training/artifacts/ppo/`: post-smoke actor checkpoint;
- `training/artifacts/deploy/`: TorchScript export;
- `training/checkpoints/`: full policy, critic, and optimizer resume state;
- `training/datasets/`: reserved for later distillation/replay data.

Their relative paths, sizes, SHA-256 hashes, formats, and reproduction commands are recorded in committed result manifests. The direct reconstruction passed, so Milestone 05 generated no behavior-distillation dataset.

Milestone 05 was a training-system proof. Milestone 06 counts only the staged campaign
steps, never the bounded preflight iteration. The live bot remains frozen Wisp unless a
trained checkpoint passes the explicit final 16-game promotion battery and deployment
review; merely exporting or evaluating a candidate does not promote it.
