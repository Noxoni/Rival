# Run Rival Dev locally

Rival Dev is an offline RLBot v5 development bot. Do not use it for online matchmaking, cheating, or any activity that breaks Rocket League's terms of service.

## Prerequisites

- Windows with Rocket League and RLBot v5 installed.
- CPython 3.12. The Wisp dependency set pins NumPy 1.x and should not be installed with the machine's current Python 3.14 runtime.
- Repository checkout at any writable path; examples below use `G:\dev\RLBot-Rival`.

## Create the environment

From the repository root in PowerShell, create `.venv` with a Python 3.12 interpreter and install the development dependencies:

```powershell
$python312 = 'C:\path\to\Python312\python.exe'
& $python312 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install --upgrade pip
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-dev.txt
```

The Codex desktop runtime used for the 2026-08-22 verification provided Python 3.12 at:

`C:\Users\patri\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`

That is a machine-local convenience path, not a required project location.

## Verify before launch

```powershell
& '.\handoff\v1.1\scripts\verify_reference_snapshot.ps1'
& '.\scripts\verify_wisp_models.ps1'
& '.\.venv\Scripts\python.exe' -m pytest --basetemp '.pytest_tmp'
& '.\.venv\Scripts\python.exe' '.\scripts\smoke_model.py'
& '.\.venv\Scripts\python.exe' '.\scripts\smoke_decision_tick.py'
```

## Add Rival Dev to RLBot v5

1. Open RLBot v5.
2. Add or import the bot configuration at `G:\dev\RLBot-Rival\bot\rival.bot.toml` (adjust the repository path if needed).
3. Confirm the UI shows display name `Rival Dev` and agent id `noxoni/rival/dev-v1`.
4. Create a 1v1 match with Rival Dev on one team and installed Nexto or Wisp v2-75B on the other.
5. Keep match speed/FPS settings at the normal RLBot values used by the reference bots.
6. Start the match and confirm Rival reaches policy inference without a model, observation-shape, action-mask, RocketSim, or RLBot transport error.

For parity comparison, run Rival Dev and installed Wisp in equivalent 1v1 setups. Milestone 01 deliberately has `STRATEGIC_OVERRIDES_ENABLED = False`; telemetry never changes the selected action.

## Decision telemetry

Telemetry is off by default and produces no file. Set environment variables before launching the RLBot process to enable it:

```powershell
$env:RIVAL_TELEMETRY_ENABLED = '1'
$env:RIVAL_TELEMETRY_PATH = 'G:\dev\RLBot-Rival\telemetry\rival_decisions.jsonl'
$env:RIVAL_POLICY_TOP_N = '5'
```

The schema-v3 JSONL record includes baseline and final discrete/controller actions, legal mask, top candidates and probabilities, confidence/margin, model tick, timestamps, raw car/ball/resource state, approximate Wisp ETAs, closing velocities, boost-pad opportunities, prior action, tick-skip/action-delay state, and the complete challenge-calibration explanation. Schema-v1/v2 evidence remains readable by the offline loader.

Full raw and masked logit arrays are retained in the in-process `PolicyDecision` object but omitted from normal JSONL to limit volume. Include them in JSONL only for focused debugging:

```powershell
$env:RIVAL_TELEMETRY_INCLUDE_LOGITS = '1'
```

Unset the variables or set `RIVAL_TELEMETRY_ENABLED=0` to disable telemetry. Restart RLBot after changing process environment variables.

## Experimental challenge-calibration mode

Milestone 03 retained three explicit modes:

```powershell
$env:RIVAL_CHALLENGE_CALIBRATION_MODE = 'off'       # exact frozen Wisp selection
$env:RIVAL_CHALLENGE_CALIBRATION_MODE = 'observe'   # log hypothetical treatment; return baseline
$env:RIVAL_CHALLENGE_CALIBRATION_MODE = 'intervene' # experimental legal-action re-ranking
```

The verified/default setting is `off`. Milestone 03 rejected both tested treatment parameter sets,
so `intervene` is for controlled research only and must not be presented as an accepted bot
improvement. See `docs/MILESTONE_03_RESULTS.md`.

## Current verification boundary

Unit/static/model smoke tests do not prove live gameplay parity. On 2026-08-22, Rival Dev was also registered from this repository and observed running in a live RLBot v5 1v1 against the installed Nexto reference. That establishes runnable RLBot/Rocket League integration for the recorded environment, but it does not establish Wisp behavioral parity, superiority over Nexto, or any future training result. See `VERIFICATION_2026-08-22.md` for the exact boundary and evidence.
