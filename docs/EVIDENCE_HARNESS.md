# Rival Milestone 02 evidence harness

The harness measures the frozen Milestone 01 Wisp-derived policy. It does not add strategic
overrides or alter observations, action masks, logits, deterministic argmax selection, tick skip,
or action delay.

## Prerequisites

- Windows with Rocket League and RLBot v5 installed.
- The repository virtual environment populated from `requirements.txt`.
- Installed Nexto and Wisp v2-75B snapshots matching `reference_manifests/v1/MANIFEST.json`.

Reference configs and executables are validated in place. The harness never writes into the
installed BotPack tree. Session-specific environment values are attached to Rival's RLBot v5
`CustomBot` configuration in memory.

## Validate configuration and policy freeze

```powershell
.\.venv\Scripts\python.exe scripts\verify_policy_freeze.py
.\.venv\Scripts\python.exe scripts\generate_match_config.py --opponent nexto --rival-team blue
.\.venv\Scripts\python.exe scripts\generate_match_config.py --opponent wisp --rival-team orange
```

The policy-freeze check compares policy-critical files and the exact observation/mask/inference/
selection block in `RivalBot.update_action` against commit
`4f2b21c00e2fcb7108ab1006fd950b066fbd0484`.

## Natural matches

```powershell
.\.venv\Scripts\python.exe scripts\run_evidence_suite.py natural --opponent nexto --rival-team blue
.\.venv\Scripts\python.exe scripts\run_evidence_suite.py natural --opponent wisp --rival-team orange
```

Steam is the default launcher; use `--launcher epic` or `--launcher no-launch` when appropriate.
Each invocation uses `rlbot.managers.MatchManager`, standard Soccar physics, `Stadium_P`, five
minutes, Default game speed, normal boost, and a unique session directory.

On the current RLBot v5 desktop build, a stale server connection can require restarting the RLBot
app before a new game. This is an RLBot app lifecycle failure, not evidence that Rival crashed. The
repository runner starts and owns its `MatchManager` server where possible; if startup still fails,
restart RLBot v5 and rerun the same command. Never substitute the legacy RLBotGUIX launcher.

## Controlled probes

The fake-challenge suite runs five repetitions of every required behavior by default:

```powershell
.\.venv\Scripts\python.exe scripts\run_evidence_suite.py probe-fake --rival-team blue
```

Behaviors are `true_commit`, `boost_then_brake`, `boost_then_veer`, `jump_fake`, and
`delayed_challenge`. A single behavior can be selected with `--behavior`.

The resource-aerial command runs the documented compact grid over boost, ball height/distance,
velocity, pressure, field position, and ground-alternative availability:

```powershell
.\.venv\Scripts\python.exe scripts\run_evidence_suite.py probe-aerial --rival-team blue
```

The deterministic probe opponent and state generators are measurement tools. They do not feed
labels or detector thresholds into Rival's controller path.

## Raw evidence and analysis

Each session writes ignored local artifacts beneath `evidence/raw/<session-id>/`:

- `session_start.json`: metadata passed to Rival before launch;
- `decisions.jsonl`: telemetry schema v2 start/decision/end records;
- `session_manifest.json`: final score/status, schedule, raw hash, replay pointers, and anomalies.

Analyze a file, session directory, or complete raw tree:

```powershell
.\.venv\Scripts\python.exe -m tools.evidence.analyze evidence\raw --output-dir evidence\reports\current --format both
.\.venv\Scripts\python.exe -m tools.evidence.analyze evidence\raw --output-dir evidence\reports\current --curate fixtures\evidence
.\.venv\Scripts\python.exe -m tools.evidence.analyze evidence\raw --output-dir evidence\results\v2 --curate fixtures\evidence --max-events-per-class 10
```

The analyzer supports legacy schema v1 input where practical, splits sequences at score/reset/time
rewind boundaries, persists all detector parameters, and ranks these candidate classes:

- `resource_stressed_aerial`
- `boost_detour_possession_loss`
- `apparent_vs_actual_challenge`

A detector hit is explicitly a candidate for review, never a confirmed defect by itself. Curated
fixtures are created only for event classes actually observed in live evidence. Controlled
state-setting sessions are routed only to their matching probe-backed detector; discontinuous
state resets are not treated as boost pickups or unrelated challenge events.
