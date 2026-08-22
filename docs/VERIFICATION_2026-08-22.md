# Milestone 01 verification — 2026-08-22

This record covers the first runnable, instrumented Rival Dev baseline. It separates source/model integrity, deterministic automated checks, and a bounded live launch. It is not a claim that Rival is stronger than Wisp or Nexto.

## Environment

- Repository: `Noxoni/Rival`, branch `main`
- Operating system: Windows
- RLBot: installed RLBot v5 GUI/server at `C:\Users\patri\AppData\Local\RLBot5`
- Project runtime: CPython 3.12.13 in repository-local `.venv`
- Installed packages used for verification: RLBot 2.0.0b54, RocketSim 2.2.1, NumPy 1.26.4, Numba 0.63.1, Torch 2.13.0+cpu, pytest 9.1.1, Ruff 0.16.4
- Rival identity: display name `Rival Dev`, agent id `noxoni/rival/dev-v1`

## Reference and model integrity

The installed BotPack remained read-only. `handoff\v1.1\scripts\verify_reference_snapshot.ps1` passed against the ignored local snapshot and tracked manifest. After live verification, all eight installed Wisp/Nexto files were independently rehashed in place; every size and SHA-256 still matched `reference_manifests/v1/MANIFEST.json`.

`scripts\verify_wisp_models.ps1` passed for the incorporated, unchanged Wisp v2-75B artifacts:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `bot\models\POLICY.lt` | 7,689,613 | `1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7` |
| `bot\models\SHARED_HEAD.lt` | 5,995,907 | `3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42` |

No Nexto/Necto source, model, or weight artifact is incorporated. Exact installed and upstream provenance is recorded in `SOURCE_PROVENANCE.md`, `docs/SOURCE_PROVENANCE.md`, `reference_manifests/v1/UPSTREAM_SOURCES.json`, and `reference_manifests/v1/MANIFEST.json`.

## Automated checks

All commands were run from the repository root with the Python 3.12 `.venv`:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest --basetemp '.pytest_tmp'
& '.\.venv\Scripts\python.exe' '.\scripts\smoke_model.py'
& '.\.venv\Scripts\python.exe' '.\scripts\smoke_decision_tick.py'
& '.\.venv\Scripts\python.exe' -m ruff check bot\bot.py bot\config.py bot\backend\model.py bot\policy bot\analysis bot\telemetry tests scripts
& '.\.venv\Scripts\python.exe' -m compileall -q bot tests scripts
```

Results:

- pytest: 15 passed; two TorchScript deprecation warnings only.
- model smoke: 432-element observation accepted, 90 finite policy outputs produced, selected action matched the original Wisp compatibility wrapper, and both model hashes matched.
- full decision-tick smoke: a synthetic RLBot packet and ball prediction passed through the real RocketSim state adapter, 432-element Wisp observation builder, real Wisp model, legal-action mask, controller mapping, tactical metrics, and JSONL logger. It selected action 12 from 42 legal actions; the returned controller exactly matched the structured decision; one schema-v1 record was parsed successfully.
- targeted Ruff check over Rival-authored/modified runtime code, tests, and scripts: passed.
- compileall: passed.

A separate broad `ruff check bot tests scripts` audit reported four findings in byte-faithful, otherwise-unmodified upstream Wisp files: three in `bot/backend/gamestate/rot_mat.py` and one unused import in `bot/eta.py`. Those files were not silently rewritten during this extraction milestone; the findings are outside the Rival decision/telemetry seam and remain explicit follow-up debt.

The tests also cover exact masked argmax behavior, empty-mask fallback, illegal-action exclusion, top-candidate/probability consistency, decision metadata, missing-opponent safety, tactical/resource measurements, disabled and enabled telemetry, verbose-logit telemetry, unique bot identity, model hashes, provenance manifests, and the hard-disabled strategic-override gate.

## Live RLBot v5 launch

The repository's `bot\rival.bot.toml` was added directly through RLBotGUI; no file was copied into or modified under the installed BotPack. A standard 1v1 was launched with Rival Dev on blue and the installed Nexto reference on orange.

Read-only process inspection confirmed:

- RLBotServer launched Rocket League with `-rlbot`, controller URL `127.0.0.1:23233`, and packet send rate 240.
- RLBotServer launched `G:\dev\RLBot-Rival\.venv\Scripts\python.exe bot.py` for Rival Dev.
- the child CPython 3.12 process remained responsive and accumulated CPU time while the match ran.
- installed Nexto ran from `C:\Users\patri\AppData\Local\RLBot5\bots\bob_build_x86_64-windows\Nexto`; that reference tree was not modified.

The Rocket League spectator view showed both `Rival Dev` and `Nexto` at 100% on the RLBot performance overlay and visibly showed Rival Dev controlling its car during active play. This confirms the runnable configuration, dependency environment, model load, observation/action path, controller output, and RLBot transport survived an actual match launch.

The launch was bounded rather than treated as a performance benchmark. A later process check confirmed that Rocket League, RLBotServer, Rival's Python runtime, and Nexto had exited, while RLBotGUI remained responsive; no final score or comparative-strength claim is recorded.

The live launch was performed with telemetry at its default disabled setting, so it intentionally produced no live JSONL file. The end-to-end telemetry path was instead verified by the deterministic full decision-tick smoke test.

## Portable Windows package validation

The Windows x64 release builder produced a PyInstaller one-directory package with a release-specific `rival.bot.toml` that launches `RivalDev.exe`, not the repository `.venv`. Before the final clean-commit rebuild, the generated package was subjected to the following bounded checks:

- the frozen `RivalDev.exe --self-test` passed from the generated release directory;
- the ZIP was extracted into a clean directory and all 2,461 distributed files were accounted for by `MANIFEST.sha256` with matching hashes;
- the frozen self-test passed again from that clean extraction, loading both exact Wisp model hashes and all 16 RocketSim soccar collision meshes;
- RLBot v5 launched `G:\dev\RLBot-Rival\dist\Rival-Dev-Windows-x64\RivalDev.exe` directly through `cmd.exe /c RivalDev.exe`; there was no Python interpreter in Rival's process chain;
- the process remained responsive, accumulated CPU time, and held an established localhost connection to RLBot on port 23234;
- Rocket League entered an active Rival Dev versus Nexto match, the RLBot overlay reported both agents at 100%, and the spectator view visibly followed Rival Dev controlling its car.

The first match-start attempt was blocked before either bot launched because the already-open RLBotGUI could not reconnect to its stopped `RLBotServer`. Restarting the **RLBot v5 Launcher** created a fresh server and the packaged launch then passed. This is recorded as an RLBot application lifecycle issue, not as a Rival runtime failure. The validation remains a launch/functionality check, not a strength or performance benchmark.

The final pre-commit source gates passed with 19 pytest tests and the same two TorchScript deprecation warnings, source-runtime self-test, model smoke, full decision-tick smoke, reference-snapshot verification, Wisp-model verification, targeted Ruff, compileall, PowerShell parser checks for both release scripts, and `git diff --check`.

The final archive hash is intentionally reported outside this tracked record after rebuilding from the pushed clean commit, because embedding an archive hash in the source commit that the archive itself records would create a circular provenance dependency.

## Acceptance boundary

Verified:

- exact Wisp source baseline and model artifacts load;
- policy outputs can be inspected without changing the selected action;
- structured decisions and measurement-only metrics are available;
- telemetry is inert when disabled and records valid JSONL when enabled;
- Rival Dev runs in an actual RLBot v5/Rocket League match against installed Nexto.

Not claimed:

- bitwise comparison against the installed binary-only Wisp executable;
- live behavioral parity with installed Wisp across every observation;
- superiority over Wisp or Nexto;
- strategic improvement, training, Nexto weight mixing, or online use.
