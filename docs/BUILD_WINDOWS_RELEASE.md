# Build the portable Windows release

Rival's checked-in `bot/rival.bot.toml` is the source-development configuration and intentionally launches `..\.venv\Scripts\python.exe`. Do not distribute only the `bot` directory: it has no Python runtime or installed dependencies.

The portable builder freezes the tested Python 3.12 runtime, RLBot interface, Torch, RocketSim, Numba, model artifacts, and collision meshes into a Windows x64 PyInstaller one-directory bundle. It then adds a release-specific TOML whose `run_command` is `RivalDev.exe`, the loadout, third-party notices, build metadata, and a complete SHA-256 manifest before producing a ZIP.

## Build prerequisites

Create the normal repository `.venv` with CPython 3.12, then install the build dependency:

```powershell
py -3.12 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install --upgrade pip
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-build.txt
```

Or let the build script install the tested build requirements:

```powershell
& '.\scripts\build_windows_release.ps1' -InstallBuildDependencies
```

The build is pinned to PyInstaller 6.22.2. It rejects other Python minor versions and PyInstaller versions so a successful result retains a clear evidence boundary.

## Build and verify

```powershell
& '.\scripts\build_windows_release.ps1'
& '.\scripts\verify_windows_release.ps1'
```

Outputs are ignored by Git:

- `dist\Rival-Dev-Windows-x64\` — unpacked release
- `dist\Rival-Dev-Windows-x64.zip` — shareable archive
- `dist\Rival-Dev-Windows-x64.zip.sha256` — archive hash

The builder runs `RivalDev.exe --self-test` before creating the ZIP. The verifier extracts the ZIP into a clean directory, verifies that every distributed file is present in `MANIFEST.sha256` and matches its SHA-256, confirms that the release TOML has no `.venv` dependency, and runs the frozen self-test again.

## RLBot v5 test

Extract the ZIP completely. In RLBotGUI, add the extracted `rival.bot.toml`, then launch Rival Dev in an offline match. Keep the executable beside `_internal`; moving only the EXE breaks a PyInstaller one-directory application.

On the locally tested RLBot v5 installation, the launcher sometimes leaves the GUI unable to reconnect to a stopped `RLBotServer`. If that happens before a bot process starts, fully exit and reopen the **RLBot v5 Launcher** before retrying. Do not classify that reconnect failure as a Rival exit unless `RivalDev.exe` was actually launched and then returned a nonzero exit code.

The release is unsigned. Windows SmartScreen or antivirus products may inspect or warn about locally built PyInstaller executables. Verify the archive SHA-256 before running it, and do not suppress security software globally.

## Bob metadata

The repository-level `bob.toml` points the RLBot Bob builder at `packaging/rival.spec` and the canonical `bot/rival.bot.toml`. Bob rewrites the development run commands to the produced platform binaries. The locally verified artifact remains the Windows x64 package produced by `build_windows_release.ps1`; no Linux acceptance claim is made.
