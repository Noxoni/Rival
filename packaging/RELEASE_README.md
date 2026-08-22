# Rival Dev — portable Windows x64 build

This package is the self-contained Windows build of Rival Dev for **offline RLBot v5 testing only**. Do not use it for online matchmaking, cheating, or any activity that breaks Rocket League's terms of service.

## Install and play

1. Extract the complete ZIP to a normal writable folder. Do not run it from inside the ZIP.
2. Open RLBotGUI.
3. Choose **Add/Remove**, then add the extracted `rival.bot.toml` file.
4. Put `Rival Dev` on one team and the bot or human tester on the other.
5. Start the match normally.

No separate Python, virtual environment, Torch, RocketSim, or `pip install` step is required. Keep `RivalDev.exe`, `_internal`, `rival.bot.toml`, and `loadout.toml` together.

## Verify the package before RLBot

From PowerShell in the extracted folder:

```powershell
& '.\RivalDev.exe' --self-test
```

The command must finish with exit code 0 and print JSON containing `"status": "pass"` and `"frozen": true`.

`MANIFEST.sha256` covers every distributed file other than the manifest itself. `BUILD_INFO.json` records the source commit, build environment, and incorporated model hashes.

## Troubleshooting

- If Windows SmartScreen appears, compare the ZIP SHA-256 with the value supplied by the builder before choosing whether to run this unsigned development build.
- If the self-test fails, preserve its complete console output.
- If the self-test passes but RLBot reports an exit, open RLBotGUI's Events view and preserve the complete Rival traceback.
- If RLBotGUI reports that it cannot connect or reconnect to `RLBotServer` before any bot starts, fully exit and reopen the **RLBot v5 Launcher**, then start the match again. That is an RLBot app lifecycle failure, not a Rival self-test failure.
- Do not move `RivalDev.exe` away from its `_internal` directory.

The Wisp license, upstream notice, and retained README are under `third_party/wisp`. Rival incorporates no Nexto/Necto model or source artifact.
