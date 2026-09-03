# Rival 2 Unified V5 RLBot Play Build

This build deploys the validated Unified Capability V5 checkpoint as one
recurrent policy.  Natural play, aerial, offensive-demo, and dash/recovery
behavior are contained in the same network weights and GRU state.  No runtime
router, specialist selection, task identifier, or expert action splice exists.

Launch a standard offline five-minute 1v1 from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\play_rival2_unified_v5.py --human-team blue --launcher steam
```

Use `--launcher epic` for Epic Games, or `--launcher no-launch` when Rocket
League is already running under the RLBot interface.

The bot configuration can also be selected directly in an RLBot GUI:

```text
bot/rival2_unified_v5/rival2.bot.toml
```

Run the artifact/runtime smoke checks with:

```powershell
.\.venv\Scripts\python.exe bot\rival2_unified_v5\bot.py --self-test
.\.venv\Scripts\python.exe -m pytest tests\test_rival2_unified_v5.py -q
```

The runtime evaluates one policy action per unique Rocket League physics packet
at 120 Hz.  It advances one recurrent hidden state per decision and resets that
state on match initialization, countdown/kickoff lifecycle transitions, score
changes, or frame-number regression.
