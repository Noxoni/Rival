# Rival 2 RLBot deployment

`bot/rival2_live/rival2.bot.toml` is a separate RLBot v5 bot entry for the
RivalSim Rival 2 Gameplay V2 iteration-479 checkpoint. It does not replace or
modify the historical `Rival Dev` / frozen-Wisp entry.

The deployment contract is:

`RLBot v5 packet -> RIVAL2_OBS_V1 -> deterministic RIVAL2_ACTION_V1 -> RLBot controller`

The policy runs at 30 Hz and holds each emitted eight-channel controller for
four 120 Hz Rocket League physics ticks. Kickoff and goal lifecycle transitions
reset the adapter memory. The bot is intended only for offline RLBot play.

## Add to RLBot

In RLBot v5, add this configuration directly:

`G:\dev\RLBot-Rival\bot\rival2_live\rival2.bot.toml`

Select **Rival 2** as the bot on either team and start a standard 1v1. The
development configuration uses this repository's `.venv`; do not move only the
TOML or `bot.py` to another directory.

Alternatively, launch a standard five-minute Steam match directly with the
human on Blue and Rival 2 on Orange:

```powershell
.\.venv\Scripts\python.exe scripts\play_rival2.py
```

Use `--human-team orange` to swap sides or `--launcher epic` for the Epic build.

## Identity

The checked-in JSON manifest beside the TorchScript artifact records and checks:

- the source RivalSim commit and checkpoint SHA-256;
- policy iteration and cumulative samples;
- `RIVAL2_OBS_V1` and `RIVAL2_ACTION_V1` hashes;
- model size and SHA-256;
- exact deterministic export parity evidence;
- all observation normalization and boost-pad mapping data.

Run the local artifact self-test with:

```powershell
.\.venv\Scripts\python.exe bot\rival2_live\bot.py --self-test
```

## Live-packet qualification

RLBot v5 exposes authoritative aggregate `AirState.OnGround`, but it does not
expose RivalSim/RocketSim's four individual wheel-contact bits. The deployment
therefore broadcasts the aggregate ground-contact state to all four frozen wheel
fields. This is exact when all four wheels are either in or out of contact and is
an explicit approximation during partial-wheel wall, edge, and landing contacts.

Jump, air, boost and supersonic timers not carried literally in the live packet
are maintained from authoritative `frame_num`, `AirState`, controller, boost and
supersonic transitions. Ball/car kinematics, orientation, boost, jump/dodge state,
demolition timer, touch events and boost-pad state come directly from RLBot.
