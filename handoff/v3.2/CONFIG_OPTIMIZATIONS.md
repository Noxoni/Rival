# Rival v3.2 — RLBot v5 Automated-Test Configuration

This overlay is based on the official RLBot v5 configuration-file documentation and the current `MatchConfiguration` flatbuffer schema.

## Required automated-test defaults

For automated Rival evidence/validation sessions, configure matches with the following intent:

```text
launcher                = Steam (or the existing launcher choice for lane 1)
auto_start_agents       = true
wait_for_agents         = true

game_mode               = Soccar
game_map_upk            = Stadium_P
skip_replays             = true
instant_start            = false for natural matches
existing_match_behavior  = Restart
enable_rendering         = AlwaysOff
enable_state_setting     = true only when acceleration/probes require it
auto_save_replay         = false
freeplay                 = false
performance_monitor      = NeverShow

match_length             = FiveMinutes
max_score                = Unlimited
overtime                 = Unlimited
boost_amount              = NormalBoost
boost_strength            = One
gravity                   = Default
demolish                  = Default
all other gameplay mutators = normal/default
```

The exact Python enum names must match the installed RLBot v5 package.

---

## Why these settings

### `skip_replays = true`

Keep this enabled. It removes the post-goal replay sequence without changing actual played possessions, kickoff positions, game clock, physics, boost, or scoring.

The existing Rival runner already uses this and should continue to do so.

### `auto_save_replay = false`

Automated evidence runs do not need Rocket League replay files. Milestone 02 requested replay auto-save but produced zero replay files; the raw telemetry/fixtures are the actual evidence system.

Disable replay auto-save for automated runs to avoid needless replay-save attempts and disk work.

Do not delete replay support from the project; a manual/debug mode may enable it later if visual replay review becomes useful.

### `enable_rendering = AlwaysOff`

Use the strongest current RLBot v5 debug-rendering setting for automation: `AlwaysOff`, not merely `OffByDefault`.

This ensures render attempts from any client are ignored during benchmarks and keeps graphics/debug overhead out of the test path.

Manual development/debug sessions may use another rendering mode separately.

### `performance_monitor = NeverShow`

Hide the in-game RLBot performance overlay in automated runs.

Do **not** stop collecting programmatic runtime-health evidence. Continue to capture relevant queue/missed-packet warnings, bot/process health, observed game speed, packet/decision counts, and failures in manifests/reports.

The purpose is to avoid UI/rendering overhead, not to hide performance problems.

### `wait_for_agents = true`

Keep this true. Starting a 5x match before Rival/Nexto/Wisp is ready would corrupt the comparison far more than the startup seconds saved.

### `auto_start_agents = true`

Keep this true for the standard automated runner. Rival's experiment mode and telemetry configuration are passed through process environment settings, so clean process startup remains useful and reproducible.

### `instant_start = false` for natural matches

Do **not** remove kickoff countdowns from natural-match validation just to save time.

At 5x, countdown wall-clock cost is small, while removing the countdown can alter bot initialization/kickoff timing and makes the test less representative of ordinary RLBot play.

Controlled state-setting probes may continue using instant start where their design requires it.

### `existing_match_behavior = Restart`

Keep independent evidence games clean. Do not use `RestartIfDifferent` or `ContinueAndSpawn` for natural A/B evidence runs because they can retain an existing match/session state.

The runner may reuse the same RLBotServer/Rocket League process between games, but each evidence match itself should start as a fresh match.

### `enable_state_setting`

Natural matches historically disabled state setting. v3.1 acceleration requires raw `DesiredMatchInfo.game_speed` updates, so accelerated natural runs may enable state setting solely to maintain the requested game-speed multiplier.

Do not modify car, ball, boost, score, or game clock state during a natural match.

Controlled probes can use state setting according to their existing design.

### `freeplay = false`

Keep exhibition-match semantics. Freeplay enables training/Bakkesmod behavior that is not part of the target opponent environment.

---

## Configs reviewed but intentionally not adopted

### `start_without_countdown` / `instant_start` for natural matches

Potentially saves a small amount of time, but changes kickoff timing. Not worth it once the game itself is running around 5x.

### `series_length`

RLBot exposes three-, five-, and seven-game series options. This may eventually reduce between-game setup overhead, but do not introduce it into Milestone 03 until its lifecycle/telemetry semantics are explicitly verified. Clean independent match manifests are currently more valuable than shaving a few extra seconds.

### `max_score`

Leave unlimited. Ending blowouts early would reduce the exact broad full-match coverage the project wants.

### `max_time`

Do not use the game mutator as the primary hang watchdog for Milestone 03. Keep runner/process wall-time guards instead so a normal match is not ended by a gameplay mutator.

### `possession_score`, custom scoring, altered boost/gravity/demolish/respawn

Do not use them for normal evaluation. They change game state and/or the score signal visible to learned bots, invalidating comparison with standard Soccar.

### `game_speed` mutator values

Do not use the named `TimeWarp` mutator as the 5x mechanism. v3.1 requires raw desired-match game-speed state so the multiplier can be explicitly requested and observed as approximately `5.0`.

---

## Runner verification

Codex must add/adjust tests proving the automated match configuration has these properties before launching the expensive live stages.

At minimum assert:

- goal replays skipped;
- replay auto-save disabled;
- debug rendering `AlwaysOff`;
- performance monitor `NeverShow`;
- wait-for-agents enabled;
- bot auto-start enabled;
- natural-match instant start disabled;
- natural-match behavior `Restart`;
- natural gameplay mutators remain standard;
- accelerated natural runs allow state setting only for the 5x speed mechanism;
- controlled probe behavior remains compatible with its existing state-setting requirements.
