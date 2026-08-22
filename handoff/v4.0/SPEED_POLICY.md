# Rival v4.0 — Accelerated Test Policy

## Conclusion

The direct game-speed mechanism is useful and should remain available. Milestone 03 measured approximately:

- 5x requested -> 4.9156 to 4.9950 simulated game seconds per wall second;
- 4x requested -> 3.9898;
- 3x requested -> 3.0067;
- 2x requested -> 2.0035.

The packet field `match_info.game_speed` remained `1.0` even while effective time clearly accelerated. Treat that field as a diagnostic, not the acceptance oracle for acceleration.

## Validation signals

A speed regime is accepted for a given test class when all of these hold:

1. measured simulated-game-seconds / wall-second is near the requested multiplier;
2. both bots remain responsive;
3. telemetry is valid and complete enough for the test;
4. policy action distribution is non-degenerate;
5. decisions per simulated game second stay within a declared tolerance;
6. most importantly for paired controlled tests, `off` vs `observe` reproducibility remains within the deterministic-pairing gate.

Do not reject a working 5x regime solely because the packet echo remains stale at `1.0`.

## Controlled deterministic tests

Establish the reproducibility fixture at 1x first.

Then test the same synchronized `off` vs `observe` fixture at 5x. If trace identity/reproducibility still passes, use 5x for the controlled suite. If it does not, step down through 4x/3x/2x and select the fastest regime that preserves the paired gate.

The speed search should be bounded and performed once per materially changed runtime, not before every experiment.

## Full natural matches

Preserve the five-minute Soccar match structure.

For future natural validation, prefer the fastest validated multiplier, targeting 5x. Continue to use:

- `skip_replays=True`;
- replay auto-save disabled;
- debug rendering `AlwaysOff`;
- performance monitor `NeverShow`;
- `wait_for_agents=True`;
- `auto_start_agents=True`;
- existing-match behavior `Restart`;
- normal Soccar boost/gravity/demolish/scoring/physics;
- normal kickoff countdowns.

Do not use the Rocket League `TimeWarp` mutator.

## Decision cadence caveat

Milestone 03 measured about 14.89 Rival decisions per simulated second at 1x and about 12.39 against Nexto at the 5x window. That is a real cadence reduction and must stay visible in reports.

For deterministic paired tests, reproducibility is the controlling criterion. For broad natural matches, record cadence and queue warnings alongside results so accelerated games are not silently treated as identical to 1x timing.

A final high-confidence acceptance gate may still include a small number of 1x matches if acceleration materially changes decision cadence or event rates.

## Parallel live matches

Milestone 03's first two-lane attempt failed because both RLBotServer instances raced onto port 23234 and both clients attached to the surviving server. The harness was later changed to wait for listener readiness but the capability was not rerun.

v4 authorizes **one** new bounded concurrency capability test after deterministic pairing work is stable:

- assign/verify unique server ports before launching clients;
- verify two distinct listening server processes;
- verify clients attach to their intended server only;
- verify two distinct `RocketLeague.exe` processes actually exist;
- verify independent telemetry session IDs advance simultaneously.

If Rocket League/Steam still permits only one game process, mark parallel live matches unsupported on this machine and stop revisiting it during this milestone.

Do not let concurrency work delay the primary deterministic-pairing objective.