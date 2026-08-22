# Rival Handoff v3.2 — RLBot v5 Test-Config Cleanup

v3.2 is a configuration/execution overlay for Milestone 03.

## Start here

Codex must execute:

`handoff/v3.2/CODEX_START_PROMPT.md`

Then it must read v3.1 and v3.0 in the required order.

## What changed

After reviewing the official RLBot v5 configuration-file documentation and current flatbuffer schema, automated test sessions should:

- keep goal replay skipping enabled;
- disable replay auto-save;
- force debug rendering completely off;
- hide the performance overlay;
- keep bot auto-start and readiness waiting enabled;
- preserve natural-match kickoff countdowns;
- restart cleanly between independent evidence matches;
- preserve normal five-minute Soccar rules;
- use state setting only where needed for 5x acceleration and controlled probes.

The 5x full-match strategy and bounded two-lane parallelism test remain defined by v3.1. Challenge-calibration gameplay logic and experiment acceptance remain defined by v3.0.
