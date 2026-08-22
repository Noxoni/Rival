# Rival Codex Handoff v3.2

**Handoff version:** v3.2  
**Created:** 2026-08-22  
**Supersedes for execution:** `handoff/v3.1/`  
**Gameplay target:** unchanged from v3.0 — challenge-commitment calibration only

## What changed from v3.1

v3.2 applies RLBot v5 match-configuration cleanup for automated testing after reviewing the official configuration-file documentation and current flatbuffer schema.

The gameplay experiment remains unchanged. The 5x acceleration and bounded two-lane parallelism test from v3.1 remain unchanged.

The new execution defaults are:

- preserve `skip_replays = true`;
- disable replay auto-save during automated evidence runs;
- force debug rendering completely off during automated runs;
- hide the in-game RLBot performance monitor during automated runs while still collecting programmatic warning/health evidence;
- retain bot auto-start and wait-for-agent readiness;
- retain normal kickoff countdowns for natural matches;
- retain clean `Restart` semantics between independent evidence matches;
- retain exhibition Soccar and standard gameplay mutators;
- enable state setting only as required for 5x acceleration and controlled probes.

## Version boundary

Do not rewrite or delete v3.0 or v3.1. v3.2 is a configuration/execution overlay only.
