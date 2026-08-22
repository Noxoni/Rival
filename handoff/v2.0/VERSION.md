# Rival Codex Handoff v2.0

**Handoff version:** v2.0  
**Created:** 2026-08-22  
**Baseline implementation:** Rival Milestone 01 / `Rival Dev`  
**Required starting commit:** `4f2b21c00e2fcb7108ab1006fd950b066fbd0484`

## Purpose

v2.0 turns the instrumented Rival baseline into a repeatable evidence system.

The previous handoff established a runnable Wisp v2-75B-derived policy with policy inspection and tactical telemetry. This version does **not** change gameplay strategy. It builds the match runner, richer RLBot v5 telemetry, event segmentation, controlled scenario probes, and baseline datasets needed to make the first defensible behavior change in the next milestone.

## Version boundary

Do not rewrite or squash the Milestone 01 baseline. Preserve commit `4f2b21c00e2fcb7108ab1006fd950b066fbd0484` as the recoverable pre-v2 baseline.

Stable Milestone 02 work must be committed and pushed to `Noxoni/Rival` with its evidence artifacts/manifests. Large raw telemetry, Rocket League replay binaries, local BotPack files, and machine-specific transient data should remain Git-ignored unless a small fixture is deliberately curated for tests.

## Next version

v3 will be the first gameplay-policy correction. It must target one defect demonstrated by the v2 evidence set and must be benchmarked against the frozen baseline before being accepted.
