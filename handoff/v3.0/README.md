# Rival Handoff v3.0 — Challenge-Commitment Calibration

Milestone 02 is complete. Rival now has a verified six-match natural baseline, controlled fake-challenge/resource probes, schema-v2 telemetry, ranked event extraction, and compact fixtures.

This handoff begins **Milestone 03: the first gameplay correction**.

## Start here

Codex must execute:

`handoff/v3.0/CODEX_START_PROMPT.md`

and treat these files as the implementation/acceptance contract:

- `handoff/v3.0/MILESTONE_03_SPEC.md`
- `handoff/v3.0/CHALLENGE_CALIBRATION_DESIGN.md`
- `handoff/v3.0/EXPERIMENT_MATRIX.md`

## Required starting point

`e7b68c6e33faf6fc644a3fc9a07e811d43d2918e`

This is the completed Milestone 02 evidence baseline. The original Wisp-equivalent gameplay baseline remains recoverable at:

`4f2b21c00e2fcb7108ab1006fd950b066fbd0484`

## Evidence selecting this target

Milestone 02 ran exactly six natural matches plus controlled probes. The apparent-vs-actual challenge detector produced 870 candidates, and the highest-ranked natural windows repeatedly showed the same sequence against Nexto: apparent closing pressure, closing signal abort, Rival jump/release behavior, then opponent next touch.

The controlled suite also established an important caveat: all 20 fake-pressure cases produced the deliberately broad release-like signal, but Rival still obtained the next touch in 17/20. Therefore Milestone 03 must **not** implement a blanket anti-flick or anti-jump rule.

## Milestone 03 idea

When Rival plausibly controls the ball and the opponent presents **ambiguous** pressure, allow a very short commitment-sensitive delay before a possession-releasing jump. During that delay Rival should choose the highest-scoring suitable action already present in Wisp's legal discrete action space, then reassess on the next policy tick.

- If the opponent becomes physically committed, release normally.
- If the opponent brakes, veers, or otherwise aborts, preserve possession where the Wisp alternatives support it.
- If the Wisp policy strongly prefers the jump or the state is unsafe to delay, do not intervene.

This is a policy re-ranking experiment, not a replacement hand-coded bot.

## Out of scope

Do not fix boost greed, aerial resource feasibility, unrelated Wisp lint debt, or train a new model during this milestone. Those remain separate evidence-backed workstreams.