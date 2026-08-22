# Rival Codex Handoff v3.0

**Handoff version:** v3.0  
**Created:** 2026-08-22  
**Evidence baseline:** Rival Milestone 02  
**Required starting commit:** `e7b68c6e33faf6fc644a3fc9a07e811d43d2918e`  
**Frozen Wisp gameplay baseline:** `4f2b21c00e2fcb7108ab1006fd950b066fbd0484`

## Purpose

v3.0 is Rival's first isolated gameplay-policy experiment.

Milestone 02 established a repeatable evidence system and identified challenge-commitment calibration as the strongest first target. This version tests whether Rival can avoid premature possession-releasing responses to fake pressure while preserving fast, correct responses to genuine challenges.

## Version boundary

Preserve `e7b68c6e33faf6fc644a3fc9a07e811d43d2918e` as the recoverable pre-v3 evidence baseline. Do not squash or rewrite Milestones 01-02.

The Wisp model weights, observation semantics, legal-action mask, tick skip, and action delay remain unchanged unless an implementation bug prevents the experiment from running. The v3 intervention must be separately disableable so the exact baseline policy path remains runnable for A/B comparison.

## v3.0 scope

- implement an explainable challenge-commitment estimator;
- add a narrow policy re-ranking intervention for ambiguous challenge pressure;
- use Wisp's existing legal action space rather than generating hand-coded controller actions;
- add intervention-aware telemetry and regression fixtures;
- run controlled baseline-vs-treatment experiments first;
- run a bounded natural-match regression only if the controlled experiment passes;
- leave boost-detour and resource-aerial corrections for later versions.

## Next version

If the experiment passes, v3.1/v4 can harden the accepted challenge-calibration behavior and then move to the next evidence-backed defect. If it fails, preserve the failed experiment behind a disabled flag or revert it cleanly and report why; do not force a gameplay change merely to complete the milestone.