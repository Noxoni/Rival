# Rival Codex Handoff v3.3 — Paused-Work Resume Overlay

**Handoff version:** v3.3  
**Created:** 2026-08-22  
**Supersedes for execution:** v3.2  
**Gameplay design remains:** v3.0 Milestone 03 challenge-commitment calibration

## Purpose

v3.3 exists because Codex was paused after beginning Milestone 03 locally, before any tests, probes, natural matches, commits, or pushes were run.

The paused local repository reported:

- local/origin view at pause: `5cd05a9cb1e88df65d5ad417ca6f1e8242356be7`;
- partial uncommitted work only in:
  - `bot/strategy/__init__.py`
  - `bot/strategy/challenge_commitment.py`;
- no Milestone 03 natural-match budget consumed;
- user `bot.7z` untouched and untracked.

Since that pause, the canonical remote received documentation-only execution overlays v3.1 and v3.2. The paused implementation must be preserved while those handoff commits are synchronized.

## Version boundary

Do not overwrite v3.0, v3.1, or v3.2. Do not discard the paused strategy files.

v3.3 adds only resume/synchronization instructions. Once the working tree is safely synchronized, execute v3.2 for accelerated/config-optimized testing and v3.0 for the actual gameplay experiment.

## Test execution defaults after resume

- full five-minute natural matches;
- target 5.0x Rocket League game speed after a baseline speed-integrity check;
- `skip_replays = true`;
- `auto_save_replay = false` for automated validation;
- debug rendering `AlwaysOff` where supported;
- performance monitor `NeverShow` where supported;
- normal kickoff countdowns for natural matches;
- one bounded attempt at two isolated concurrent RLBot/Rocket League lanes;
- sequential 5x fallback if true parallelism is unsupported or unstable.
