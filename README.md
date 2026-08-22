# Rival

Rival is a high-end offline Rocket League 1v1 bot project for **RLBot v5**. The current default gameplay remains the verified Milestone 02 Wisp v2-75B-derived baseline. Milestone 03 added an experimental challenge-commitment estimator, legal-action re-ranker, schema-v3 telemetry, controlled A/B tooling, and acceleration/isolation evidence, but the gameplay experiment was rejected and remains disabled.

Rival is for offline RLBot play only. It must not be used to cheat or otherwise break Rocket League's terms of service.

## Current verified baseline

- RLBot display name: `Rival Dev`
- Agent id through Milestone 02: `noxoni/rival/dev-v1`
- Runtime config: `bot/rival.bot.toml`
- Completed Milestone 02 evidence commit: `e7b68c6e33faf6fc644a3fc9a07e811d43d2918e`
- Frozen Wisp-equivalent gameplay baseline: `4f2b21c00e2fcb7108ab1006fd950b066fbd0484`
- Baseline models: unchanged Wisp v2-75B `POLICY.lt` and `SHARED_HEAD.lt`
- Six natural baseline matches completed: three vs Nexto, three vs Wisp v2-75B
- Controlled fake-challenge and resource-aerial probes completed
- 35,230 policy decisions in the primary Milestone 02 evidence set
- Large raw evidence remains local/Git-ignored; compact reports, hashes, and fixtures are committed

See `docs/MILESTONE_02_RESULTS.md`, `evidence/results/v2/candidate_events.md`, and `docs/RUN_LOCAL.md` for the current evidence and runtime details.

## Milestone 03 result

Challenge calibration is technically implemented with explicit `off`, `observe`, and `intervene` modes, but it is **not enabled**. Neither controlled treatment parameter attempt applied an intervention, so the observed case-count differences were not causal; the coarse candidate also lost one of five true-commit next-touch controls. The controlled gate failed, no natural acceptance matches were launched, and the six-game natural budget remains unused.

The direct game-speed mechanism produced approximately 5x/4x/3x/2x effective simulated-time acceleration, but RLBot packets continued reporting `match_info.game_speed=1.0`, so the strict evidence regime fell back to 1x. The single bounded two-lane attempt also failed isolation and testing continued sequentially.

See `docs/MILESTONE_03_RESULTS.md` and `evidence/results/v3/milestone_03_decision.json` for the exact parameters, paired metrics, resume evidence, runtime warnings, speed/concurrency gates, and rejection rationale.

For another Windows PC, do not distribute only `bot/`: the development TOML depends on the repository-local `.venv`. Build the self-contained Windows x64 ZIP with `scripts/build_windows_release.ps1`; the complete procedure and verification contract are in `docs/BUILD_WINDOWS_RELEASE.md`.

## Completed Codex handoff — v3.3

Start here:

`handoff/v3.3/CODEX_START_PROMPT.md`

v3.3 governed the safe resume because Milestone 03 had paused with uncommitted local work in `bot/strategy/__init__.py` and `bot/strategy/challenge_commitment.py`. The resume preserved and restored those bytes before syncing, then applied the v3.2 config optimizations and v3.1 acceleration rules over the original v3.0 gameplay experiment.

Milestone 03 was Rival's **first isolated gameplay-correction experiment**: challenge-commitment calibration. It estimates whether opponent pressure is physically committed, briefly defers marginal possession-releasing jumps while commitment is ambiguous, and re-ranks only among Wisp's existing legal actions. It did not pass the paired controlled fake/true-challenge gate.

Automated live validation preserves full five-minute Soccar games. The v3.3 run targeted **5.0x Rocket League game speed**, then fell back to 1x when the packet-observability gate failed. Goal replays stay skipped, automated replay saving is disabled, debug rendering is forced off, the performance overlay is hidden, readiness waiting remains enabled, natural kickoff countdowns are preserved, and evidence matches restart cleanly with normal gameplay settings.

The one bounded two-lane concurrency smoke did not establish independent server ports, Rocket League processes, or telemetry. Parallel live matches are therefore unsupported for this milestone on the tested machine.

No Milestone 03 natural-match budget had been consumed when work was paused, and none was consumed after resume because the controlled gate failed. The feature retains `off`, `observe`, and `intervene` modes so the exact pre-v3 action path remains available. It remains disabled rather than being patched with unrelated gameplay rules.

Previous handoffs remain under `handoff/` as recoverable project history.

## Local RLBot BotPack references

The user's installed RLBot v5 bots are located at:

`C:\Users\patri\AppData\Local\RLBot5\bots`

Treat the installed BotPack as read-only. The primary local references are Wisp v2-75B and Nexto. Exact local and upstream provenance is recorded in `docs/SOURCE_PROVENANCE.md`; machine-local snapshots remain Git-ignored.

## Repository policy

This repository is the canonical location for Rival implementation work, tests, provenance, evidence tooling, progress, and stable commits. Meaningful completed work should be committed and pushed here rather than existing only in a local Codex workspace.
