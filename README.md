# Rival

Rival is a high-end offline Rocket League 1v1 bot project for **RLBot v5**. The current verified implementation is the Milestone 02 evidence baseline: a Wisp v2-75B-derived policy with inspectable logits/actions, schema-v2 telemetry, controlled probes, natural-match evidence collection, event extraction, and replayable compact fixtures.

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

For another Windows PC, do not distribute only `bot/`: the development TOML depends on the repository-local `.venv`. Build the self-contained Windows x64 ZIP with `scripts/build_windows_release.ps1`; the complete procedure and verification contract are in `docs/BUILD_WINDOWS_RELEASE.md`.

## Current Codex handoff — v3.3

Start here:

`handoff/v3.3/CODEX_START_PROMPT.md`

Codex must read v3.3 first because Milestone 03 was paused with uncommitted local work in `bot/strategy/__init__.py` and `bot/strategy/challenge_commitment.py`. v3.3 preserves and restores that paused work before syncing newer remote handoff commits, then layers the v3.2 config optimizations and v3.1 acceleration rules over the original v3.0 gameplay experiment.

Milestone 03 remains Rival's **first isolated gameplay correction**: challenge-commitment calibration. The experiment estimates whether opponent pressure is physically committed, briefly defers marginal possession-releasing jumps while commitment is ambiguous, and re-ranks only among Wisp's existing legal actions. It must pass paired controlled fake/true-challenge tests before bounded natural-match validation.

Automated live validation preserves full five-minute Soccar games but targets **5.0x Rocket League game speed** after an integrity check. Goal replays stay skipped, automated replay saving is disabled, debug rendering is forced off, the performance overlay is hidden, readiness waiting remains enabled, natural kickoff countdowns are preserved, and evidence matches restart cleanly with normal gameplay settings.

Codex may attempt one bounded two-lane concurrency smoke using genuinely isolated RLBotServer endpoints and Rocket League processes. If true parallel matches are unsupported or unstable, it must fall back immediately to sequential 5x execution rather than spending time on launcher workarounds.

No Milestone 03 natural-match budget had been consumed when work was paused. The feature must retain `off`, `observe`, and `intervene` modes so the exact pre-v3 action path remains available. If the experiment fails its evidence gates, it remains disabled rather than being patched with unrelated gameplay rules.

Previous handoffs remain under `handoff/` as recoverable project history.

## Local RLBot BotPack references

The user's installed RLBot v5 bots are located at:

`C:\Users\patri\AppData\Local\RLBot5\bots`

Treat the installed BotPack as read-only. The primary local references are Wisp v2-75B and Nexto. Exact local and upstream provenance is recorded in `docs/SOURCE_PROVENANCE.md`; machine-local snapshots remain Git-ignored.

## Repository policy

This repository is the canonical location for Rival implementation work, tests, provenance, evidence tooling, progress, and stable commits. Meaningful completed work should be committed and pushed here rather than existing only in a local Codex workspace.
