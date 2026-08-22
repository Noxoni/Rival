# Rival Codex Handoff

**Package version:** v1.1  
**Repository:** `Noxoni/Rival`  
**Canonical URL:** `https://github.com/Noxoni/Rival`  
**Target:** RLBot v5 on Windows

## Changes from v1.0

v1.1 makes `Noxoni/Rival` the authoritative project repository.

- Codex must clone/use `Noxoni/Rival`; it must not initialize an unrelated repository.
- Stable implementation work, tests, provenance, and progress are committed and pushed to `main` unless a feature branch is more appropriate.
- Local installed Wisp/Nexto reference snapshots remain read-only and are **not committed wholesale** to the public repository by default.
- Hash manifests and provenance for local references are committed so the exact inputs remain traceable.
- Third-party code/model artifacts copied into Rival must retain applicable license/attribution and must not be committed until Codex verifies redistribution terms.
- Handoff instructions now assume the package lives at `handoff/v1.1/` inside Rival.

## Versioning rule

Do not overwrite prior handoff versions.

- `handoff/v1.1/` remains recoverable.
- Future handoff changes go to `handoff/v1.2/`, `v2.0`, etc.
- Keep stable implementation milestones as Git commits/tags.
- Record material implementation changes in repository `CHANGELOG.md`.

## v1.1 implementation scope

The first implementation milestone remains:

- Wisp v2-75B behavior as the initial baseline.
- Separate Rival RLBot identity.
- Policy-logit / action inspection.
- Structured decision telemetry.
- Initial tactical/resource metrics.
- No crude strategic overrides yet.
