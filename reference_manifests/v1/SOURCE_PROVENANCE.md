# Reference Snapshot v1 Provenance

The tracked manifest in this directory is an exact copy of the manifest produced by `handoff/v1.1/scripts/snapshot_reference_bots.ps1` on 2026-08-22. The immutable machine-local trees remain under `.local_reference_sources/v1/` and are intentionally ignored by Git.

- Installed Wisp: `C:\Users\patri\AppData\Local\RLBot5\bots\bob_build_x86_64-windows\WispV2`
- Installed Nexto: `C:\Users\patri\AppData\Local\RLBot5\bots\bob_build_x86_64-windows\Nexto`
- Snapshot verification: `handoff/v1.1/scripts/verify_reference_snapshot.ps1` passed after the copy.
- Installed BotPack mutation: none; discovery and copying were read-only with respect to the BotPack.

Both installed references are packaged executable distributions rather than usable source trees. See `docs/SOURCE_PROVENANCE.md` and `UPSTREAM_SOURCES.json` for the exact upstream source commits, model-artifact hashes, license boundary, and baseline-selection decision.
