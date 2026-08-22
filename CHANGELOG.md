# Changelog

All notable changes to Rival are documented in this file.

## Unreleased

### Added

- Immutable local reference snapshots for installed Wisp v2-75B and Nexto, with tracked SHA-256 manifests and successful post-copy verification.
- Exact upstream Wisp, NectoFamily, and Necto source commits plus license/provenance records.
- Explicit Milestone 01 boundary: Wisp is the runnable baseline; no Nexto/Necto source or model artifact is incorporated.
- Runnable `Rival Dev` RLBot v5 source baseline with unique agent id `noxoni/rival/dev-v1`.
- Unchanged Wisp v2-75B policy/shared-head TorchScript artifacts and retained third-party notices.
- Device-resident raw/masked policy output seam, structured top-candidate inspection, confidence/margin, and exact compatibility action wrapper.
- Pure measurement-only tactical/resource metrics and toggleable JSONL decision telemetry.
- Python/model/unit verification scripts and local RLBot launch documentation.
- Recorded 2026-08-22 verification evidence, including a live RLBot v5 Rival Dev versus installed Nexto match.
- Reproducible PyInstaller/Bob packaging for a self-contained Windows x64 ZIP, including frozen-runtime self-test, clean-extraction verification, complete file-hash manifest, build provenance, and friend-facing launch instructions.
