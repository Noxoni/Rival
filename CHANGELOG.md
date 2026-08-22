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
- Milestone 03 challenge-commitment estimator with finite explainable score/state, short-history trends, abort evidence, and reset handling.
- Explicit `off`, `observe`, and `intervene` challenge-calibration modes with a one-policy-tick bounded re-ranker over existing legal Wisp actions only.
- Schema-v3 telemetry preserving baseline, hypothetical, and final actions plus gate, safety, commitment, preference-gap, and deferral explanations.
- Refined challenge detector, three curated natural Nexto fixtures, paired controlled A/B runners, direct game-speed integrity/fallback gates, and bounded two-lane isolation evidence.
- `docs/MILESTONE_03_RESULTS.md` and compact machine-readable Milestone 03 acceptance evidence.

### Changed

- Automated RLBot v5 evidence matches now skip replays, disable replay auto-save, disable debug rendering and the performance overlay, wait for agents, restart independent matches, and preserve standard five-minute Soccar rules.
- Direct accelerated simulation is requested only through `DesiredMatchInfo.game_speed`; the named `TimeWarp` mutator is not used.

### Experimental results

- Rejected Milestone 03 challenge calibration: both controlled treatment attempts had zero eligible/applied interventions, so no causal fake-pressure improvement was demonstrated; the coarse candidate also lost one true-commit next-touch case.
- Kept `RIVAL_CHALLENGE_CALIBRATION_MODE=off` and agent id `noxoni/rival/dev-v1`; zero natural acceptance matches were run and the six-game budget remains unused.
- Rejected 5x/4x/3x/2x as evidence speeds on the tested machine because packets reported `game_speed=1.0` despite correct effective wall-clock acceleration; selected 1x.
- Marked two-lane parallel live matches unsupported for this milestone after the single bounded attempt failed server-port, Rocket League-process, and telemetry isolation.
