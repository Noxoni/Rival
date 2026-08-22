# Rival evidence fixtures

This directory contains small, curated windows from real Rival Milestone 02 evidence sessions.
Each JSON file uses `fixture_schema_version: 1`, identifies the source session/event/raw SHA-256,
preserves a bounded sequence of telemetry v2 records, and labels detector output as a candidate
rather than a confirmed defect.

Fixtures are produced with:

```powershell
.\.venv\Scripts\python.exe -m tools.evidence.analyze evidence\raw --curate fixtures\evidence
```

An event class gets a fixture only after that class is actually observed. Large raw JSONL logs and
Rocket League replay binaries remain local and are referenced by hashes in committed reports.
