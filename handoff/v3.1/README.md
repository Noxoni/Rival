# Rival Handoff v3.1 — Accelerated Milestone 03 Execution

v3.1 keeps the **entire Milestone 03 gameplay experiment from v3.0** and changes only the execution strategy for live tests.

## Start here

Codex must execute:

`handoff/v3.1/CODEX_START_PROMPT.md`

Then it must read and obey the complete `handoff/v3.0/` package.

## Key change

Do not shorten five-minute natural matches simply to save time.

Instead:

- target raw Rocket League **5.0x game speed** for automated live validation;
- verify that 5x is truly active and stable before consuming the full test budget;
- keep the normal five-minute regulation clock and normal gameplay rules;
- attempt one two-lane parallel-live-match capability test;
- use concurrency 2 only if each lane has an independent RLBotServer/port and Rocket League session;
- otherwise continue sequentially at 5x without spending time engineering unsupported multi-instance behavior.

See `EXECUTION_ACCELERATION.md` for the exact contract.

## Gameplay target unchanged

Milestone 03 still changes exactly one behavior:

**challenge-commitment calibration**.

All policy intervention rules, fixtures, A/B gates, natural-match limits, and acceptance/rejection logic remain defined by `handoff/v3.0/`.

Previous handoffs remain intact and recoverable.
