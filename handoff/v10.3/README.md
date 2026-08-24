# Rival Milestone 10.3 — Stage-1 anti-idle repair + progressive retry

Milestone 10.3 repeats the Milestone 10.2 progressive skill ladder, but does **not** resume the failed M10.2 Stage-1 actor.

Start from closed `main` commit `52dc505af3f43a28f6b2a5a5eb20d4d034f842d2` and from the exact clean v10.1 +10h actor:
- actor SHA-256: `e6b9fd1a38dbb5c6670711a8b47769a628d2f20caf7c88738f372d0839b7a3b6`
- checkpoint manifest SHA-256: `d1a785ef439b0127b5ab1a9ff1693ade1aa11d850151cd17b9733bbeb98dacb3`

Only Stage 1 changes:
1. `RivalBallAcquisitionRewardV2` adds a small physically-scaled anti-idle penalty before first touch.
2. `RivalBallAcquisitionCurriculumV2` makes the ordinary easy families approximately face the ball.
3. `RivalBallAcquisitionEvaluationV2` uses fresh V2 corpora and records idle telemetry.

Policy/critic architecture, `RivalObsV1`, `RivalActionV1`, canonical state, 120 Hz control, one-tick delay, no action repeat, PPO defaults, and Stage 2–4 lesson definitions remain unchanged.

Stage-1 boundaries remain:
`+1h, +2.5h, +5h, +7.5h, +10h, +12.5h, +15h`

Use the same Phase-A/Phase-B mastery thresholds and the same +5h no-material-learning stop from M10.2. If Stage 1 passes, continue Stage 2 ground control, Stage 3 aerial control, and Stage 4 finishing under the exact M10.2 protocols. Never skip a failed prerequisite.

Repeat the same 10-hour progressive wall-clock envelope. Production promotion remains forbidden.

Frozen Wisp must remain byte-identical:
- `POLICY.lt`: `1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7`
- `SHARED_HEAD.lt`: `3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42`

See `M10_3_CAMPAIGN.json` for machine-readable authority.
