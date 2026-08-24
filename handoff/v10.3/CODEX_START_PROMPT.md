# Codex start prompt — Rival Milestone 10.3

Begin from current `origin/main`. Verify the M10.2 closeout commit `52dc505af3f43a28f6b2a5a5eb20d4d034f842d2` is in history and preserve the existing paused strategy stash.

Read:
1. `handoff/v10.3/README.md`
2. `handoff/v10.3/STAGE_1_REPAIR.md`
3. `handoff/v10.3/M10_3_CAMPAIGN.json`
4. each exact inherited M10.2 protocol/stage file referenced there.

Do **not** resume the M10.2 4.944692h actor. Start Stage 1 from the exact v10.1 +10h actor `e6b9fd1a38dbb5c6670711a8b47769a628d2f20caf7c88738f372d0839b7a3b6` with manifest `d1a785ef439b0127b5ab1a9ff1693ade1aa11d850151cd17b9733bbeb98dacb3`. Transfer actor weights only; create fresh critic and fresh actor/critic optimizers.

Do not change RivalPolicyV1, RivalObsV1, RivalActionV1, RivalCanonicalStateV1, native 120 Hz control, no-repeat action, or one-tick delay semantics.

Implement Stage-1 V2 exactly: anti-idle reward, repaired ordinary-family headings, fresh V2 frozen/unseen corpora, source baseline on those corpora before PPO, idle telemetry, and reward truth-table tests. Reuse the fixed M10.2 boundary batching behavior and retain regression coverage for a positive final rollout smaller than one ordinary minibatch.

Run the same Stage-1 ladder:
`+1, +2.5, +5, +7.5, +10, +12.5, +15 learner-simulated hours`

Use the inherited M10.2 mastery thresholds and stop rules against the V2 corpus/baseline. If the +5h no-material-learning condition is met, stop the entire campaign. If and only if Stage 1 passes, continue Stage 2, Stage 3, then Stage 4 under the exact inherited M10.2 documents. A failed prerequisite stops progression even if wall time remains.

Use the same 10-hour progressive wall-clock envelope and finalization reserve.

Write evidence under `training/results/milestone10_3/` and human closeout at `docs/MILESTONE_10_3_RESULTS.md`. Push stable commits at implementation/preflight and each evaluation boundary. Preserve clean recoverable checkpoints locally/Git-ignored and verify independent reload where required.

Do not modify or promote production. Frozen Wisp hashes must remain:
- POLICY.lt `1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7`
- SHARED_HEAD.lt `3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42`

If a gate fails, stop honestly at that gate; do not silently continue past an authority stop. At completion verify tests, Ruff, compileall, JSON parsing, ignored checkpoint binaries, worker cleanup, clean worktree, preserved stash, and remote `origin/main` readback. Report the final remote SHA and exact campaign decision.
