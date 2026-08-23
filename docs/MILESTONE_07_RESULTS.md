# Milestone 07 Results

Milestone 07 isolated the Milestone 06 RocketSim-to-RLBot failure without resuming
serious PPO training. The diagnosis is mixed: the four-tick cadence breaks the
zero-step Wisp-compatible policy before learning, the 20M actor substantially drifted
inside legacy actions, and the training observation approximation materially changes
frozen-Wisp decisions on live states. Spatial action semantics are exact. Short-horizon
physics replay is close for one to two four-tick decisions but develops material car
orientation divergence by 16 physics ticks.

Production Rival remains the frozen Wisp policy at tick skip 8. No checkpoint was
promoted, no optimizer/PPO step was taken, and Stage C was not started.

## Boundary and provenance

- Starting authority commit: `2622a63145fca4ca719a4e2950e9ac5e1e3c3afd`.
- Canonical M06 rollback commit remains an ancestor:
  `652395a9f512ce835830bfc5bc3a7cb078f6105e`.
- Frozen `POLICY.lt` SHA-256:
  `1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7`.
- Frozen `SHARED_HEAD.lt` SHA-256:
  `3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42`.
- Rejected M06 20M actor SHA-256:
  `fcb557bbc19c945dba05c986c04cba484d75cafa283f77341addba7e1dafe889`.
- Zero-step M07 deployment actor SHA-256:
  `4480696d89a3c88770807212dfbb5bc3cd4fb4492511e67a5b7dd1213c9aa3a7`.
- Exact 158-row action-table file SHA-256:
  `bed450ded25a7bc624b17695913626fd478d00e11936413c7f5af5022585de2f`.

The zero-step actor came from the completed direct trainable reconstruction, contains
158 outputs, has exact frozen-Wisp weights/logits for indices 0 through 89, has zero
appended weights and appended biases of exactly `-12`, and records zero optimizer and
PPO updates. The ignored actor/checkpoint binaries and raw telemetry remain local;
their paths, sizes, and hashes are recorded in the compact committed reports.

## Diagnostic-only implementation

The RLBot candidate seam now accepts explicit Milestone 07 controls:

- `RIVAL_TRANSFER_DIAGNOSTIC_MODE=1` gates all transfer-only flexibility;
- `RIVAL_TICK_SKIP=4|8` selects candidate cadence only inside that mode;
- `RIVAL_CANDIDATE_LEGACY_ONLY=1` keeps the exact first-90 Wisp mask and hard-masks
  indices 90 through 157;
- `RIVAL_CANDIDATE_RUNTIME_LABEL` gives diagnostic sessions an unambiguous label;
- `RIVAL_DIAGNOSTIC_CAPTURE_OBSERVATIONS=1` and a positive
  `RIVAL_DIAGNOSTIC_OBSERVATION_STRIDE` add sampled exact 432 tensors to enabled
  telemetry.

These switches are off by default. Without a candidate, production still requires
tick skip 8 and loads the frozen Wisp artifacts. Ordinary candidate deployment still
requires tick skip 4. Observation capture additionally requires explicit telemetry,
so it cannot silently add production overhead.

## Zero-step export and same-observation parity

The export gate passed before any student match testing:

- 1,024 seeded random 432 tensors: first-90 logits bit-exact, zero maximum/mean
  error, and 100% top-1 agreement;
- production and training Python runtimes both loaded the same export and emitted a
  finite `[64, 158]` tensor;
- 1,276 sampled exact live P0 observations: first-90 logits remained bit-exact,
  masked top-1 agreement was 100%, KL and JS divergence were zero, and recomputed
  frozen actions agreed with the actions logged during play on every sample.

This rules out reconstruction, TorchScript export, model loading, and the first-90
mask as explanations for the zero-step match differences.

On those same 1,276 observations, the 20M actor versus frozen Wisp measured:

| Metric | Result |
| --- | ---: |
| Masked legacy top-1 agreement | 68.0251% |
| Disagreements | 408 / 1,276 |
| Mean absolute first-90 logit drift | 0.437580 |
| Maximum absolute first-90 logit drift | 12.617197 |
| Mean JS divergence | 0.015496 |
| Frozen / trained mean confidence | 0.158166 / 0.171906 |
| Frozen / trained mean top-1/top-2 margin | 0.066097 / 0.073293 |

The drift is state-dependent. Agreement was 62.09% for close-ball samples versus
90.05% for far-ball samples, 64.47% in ordinary active play versus 94.12% on the
kickoff-flag subset, and 61.17% at high boost versus 74.29% when boost was empty.
The report includes compact action-frequency, cross-policy action mapping, and
stride-sampled transition counts.

## RLBot v5 transfer matrix

All required cells used full five-minute Soccar, game speed 5, normal kickoff
countdowns, installed Nexto and Wisp v2-75B, and one blue plus one orange game against
each opponent. No affected mode was expanded because the controlled direction was
already coherent at the authorized four-game boundary.

| Mode | Policy and cadence | Record | Goals | Goal differential | Runtime clean |
| --- | --- | ---: | ---: | ---: | ---: |
| P0 | frozen Wisp, tick 8 | 1-3 | 11-13 | -2 | 4 / 4 |
| Z8 | zero-step, legacy-only, tick 8 | 2-2 | 22-18 | +4 | 3 / 4 |
| Z4 | zero-step, legacy-only, tick 4 | 0-4 | 13-24 | -11 | 4 / 4 |
| T8 | 20M actor, legacy-only, tick 8 | 1-3 | 15-20 | -5 | 4 / 4 |
| T4 | 20M actor, legacy-only, tick 4 | 1-3 | 18-25 | -7 | 4 / 4 |

Opponent splits were:

| Mode | Nexto | Wisp v2-75B |
| --- | --- | --- |
| P0 | 1-1, 7-7 | 0-2, 4-6 |
| Z8 | 2-0, 14-8 | 0-2, 8-10 |
| Z4 | 0-2, 9-14 | 0-2, 4-10 |
| T8 | 0-2, 6-14 | 1-1, 9-6 |
| T4 | 1-1, 9-12 | 0-2, 9-13 |

The four-game cells are causal diagnostics, not precise skill estimates. The direct
controlled effects are nevertheless large enough to answer the v7 questions:

1. **Does zero-step Wisp preserve RLBot gameplay at tick 8?** Yes, reasonably at
   this boundary. Z8 preserved exact logits and finished 2-2 / +4 versus P0's 1-3 /
   -2. It did not reproduce every score, and one completed Z8 game is not
   runtime-clean, but there is no zero-step tick-8 collapse.
2. **What happens at tick 4?** The identical zero-step actor fell from Z8's 2-2 / +4
   to Z4's 0-4 / -11: two fewer wins and a 15-goal differential swing. This is the
   strongest controlled effect and exists before learning.
3. **How much did the 20M actor drift?** It changed the masked legacy top-1 on 408 of
   1,276 live states and had mean/max first-90 logit drift of 0.437580/12.617197.
4. **Is the 20M actor healthier at tick 8?** Only marginally by this sample. T8 and
   T4 were both 1-3; T8 was -5 and T4 was -7. This does not justify claiming a
   material tick-8 recovery. Relative to Z8, T8 lost one win and nine goal-difference
   points, showing harmful learned drift at the healthy cadence. Relative to Z4, T4
   gained one win and four goal-difference points, showing that tick-4 training partly
   adapted to, but did not repair, the pre-existing cadence collapse.

Across the 20 included sessions, 20/20 telemetry invariant checks passed, no appended
action was selected, and every first-90 mask/selection was valid. Nineteen sessions
were runtime-clean. One Z8 overtime match produced the same one-packet non-finite ETA
exception in Rival and the installed Wisp opponent; both recovered and the match
completed. Its score is retained for behavioral context but runtime health is marked
unclean. Three incomplete/invalid setup sessions are explicitly listed and excluded.
After the shared application-state anomaly, each remaining match used a freshly
verified `RocketLeague.exe` process.

## Observation-domain audit

The audit used the same 1,276 natural P0 decision states. Each exact live 432 tensor
was paired with the closest `WispCompatible432RLGymV1` reconstruction from the same
RLBot packet. The conversion does not claim an exact RocketSim state: packet
kinematics/boost were mapped without a settling step, RLBot jump flags were mapped to
the closest RLGym flags, the one opponent block was aligned to the same randomized
slot, and the training builder intentionally recomputed RocketSim ball prediction,
bounded ETA, and landing normal.

Whole-vector error looks deceptively small because many fields are exact: mean
absolute error 0.026867 and p95 0.001222. Frozen Wisp, however, agreed on only
596/1,276 masked actions (46.7085%) when given the training-style representation;
680 actions changed. Mean/max first-90 logit error was 0.812305/20.058516, mean
confidence fell from 0.158166 to 0.139571, and mean margin fell from 0.066097 to
0.046483.

The largest raw differences and policy-sensitive substitutions were:

| Feature source | Raw mismatch | Training-to-live substitution effect |
| --- | --- | --- |
| Self ETA | MAE 4.5144 s; p95 9.6903 s; correlation 0.1930 | Changed 34.40% of training-style top-1 actions; recovered 174 baseline disagreements |
| Opponent ETA slots | MAE 1.5416 s; p95 9.0306 s; correlation 0.6341 | Changed 34.80%; recovered 139 disagreements |
| Self car block | MAE 0.0977 across the 51-value block; max 9.9872 | Changed 34.72%; recovered 180 disagreements |
| Opponent blocks | MAE 0.0340 across padded blocks; max 9.9818 | Changed 35.66%; recovered 146 disagreements |
| Touch/handbrake | MAE 0.5089 for the targeted fields; correlation 0.0725 | Changed 23.75%; recovered 112 disagreements |
| Self turn/touch/handbrake group | MAE 0.3693; p95 1.0 | Changed 26.49%; recovered 126 disagreements |
| Ball-prediction horizons | MAE 0.002243; max 0.8490 | Changed 4.15%; recovered 17 disagreements |
| Landing normal | MAE 0.0621; p95 0.5093 | Changed 2.12%; recovered 3 disagreements |

Ball kinematics, goal vectors, wall distances, close-pad positions, kickoff flag,
previous action, score differential, and teammate-padding blocks were exact or
numerically negligible on this corpus and did not change top-1. Boost-pad timer
outliers existed but changed only 0.31% of training-style top-1 actions. The main
observation seam is therefore cached/bounded ETA plus live analog touch/handbrake/car
state, not basic coordinate conversion.

## Action-function parity

Spatial/controller semantics passed exactly:

- all 90 legacy rows were compared across production Wisp, the 158-action
  legacy-only deployment parser, and the RLGym parser;
- blue/orange and both X-mirroring regimes produced 360/360 exact row-case
  comparisons with zero maximum controller error;
- diagnostic legacy-only deployment rejected index 90 as well as masking the entire
  appended suffix;
- the generated and committed 158-row tables were exact.

Temporal semantics must be treated separately. Source-derived steady-state windows
showed:

- tick 4 / delay 3: live and `RocketSimEngine(rlbot_delay=True)` both apply
  `[previous, new, new, new]` -- exact;
- tick 8 / delay 7: live Wisp applies
  `[previous, previous, previous, previous, previous, new, new, new]`, while the
  generic RocketSim delay applies `[previous, new, new, new, new, new, new, new]`.

Thus there is no hidden spatial parser defect, and M06's four-tick training/live
action window is aligned. The zero-step tick-4 failure is a Wisp policy/cadence domain
failure, not a parser mismatch. The optional legacy8 RocketSim environment does not,
however, reproduce production Wisp's temporal window and must be corrected before it
is used for future legacy-cadence training.

## Short-horizon transition audit

Thirty-two natural T4 windows were selected evenly across grounded/airborne and
contact/free categories. RocketSim was initialized from exact observable packet
kinematics/boost plus closest jump/ground flags, with an Octane hitbox. Rival's exact
four-tick applied sequence was replayed; the opponent's packet `last_input` was held
for each observed four-tick segment because opponent outputs were not logged.

Contact-free p95 results were:

| Physics ticks | Windows | Self position | Self velocity | Self orientation | Ball position | Ball velocity |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 30 | 1.03 uu | 49.30 uu/s | 4.30 deg | 0.018 uu | 0.042 uu/s |
| 8 | 28 | 3.18 uu | 88.54 uu/s | 7.59 deg | 0.034 uu | 0.085 uu/s |
| 16 | 26 | 12.12 uu | 182.74 uu/s | 15.45 deg | 0.053 uu | 0.175 uu/s |
| 32 | 22 | 29.24 uu | 215.45 uu/s | 27.56 deg | 0.085 uu | 0.348 uu/s |
| 64 | 16 | 144.30 uu | 468.28 uu/s | 39.08 deg | 0.405 uu | 3.19 uu/s |

Touch occurrence agreed in every selected comparison. Self ground agreement was
93.33% at 4 ticks, 92.86% at 8, and 100% at 16/32 in the shrinking contact-free set;
flip/jump agreement was 100% through 32 ticks. The all-window report includes contact
cases, where ball divergence is naturally larger.

There is measurable transition divergence large enough to matter: the bounded rule
trips at 16 ticks because contact-free self-orientation p95 exceeds 10 degrees. It is
not the earliest or best-supported dominant cause: at one or two four-tick decisions,
position and ball evolution remain tight, and the reconstruction/opponent-control
limitations can inflate car errors. M08 should improve transition matching and
contact robustness, but fixing physics alone would not address the much larger
controlled cadence and observation effects.

## Ranked causal diagnosis

1. **Four-tick strategic-policy cadence mismatch -- primary controlled cause.** The
   exact zero-step policy moved from 2-2 / +4 at tick 8 to 0-4 / -11 at tick 4. The
   Wisp policy was not behaviorally invariant to being queried twice as often with a
   different repeat/delay history.
2. **Training/live observation mismatch -- major interacting cause.** Reconstructing
   training-style observations on live states changed 53.29% of frozen-Wisp actions.
   ETA and touch/handbrake/car-block fields dominate sensitivity. PPO optimized under
   this observation distribution, then the actor was deployed on live Wisp tensors.
3. **Learned legacy-policy drift -- major interacting cause.** The 20M actor changed
   31.97% of live legacy top-1 decisions. At tick 8 it degraded Z8 by one win and nine
   goal-difference points. Tick-4 training partially improved Z4 but still did not
   preserve the production baseline.
4. **Transition/physics mismatch -- secondary measured contributor.** Natural
   contact-free orientation divergence becomes material by 16 ticks and grows with
   horizon, but it is much smaller over the first 4-8 ticks and cannot explain exact
   zero-step policy/cadence or same-state observation findings by itself.
5. **Export, mask, and spatial action mapping -- ruled out at measured precision.**
   Zero-step logits were exact, all appended actions were hard-masked and never
   selected, and all 360 controller/mirroring comparisons were exact.

The interaction matters. Four-tick cadence creates a failure before learning;
four-tick PPO partly adapts to it while simultaneously moving the live legacy policy
away from the healthy tick-8 control. Observation mismatch gives PPO a materially
different decision surface from deployment, and physics divergence adds longer-horizon
noise. No single one of these measurements justifies resuming the unchanged M06
campaign.

## Milestone 08 correction

Milestone 08 should implement a dual-rate, transfer-gated design rather than continue
the current monolithic four-tick warm start:

1. Keep the Wisp-compatible strategic/legacy branch at its native eight-tick live
   cadence. Initially freeze its trunk and first-90 head, or enforce exact teacher
   logits/top-1 with a strong constraint. Hold its base action using production's
   actual eight-tick temporal sequence.
2. Add a separately trained four-tick mechanics/recovery branch. It may select
   appended actions or a bounded residual only under an explicit gate; with the gate
   disabled, behavior must reduce to the verified Z8 control. Do not make every
   four-tick decision overwrite the strategic legacy action.
3. Make one versioned observation contract shared by training and deployment. Port or
   replace cached ETA, touch/handbrake, landing, and state-flag semantics so both
   engines compute the same defined fields. Run feature-group and Wisp-logit parity on
   a held live corpus before PPO. Ball-prediction differences can be addressed after
   the dominant ETA/analog fields.
4. Model both temporal action functions explicitly. Preserve the exact tick-4 window
   for the mechanics branch and implement production's five-previous/three-new
   tick-8 window for the strategic branch; do not use the current generic legacy8
   `rlbot_delay=True` behavior as an equivalence claim.
5. Stage learning: train the new head/critic first, then permit bounded shared-trunk
   updates only with frozen-teacher KL/logit/top-1 regularization on first-90 actions,
   historical-opponent diversity, and separate reporting of legacy versus appended
   behavior.
6. Gate budget prospectively. Before a large run, require: exact zero-step export,
   observation/temporal parity, a disabled-mechanics Z8 RLBot control that reasonably
   preserves P0, then small-step RLBot matrices showing no baseline regression at
   each clean boundary. A headless RocketSim improvement alone cannot authorize
   continuation or promotion.

If M08 instead chooses a monolithic four-tick actor, it must first distill the frozen
Wisp to four ticks under live-equivalent observation and temporal semantics and prove
that zero/small-step RLBot control. The present evidence does not support simply
unfreezing the M06 student and continuing PPO.

## Optional RLViser spectator

Milestone 07 also adds `RivalRLViserSpectatorV1`, an opt-in, separate process with one
independent RocketSim environment and `RLViserRenderer`. It is disabled by default,
uses CPU inference with one Torch thread and below-normal process priority, is paced
at approximately real time from the 120 Hz physics rate, and is never imported by
rollout workers or the RLBot diagnostics. It does not render any training worker or
mutate checkpoint/training state.

Install the pinned optional bridge/viewer and verify it without opening a window:

```powershell
./training/install_rlviser_spectator.ps1

training/.venv/Scripts/python.exe training/scripts/run_m07_rlviser_spectator.py `
  --checkpoint current --check
```

Watch the latest local campaign checkpoint against frozen Wisp:

```powershell
training/.venv/Scripts/python.exe training/scripts/run_m07_rlviser_spectator.py `
  --checkpoint current --opponent frozen-wisp --tick-skip 4
```

`--checkpoint frozen-wisp` selects production Wisp; a campaign directory,
`PPO_POLICY.pt`, portable actor checkpoint, or TorchScript actor may also be supplied.
`--opponent selected` enables rendered self-play, `--legacy-only` masks the appended
suffix, `--playback-speed` changes pacing, and Ctrl+C stops an unbounded session.

The actual smoke launched pinned RLViser v0.8.2, positively observed the expected
`rlviser.exe` process, rendered 58 decisions over 2.007 seconds, and shut down without
a lingering viewer. `rlviser-py==0.6.13` is isolated as an optional no-dependencies
install because 0.6.14 requires NumPy 2 while RLGym Rocket League 2.0.1 requires NumPy
below 2. The executable is downloaded only after an exact SHA-256 gate and is ignored
by Git.

## Verification

The final local verification boundary passed:

- production suite: 81 tests passed;
- complete training suite: 30 tests passed;
- direct production/training action-prefix proof: 90/90 rows exact, zero error;
- RLGym/RocketSim environment and both cadence smokes: passed on Python 3.12.13,
  RLGym Rocket League 2.0.1, RocketSim 2.2.1, Torch 2.13.0+cu130, and the RTX 5090;
- frozen-teacher/direct-bootstrap proof: 4,096 CUDA tensors exact and teacher hashes
  unchanged;
- 5,000-decision / 10,000-agent-step RocketSim stress: finite, 5,000 decisions
  completed in 4.257 seconds;
- Ruff on every changed production file plus the complete training subtree: passed;
- Python compile checks for changed production code, tests, and training code: passed;
- optional installer PowerShell parser check and RLViser dependency/policy preflight:
  passed;
- all required M07 JSON parsed, report statuses/invariants passed, and manifest hashes
  matched recomputed bytes;
- `git diff --check`: passed.

A repository-wide Ruff diagnostic also found four pre-existing issues in untouched
legacy `bot/backend/gamestate/rot_mat.py` and `bot/eta.py`. They are outside the M07
change and were not silently mixed into this transfer-diagnostic commit. The changed
production surface and full training subtree are clean.

## Evidence index

- `training/results/milestone07/zero_step_export.json`
- `training/results/milestone07/zero_step_runtime_production.json`
- `training/results/milestone07/zero_step_runtime_training.json`
- `training/results/milestone07/live_policy_parity.json`
- `training/results/milestone07/transfer_matrix.json`
- `training/results/milestone07/transfer_telemetry.json`
- `training/results/milestone07/observation_domain.json`
- `training/results/milestone07/action_parity.json`
- `training/results/milestone07/transition_audit.json`
- `training/results/milestone07/rlviser_spectator_smoke.json`
- `training/results/milestone07/evidence_manifest.json`

The manifest records committed report hashes and the exact local source-artifact
hashes. All machine-readable reports set production promotion/modification false where
applicable. Final test commands and remote readback are reported separately because a
commit cannot contain a hash/readback of itself.
