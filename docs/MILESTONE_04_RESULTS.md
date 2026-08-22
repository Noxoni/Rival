# Rival Milestone 04 results

Milestone 04 is technically complete and the tested gameplay adjustment is **rejected**. Rival ran
16 full natural RLBot v5 matches at approximately 5x effective simulation speed, produced 86,950
policy decisions, and compared an eight-match baseline with an eight-match treatment against the
same balanced Nexto/Wisp and blue/orange mix. The one treatment changed Wisp's selected action 17
times, but the targeted outcome became worse. Normal Rival therefore keeps both
`RIVAL_NATURAL_ADJUSTMENT_MODE=off` and `RIVAL_CHALLENGE_CALIBRATION_MODE=off`.

Scores and aggregate telemetry are useful engineering context, not a skill-ranking claim. The
natural trajectories are deliberately unrelated and unpaired.

## Authority and repository boundary

This run followed `handoff/v4.1/CODEX_START_PROMPT.md`. The v4.1 authority superseded the
unexecuted deterministic-scenario direction in `handoff/v4.0/` before execution. The paused v4.0
strategy work remains byte-preserved in the local named stash
`rival-v4-paused-superseded-before-v4.1`; it was not applied, dropped, or mixed into v4.1. Ignored
raw telemetry, `.resume_backup/`, `.pytest_tmp/`, and unrelated user files remained outside the
committed source tree.

The v5.0 handoff documents arrived on `origin/main` while this milestone was active. Their package
manifest explicitly marks v5.0 as queued after a coherent v4.1 boundary. Those documentation-only
commits were preserved without overriding the running experiment; v5.0 was not executed here.

## Stable implementation commits

| Scope | Commit |
| --- | --- |
| Accelerated natural runner | `529217f29c915684055c1727101ec2d1b312266a` |
| Sustained in-play speed validation | `ec2326ab66f3b8417be05833c51d1106f3d40556` |
| Baseline aggregation and evidence | `3c15ff55ba6005777c3ab6457dc3d14e8453a966` |
| Low-resource aerial treatment | `463e74052b80500e348216a3bdca5676770f2f60` |

The result/evidence commit and final remote readback SHA are reported with the completed handoff
after the final push. This avoids putting a self-referential commit id inside the commit itself.

## Natural runner and throughput

Every match used `Stadium_P`, full five-minute Soccar, normal physics/boost/gravity/demolition and
scoring, normal kickoff countdowns, skipped goal replays, disabled replay auto-save, rendering
`AlwaysOff`, performance monitor `NeverShow`, agent readiness waiting, and clean sequential match
restart. The only natural state setting was `DesiredMatchInfo.game_speed=5.0`.

The runner validates acceleration from sustained simulated-game-time progression rather than the
stale packet `game_speed=1.0` echo. Reset, goal, and kickoff gaps are excluded from this measurement.

| Phase | Matches | Effective speed min / median / max | Decisions per simulated second min / median / max | Policy decisions | Raw bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 8 | 4.9995 / 4.9999 / 5.0005 | 13.6263 / 13.9459 / 14.2085 | 43,792 | 598,494,271 |
| Treatment | 8 | 4.9863 / 4.9998 / 5.0002 | 13.6654 / 13.9544 / 14.2433 | 43,158 | 635,593,247 |
| Combined | 16 | 4.9863 / approximately 4.9998 / 5.0005 | 13.6263 / approximately 13.95 / 14.2433 | 86,950 | 1,234,087,518 |

All 16 matches passed completion, bot responsiveness, telemetry validity, decision cadence,
effective-speed, and non-degenerate-action health gates. Runs were sequential; no further time was
spent on multi-instance launcher engineering.

## Baseline batch and recurring patterns

The baseline treatment mode was `off`: four matches each against installed Nexto and Wisp v2-75B,
with Rival blue four times and orange four times. It finished 4–4, 35–30 goals (+5), while
collecting 43,792 policy decisions. The analyzer collapsed same-pattern anchors within two
simulated seconds and ranked recurring observational candidates by cross-match frequency plus
opponent-next-touch and near-term-concession consequence.

| Baseline pattern | Independent episodes | Per match | Opponent next touch | Conceded next goal within 10 s |
| --- | ---: | ---: | ---: | ---: |
| `apparent_pressure_release_after_closing_abort` | 510 | 63.75 | 173 / 510 (33.9%) | 42 / 510 (8.2%) |
| `low_resource_aerial_commitment` | 104 | 13.00 | 38 / 104 (36.5%) | 13 / 104 (12.5%) |
| `boost_pickup_with_eta_possession_flip` | 66 | 8.25 | 15 / 66 (22.7%) | 6 / 66 (9.1%) |

The pressure-release candidate was most frequent, but its broad detector is inference-only and
overlaps the rejected Milestone 03 challenge direction. The low-resource aerial pattern occurred
in all eight matches, had the highest near-term concession rate, and offered a narrow correction
boundary. It was therefore selected prospectively before the treatment batch.

Detector outputs remain candidate screens, not ground-truth behavior labels. No synthetic scenario
was used to create or inflate any exposure.

## The one state-conditioned treatment

Parameter version `m04p1-low-resource-aerial-v1` applies a graded logit penalty over Wisp's existing
legal actions only. It does not change model weights, create controller values, or see a scenario
identity. It acts only during a new aerial-like baseline transition when all of the following live
conditions hold:

- match phase is Active and defensive-emergency safety is false;
- self boost is below 30;
- ball height is at least 300 uu and self-to-ball distance is at least 650 uu;
- possession ETA advantage is at most 0 seconds;
- Wisp's baseline choice is resource-committing;
- the previous baseline decision was not already aerial-like.

The severity uses boost deficit (0.45 weight), ETA disadvantage (0.30), ball height (0.15), and
ball distance (0.10), clamped to a 0.15–0.55 logit penalty. Grounded states penalize jump actions;
airborne states penalize boost actions. Telemetry records the baseline action, hypothetical action,
final action, severity, eligibility, gate failures, and reason. The exact baseline-off path remains
available and is the default.

## Natural baseline versus treatment

The treatment repeated the same mix: four Nexto, four Wisp v2-75B, four blue, four orange. It
finished 4–4, 33–33 goals (0), with 43,158 policy decisions. The frozen treatment was not tuned or
given compensating rules after results were visible.

| Aggregate metric | Baseline | Treatment | Treatment minus baseline |
| --- | ---: | ---: | ---: |
| Wins / losses | 4 / 4 | 4 / 4 | 0 wins |
| Goals for / against | 35 / 30 | 33 / 33 | goal differential −5 |
| Favorable ETA share | 64.07% | 62.43% | −1.65 percentage points |
| Rival share of recorded self/opponent touches | 75.74% | 71.33% | −4.40 percentage points |
| Possession-loss transitions | 367 | 365 | −2 |
| Conceded within 10 s per possession loss | 41 / 367 (11.17%) | 36 / 365 (9.86%) | −1.31 percentage points |
| Low-resource aerial episodes | 104 (13.0/match) | 96 (12.0/match) | −8 (−1.0/match) |
| Low-resource aerial opponent-next-touch rate | 36.54% | 47.92% | **+11.38 percentage points** |
| Low-resource aerial near-term concession rate | 12.50% | 12.50% | 0 |
| Low-resource aerial near-term scored rate | 9.62% | 5.21% | −4.41 percentage points |

The adjustment was eligible on 33 decisions and actually changed Wisp's action on 17 decisions
(0.0394% of treatment decisions). Following those 17 changed actions, the next touch was Rival 1,
opponent 15, and none 1. Within 10 simulated seconds the next goal was Rival 1, opponent 2, and
none 14.

The lower aggregate concession rate after possession loss is directionally favorable, but it does
not outweigh the directly targeted regression, 15/17 opponent next touches after interventions,
lower favorable-ETA/touch shares, and worse score context. Because trajectories are unpaired,
these deltas are not a causal effect estimate; they are sufficient directional evidence to reject,
not sufficient evidence to claim the baseline is universally stronger by an exact amount.

## Acceptance decision

The treatment is **rejected** and remains **off by default**. Its selected behavior frequency fell
slightly, but the selected behavior's opponent-next-touch rate materially worsened, near-term
concessions did not improve, and broader possession proxies regressed. No post-freeze tuning or
second gameplay fix was attempted. The code and telemetry path remain switchable for reproducible
inspection; normal gameplay retains the frozen Wisp baseline.

Milestone 03 challenge calibration also remains off. The run did not reuse its rejected threshold
sets or scenario labels.

## Runtime observations and limitations

- Native RLBotServer startup queues reported 1, 10, and 100 missed messages during startup
  transitions. Those messages are on native stdout, outside Python `runtime_warnings`; every bot
  became responsive and every session completed.
- Baseline match 6, a Wisp overtime, produced one recovered invalid-value/NaN packet-processing
  exception in both Rival and Wisp. Both managers resumed and completed overtime. The session met
  the frozen health gates and is retained with this caveat; it is not represented as error-free.
- Treatment sessions recorded zero session errors, zero Python runtime-warning entries, and zero
  invalid JSON lines. This does not erase the separate native startup warning caveat.
- All raw JSONL remains local and Git-ignored. Compact manifests retain each session id, byte count,
  SHA-256, health result, schedule, score, and runtime metadata. Both analyses independently loaded
  those raw files and verified every batch-manifest hash.
- Text-artifact SHA-256 values in the decision record are hashes of committed LF-normalized Git
  blobs, avoiding Windows working-copy CRLF ambiguity.

## Evidence artifacts

- `evidence/results/v4.1/natural_baseline_batch.json`
- `evidence/results/v4.1/natural_baseline_analysis.json`
- `evidence/results/v4.1/natural_baseline_analysis.md`
- `evidence/results/v4.1/baseline_runtime_observations.json`
- `evidence/results/v4.1/treatment_definition.json`
- `evidence/results/v4.1/natural_treatment_batch.json`
- `evidence/results/v4.1/natural_treatment_analysis.json`
- `evidence/results/v4.1/natural_treatment_analysis.md`
- `evidence/results/v4.1/treatment_runtime_observations.json`
- `evidence/results/v4.1/milestone_04_decision.json`

`milestone_04_decision.json` is the compact machine-readable comparison and acceptance record.
The batch manifests are the committed ledgers for the ignored raw telemetry.

## Verification

Final verification passed:

- complete pytest suite with a fresh task-specific basetemp: **70 passed**, with only two
  known `torch.jit.load` deprecation warnings;
- focused natural-adjustment, runner, analyzer, and telemetry tests: **20 passed**;
- Ruff on all 14 Python files authored or modified for v4.1: passed;
- compileall over `bot`, `tools`, `scripts`, and `tests`: passed;
- frozen-policy comparison against `4f2b21c00e2fcb7108ab1006fd950b066fbd0484`: passed;
- model smoke: finite 90-value policy output from a 432-value observation, with `POLICY.lt`
  `1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7` and `SHARED_HEAD.lt`
  `3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42`;
- installed Nexto and Wisp v2-75B files: every recorded size/hash matched the read-only reference
  manifest;
- default-off and correctly versioned treatment-mode decision smokes: passed, each with one valid
  schema-v3 decision plus start/end records;
- all eight v4.1 JSON artifacts: parsed successfully;
- independent baseline and treatment re-analysis: exactly reproduced all committed metrics after
  excluding only generation timestamps and re-verified all 16 raw telemetry hashes;
- `git diff --check`: passed.

The release step additionally reviews staged Git blobs for path/credential leakage and verifies
the exact `origin/main` commit by remote readback after push. The exact final SHA is reported in the
external completion handoff so the result commit does not need to identify itself recursively.

## Next natural-play target

`apparent_pressure_release_after_closing_abort` remained the highest-frequency candidate in both
batches and rose to a 41.8% opponent-next-touch rate in treatment. Before another gameplay change,
narrow that inference-only detector with observable opponent closing-trend and Rival action-history
attribution. If it remains frequent and consequential, evaluate one natural state-conditioned
response without scripted scenario labels or the rejected Milestone 03 thresholds.
