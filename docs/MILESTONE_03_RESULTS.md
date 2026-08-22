# Rival Milestone 03 results

Milestone 03 is technically complete and **rejected as a gameplay change**. The challenge-
commitment estimator, conservative legal-action re-ranker, schema-v3 telemetry, offline analyzer,
fixtures, and reproducible RLBot v5 runners are preserved. The normal bot remains
`noxoni/rival/dev-v1` with `RIVAL_CHALLENGE_CALIBRATION_MODE=off` because neither controlled
treatment attempt applied an intervention or established causal fake-pressure improvement.

The controlled gate therefore never opened. Milestone 03 ran **zero natural acceptance matches**
and consumed **zero of the six-match natural budget**. No result below is a skill-ranking claim.

## Resume and synchronization evidence

The v3.3 resume procedure ran before repository synchronization.

| Item | Result |
| --- | --- |
| Pre-resume local `HEAD` | `5cd05a9cb1e88df65d5ad417ca6f1e8242356be7` |
| Pre-fetch local view of `origin/main` | `5cd05a9cb1e88df65d5ad417ca6f1e8242356be7` |
| Fetched `origin/main` | `23ca70e23f6518b8f66251319c3b113aed4d0b3c` |
| Synchronization | named stash, fast-forward pull, stash pop, byte-exact restore from raw backup |
| Named stash | `rival-m03-paused-before-v3.3-resume` (dropped by successful pop) |
| Local ignored backup | `.resume_backup/m03-v3.3-20260822-150302/` |
| First coherent implementation commit | `f025344b5ab325c3b6bfb082770b95a81e6f9809` |

The paused files were captured independently before the fetch/pull:

| Paused file | Pre-sync Git blob | Raw backup SHA-256 |
| --- | --- | --- |
| `bot/strategy/__init__.py` | `1cbfe84ea48f014c8a172ad8b2308c2395bc3219` | `fe79e988577b822383f396d0dca9417f6eef081fe893498b09b12fea7adfe63a` |
| `bot/strategy/challenge_commitment.py` | `89d829c5fc1df355973b8ea1bfdd42faf510c985` | `36c33b6c6ccbbe3152ed50dde1641aa3b86fbd944912f351030396d439b18177` |

Stash pop converted line endings, so the raw backup was copied back before development continued.
Both files then matched their pre-sync blob and byte hashes. The implementation subsequently
expanded the restored commitment file intentionally; the untouched pre-sync bytes remain in the
ignored backup. The unrelated user archive `bot.7z` was absent from status at resume and was not
read, moved, changed, committed, or deleted.

Every file under v3.3, v3.2, v3.1, and v3.0 was read in the required order. The v3.3 and v3.0
package manifests were checked against committed Git blobs before implementation.

## Implemented experiment boundary

The implementation follows the v3.0 gameplay boundary:

- `off` bypasses the estimator/action resolver and returns the exact masked-Wisp argmax path;
- `observe` computes and logs a hypothetical continuation but returns the baseline action;
- `intervene` can select only an existing legal Wisp discrete action;
- no controller values are synthesized;
- no observation, legal mask, model, policy cadence, action delay, or frozen model artifact changed;
- the treatment requires ground control, apparent pressure, non-high commitment, a grounded
  baseline jump initiation, safety clearance, and a conservative Wisp-preference gap;
- deferral is hard-bounded to one policy decision tick for the tested configurations;
- kickoff, goal/reset, time rewind, demolition/reset, and discontinuity handling reset estimator
  state;
- schema-v3 telemetry records baseline, hypothetical, and final actions together with commitment
  score/state, history/trends, gate failures, safety exclusions, continuation candidate, preference
  gaps, and deferral budget;
- full logits remain disabled by default.

The normal config still defaults to `off`; the RLBot display name remains `Rival Dev`, and the
agent id remains `noxoni/rival/dev-v1`. The rejected experiment was not rebranded as v3.

## Frozen policy and reference identity

| Artifact/reference | SHA-256 |
| --- | --- |
| `POLICY.lt` | `1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7` |
| `SHARED_HEAD.lt` | `3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42` |
| Installed Nexto config | `0e4818a04e2364a6b8c84b38be24db8f6856f7377f7cf3b802fdfac4b526828f` |
| Installed Nexto executable | `6371a5b9dd740aea858d86532f928e0fdb13bf387d8379f76ccd679c9b33e845` |
| Installed Wisp v2-75B config | `75a60b870759f855c4d900561553ae7535c379eed4df20dc63a0969bcac18825` |
| Installed Wisp v2-75B executable | `b9dbe32bdae28c299daffcc0673f5a438d8013a553be3938dafaa112884e7184` |

The installed BotPack remained read-only. Runtime versions in the live manifests were Python
3.12.13, RLBot 2.0.0b54, RocketSim 2.2.1, and Torch 2.13.0.

## Offline evidence and fixtures

The v3 detector narrows Milestone 02's broad `boost|pitch|jump` release-like ranking signal to:

> grounded final jump/dodge initiation while Rival is within the ground-control
> distance/height/ETA gate, apparent pressure is present, commitment is below the high threshold,
> and the opponent is outside the unavoidable-intercept boundary.

Re-analysis of the original controlled corpus found only **1/20** fake/non-immediate cases under
this refined metric, not the broad detector's 20/20. Rival had the next touch in 17/20 refined
baseline cases, and all 5/5 true-commit cases reached high commitment. The original 50% target was
therefore not informative enough on its own. The prospective replacement used for the live gate
was: eliminate the one refined exposure with at least one actually applied/explainable treatment
intervention, lose no more than one self-next-touch case, retain the one-tick bound, and protect
the true-commit controls. Zero-exposure count differences do not pass this replacement gate.

Three required natural Nexto fixtures were curated from the locally available Milestone 02 raw
session whose SHA-256 is
`8d43127ccddb204458bcf9386523d4a102c20cff12e6e4e18038cf465189cef9`:

| Fixture id | Committed fixture SHA-256 |
| --- | --- |
| `appa-2853d7379f43` | `92887786f635017da03d9bcacd04dc6a107eb7af6f46d311575528ac55ca2b63` |
| `appa-1d8d681ee3a9` | `5cb5c085ee7d29dff2c8d20ca46e6c180ae2000122ee356895c4caac2752cfd6` |
| `appa-b91d6aa6af1e` | `2c1b632a119a67ff769aa9dd97fbd4f501e31f4fd9e595cccb758bebd270fe8d` |

Fixtures are structural/offline evidence, not a causal gameplay result.

## RLBot v5 match configuration

Automated runners used the v3.2 contract: `Stadium_P`, five-minute Soccar, standard boost,
gravity, demolish, scoring and physics, `skip_replays=true`, `auto_save_replay=false`, debug
rendering `AlwaysOff`, performance monitor `NeverShow`, automatic agent start, readiness waiting,
and `ExistingMatchBehavior.Restart`. Natural/speed windows retained kickoff countdowns. Controlled
state-setting probes used instant start. Accelerated windows used only
`DesiredMatchInfo.game_speed`; they did not use the `TimeWarp` mutator and did not state-set cars,
ball, boost, score, or clock during natural play.

## Game-speed integrity gate

The user's 120 Hz physics-limit note is consistent with the roughly 15 policy decisions per
simulated game-second expected from Wisp's eight-physics-tick decision cadence. It is not necessary
or desirable to demand 600 policy decisions per wall-second at 5x. The gate separately measured
simulated-time acceleration, packet-reported `match_info.game_speed`, and responsiveness.

| Window | Requested | Packet speed min/median/max (samples) | Effective game-sec/wall-sec | Decisions/game-sec | Distinct actions | Wall seconds |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Nexto baseline | 1x | 1.0 / 1.0 / 1.0 (598) | 1.0006 | 14.8917 | 72 | 44.313 |
| Nexto accelerated | 5x | 1.0 / 1.0 / 1.0 (123) | 4.9156 | 12.3886 | 58 | 11.250 |
| Wisp accelerated | 5x | 1.0 / 1.0 / 1.0 (122) | 4.9950 | 13.6763 | 60 | 11.672 |

All three windows completed, bots remained responsive, JSON telemetry was valid, representative
action distributions were present, and the Nexto accelerated/1x decision-cadence ratio was
0.8319. Direct state setting therefore produced real wall-clock acceleration. However, every
active 5x packet still reported `game_speed=1.0`; the primary schema defines this field as the
current game-speed multiplier. The strict handoff gate consequently rejected 5x rather than
silently treating effective wall acceleration as packet validation. See the
[RLBot game-data schema](https://github.com/RLBot/flatbuffers-schema/blob/main/schema/gamedata.fbs)
and [desired-state schema](https://github.com/RLBot/flatbuffers-schema/blob/main/schema/interfacepacket.fbs).

The bounded fallback search was also explicit:

| Requested fallback | Active packet samples | Packet median | Effective game-sec/wall-sec | State-setting applies | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| 4x | 62 | 1.0 | 3.9898 | 7 | rejected |
| 3x | 82 | 1.0 | 3.0067 | 9 | rejected |
| 2x | 122 | 1.0 | 2.0035 | 13 | rejected |

The selected evidence speed was therefore **1x**. Speed-integrity/fallback work consumed zero
natural matches and took 107.547 wall seconds in total. The native server reported startup queue
warnings at 1/10 missed messages in the 1x window and at 1/10/100 in the accelerated windows.
Those native console warnings are called out because Python's logging handler cannot capture the
server's separate stdout stream.

## Two-lane capability test

Exactly one bounded two-lane attempt was made. It is **unsupported on this machine for this
milestone**:

- two uniquely named RLBotServer processes started, but the startup race assigned both port
  `23234`;
- lane 2 failed to bind with Windows `SocketException 10048` (address already in use);
- both Python clients then attached to the surviving lane-1 server/port;
- the maximum simultaneous `RocketLeague.exe` process count was one (PID 18228);
- one smoke window completed against the shared match, the other timed out;
- two independent valid telemetry sessions were not established;
- cleanup reported connection-aborted `WinError 10053` after the shared server stopped, and a
  read-only process check confirmed no RLBotServer/Rocket League process remained.

Two clients sharing one server/match explicitly do not count as parallel matches. The harness was
stabilized after the run to wait for listener readiness, but the test was not repeated because the
handoff permits exactly one concurrency attempt. All subsequent execution used one sequential
lane. The committed report reduces 282 sampled process snapshots to 17 state-changing snapshots
while preserving the raw count and exact rejection reasons.

## Paired controlled A/B

### Attempt 1: `m03-conservative-v1`

Parameters: low/high commitment 0.34/0.70, pressure distance 1,900, pressure ETA 1.40 s,
projected-miss reference 450, control distance 650, max logit gap 0.85, and one policy-tick
deferral.

| Fake-pressure aggregate (20 cases/mode) | Baseline `off` | Treatment `intervene` | Delta |
| --- | ---: | ---: | ---: |
| Refined premature-release cases | 1 | 3 | +2; relative reduction -2.0 |
| Self next touch | 17 | 18 | +1 |
| Opponent next touch | 0 | 0 | 0 |
| None next touch | 3 | 2 | -1 |
| Mean maximum separation increase | 552.510 | 261.036 | -291.474 |
| Median maximum separation increase | 154.662 | 105.439 | -49.223 |
| Mean ending ETA advantage | 2.8606 | 2.8481 | -0.0125 |
| Eligible decisions / interventions | 0 / 0 | 0 / 0 | no treatment exposure |

The default treatment applied no action change. The treatment's three refined cases were all in
`boost_then_brake`; the baseline's one was in `delayed_challenge`. Because no intervention was
applied, live trajectory/count differences cannot be attributed to treatment.

True-commit controls reached high commitment in 5/5 cases under both modes. Both had self next
touch in 5/5, no opponent next touch, and no deferral. This protected result is useful, but it
cannot compensate for zero fake-pressure exposure and no demonstrated improvement.

### One coarse candidate: `m03-candidate-low0-gap1p5`

The initial audit found three target release events. Two were blocked only by low-state
classification after explicit abort evidence; one otherwise valid ambiguous event had logit gap
1.349 against a 0.85 limit, while baseline confidence was only 0.090 and probability gap 0.066.
The single allowed coarse candidate changed only low threshold to 0.0 and max logit gap to 1.5;
the deferral bound stayed at one tick.

| Candidate comparison (same recorded baseline) | Baseline `off` | Candidate treatment | Delta |
| --- | ---: | ---: | ---: |
| Refined premature-release cases | 1 | 0 | nominal -1 |
| Self next touch | 17 | 18 | +1 |
| Mean maximum separation increase | 552.510 | 473.739 | -78.771 |
| Median maximum separation increase | 154.662 | 235.761 | +81.099 |
| Mean ending ETA advantage | 2.8606 | 3.2995 | +0.4389 |
| Eligible decisions / interventions | 0 / 0 | 0 / 0 | no treatment exposure |

The apparent 1/20 to 0/20 count change is not causal: the candidate again applied zero
interventions. In the true-commit controls, high commitment remained 5/5, but next touch changed
from 5 self / 0 opponent to 4 self / 1 opponent. That observed regression and zero exposure reject
the candidate. No two-tick candidate or additional tuning was attempted.

Across both live controlled runs, 15 sessions and 75 scheduled cases completed in 398.749 runner
wall seconds, producing 4,356 policy decisions and 65,234,977 raw bytes with zero invalid JSON
records. Native RLBotServer startup queues reached 1 then 10 missed messages per family; no
sustained active-window escalation or session failure was observed. Large raw JSONL remains
local/Git-ignored; session ids, hashes, byte counts, status, schedules, and metrics are committed.

## Acceptance decision and natural budget

Challenge calibration is **rejected** because:

1. neither treatment attempt produced an eligible/applied intervention;
2. the default attempt worsened the refined count from 1/20 to 3/20;
3. the candidate's nominal improvement had no treatment exposure and therefore no causal meaning;
4. the candidate also lost one true-commit next-touch case to the opponent.

The implementation remains experimental and disabled. Stage 4 was correctly skipped:

| Natural item | Result |
| --- | --- |
| Nexto treatment matches | 0 |
| Wisp v2-75B treatment matches | 0 |
| Total natural budget consumed | 0 / 6 |
| Natural-match wall duration | 0 seconds |
| Natural challenge/intervention rates | not measured; controlled gate failed |
| Intervention next-touch outcomes | 0 self / 0 opponent / 0 none, because interventions = 0 |

## Runtime anomalies and limitations

- Native RLBotServer queue warnings are not represented in the Python `runtime_warnings` arrays;
  they are explicitly preserved in this document and the concurrency report.
- `Error connecting to 127.0.0.1:23233` appeared after intentional server/Rocket League shutdown,
  after evidence manifests had finalized; it was not a treatment-session crash.
- The user's known RLBot app lifecycle behavior was handled by reusing one runner-owned
  `MatchManager` through each series. No claim is made that this fixes the app's separate need to
  restart between manually launched games.
- The controlled harness matches state parameters but did not reproduce bit-identical trajectories
  across separate baseline/treatment launches. With zero interventions, the differing counts
  demonstrate why outcome deltas must not be called causal.
- The 5x/4x/3x/2x packet field remained 1.0 despite correct effective acceleration. Until the
  packet-observability discrepancy is prospectively resolved, 1x is the only handoff-valid
  evidence speed on this machine.

## Evidence artifacts

- `evidence/results/v3/offline_m02_controlled_refined.json`
- `evidence/results/v3/speed_integrity.json`
- `evidence/results/v3/speed_fallback_search.json`
- `evidence/results/v3/concurrency_capability.json`
- `evidence/results/v3/controlled_ab.json`
- `evidence/results/v3/controlled_candidate_low0_gap1p5.json`
- `evidence/results/v3/milestone_03_decision.json`

`milestone_03_decision.json` is the compact machine-readable acceptance record. The detailed A/B
reports retain per-behavior/per-case metrics and raw-session hashes.

## Verification

The implementation checkpoint passed 53 tests before live execution. Final verification on
2026-08-22 passed all of the following:

- the complete pytest suite with a fresh repository-local Windows `--basetemp`: **54 passed**,
  with only two known `torch.jit.load` deprecation warnings;
- Ruff over all 22 Rival-authored/modified Python files: passed;
- compileall over `bot`, `tools`, `scripts`, `probes`, and `tests`: passed;
- policy-freeze comparison against `4f2b21c00e2fcb7108ab1006fd950b066fbd0484`: passed;
- model smoke/hash verification: finite 90-value policy output from a 432-value observation, with
  frozen model hashes unchanged;
- decision-tick smoke: passed with one schema-v3 decision and start/decision/end records;
- parsing of all 13 v3 result and fixture JSON artifacts: passed;
- v3.3 and v3.0 committed-blob package manifests: passed;
- committed report invariants and all locally retained controlled raw-telemetry hashes: passed;
- installed Nexto/Wisp reference hashes and recorded match-configuration invariants: passed;
- `git diff --check`: passed.

The final release step also includes staged-blob review and remote `origin/main` readback after
push.

The broader untouched baseline/Wisp-source Ruff scan still contains the inherited findings already
documented by Milestone 02; no unrelated baseline cleanup was mixed into this rejected experiment.

## Next smallest evidence-backed target

Do not add another tactical rule. The next prospective authority should first make the controlled
challenge exposure reproducible: isolate/reset bot policy history and challenge episodes per case
or run each paired repetition from a fresh synchronized bot/match state, then prove that the same
release-sensitive baseline event appears under both modes. Only after deterministic exposure exists
should a new, versioned treatment parameter set be evaluated. The current rejected data must not be
reused as if it were held-out acceptance evidence.
