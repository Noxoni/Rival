# Rival Milestone 02 results

Milestone 02 is complete as an evidence and reproducibility milestone. Rival's normal controller
selection remains the frozen Milestone 01 Wisp-derived action. No gameplay correction was made.
All detector findings below are review candidates, not confirmed defects or claims of superiority.

## Provenance and frozen-policy invariant

- Milestone 01 policy baseline: `4f2b21c00e2fcb7108ab1006fd950b066fbd0484`
- Portable Windows packaging commit: `5c4ab880ce593dc944b4f42d3f22e31574a4ea57`
- Milestone 02 instrumentation/harness commit used by every primary session:
  `36e59e5c0c84b6b642fdfb09d8d3c7fbe91b6614`
- `STRATEGIC_OVERRIDES_ENABLED`: `false`
- Deterministic policy: `true`
- Full raw logits: disabled
- `POLICY.lt` SHA-256:
  `1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7`
- `SHARED_HEAD.lt` SHA-256:
  `3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42`

`scripts/verify_policy_freeze.py` exact-compares policy-critical files and the observation,
masking, inference, and selection core against the baseline commit. It passed after all Milestone
02 analysis changes.

## Environment and reference identity

| Component | Version or SHA-256 |
| --- | --- |
| Python | `3.12.13` |
| RLBot | `2.0.0b54` |
| RocketSim | `2.2.1` |
| Torch | `2.13.0` |
| Nexto config | `0e4818a04e2364a6b8c84b38be24db8f6856f7377f7cf3b802fdfac4b526828f` |
| Nexto executable | `6371a5b9dd740aea858d86532f928e0fdb13bf387d8379f76ccd679c9b33e845` |
| Wisp v2-75B config | `75a60b870759f855c4d900561553ae7535c379eed4df20dc63a0969bcac18825` |
| Wisp v2-75B executable | `b9dbe32bdae28c299daffcc0673f5a438d8013a553be3938dafaa112884e7184` |

The installed reference configs and executables were validated in place. The runner did not copy,
overwrite, or otherwise mutate the installed RLBot v5 BotPack tree.

## Natural-match baseline

All six requested natural matches completed under standard `Stadium_P` Soccar, five-minute match
length, Default game speed, normal boost, default gravity/physics, and deterministic Rival policy.
Scores are shown as **Rival-opponent**, only as baseline observations.

| Opponent | Rival side | Rival-opponent | Session id | Decisions | Raw bytes | Raw SHA-256 |
| --- | --- | ---: | --- | ---: | ---: | --- |
| Nexto | blue | 4-3 (OT) | `rival-v2-natural-nexto-blue-20260822T172339Z-79e6298a` | 5,468 | 66,905,034 | `8d43127ccddb204458bcf9386523d4a102c20cff12e6e4e18038cf465189cef9` |
| Nexto | orange | 5-7 | `rival-v2-natural-nexto-orange-20260822T173013Z-345c6877` | 5,844 | 71,186,002 | `c2aba9f23cda34af8245d436188c2c2dc2379765bcff6d4620f0c1f0c654cf8d` |
| Nexto | blue | 3-7 | `rival-v2-natural-nexto-blue-20260822T173729Z-ec302968` | 5,484 | 66,968,433 | `98565158bf95e9a462bc64b75ee620c342fdd6fa65396ac3bfaeec2e8d55b5f1` |
| Wisp v2-75B | blue | 4-6 | `rival-v2-natural-wisp-blue-20260822T174422Z-1d48f31f` | 5,561 | 67,864,455 | `f940b9dc1b304ce33cdea3ae46ec728d169c0fdcb01f77c4d8d2dbb632d772b5` |
| Wisp v2-75B | orange | 6-7 (OT) | `rival-v2-natural-wisp-orange-20260822T175114Z-7c4865d5` | 6,066 | 73,895,761 | `7e6ad997790453f686f4d2dff29883c38393e344fc77d71319dff1d9fce2da0f` |
| Wisp v2-75B | blue | 3-1 | `rival-v2-natural-wisp-blue-20260822T175850Z-72963bbf` | 5,000 | 61,477,325 | `d981207120ba8425742408ca40b22e53f110cd0ef923552d261a3e0e69cd23b8` |

This is three complete matches per opponent, with blue/orange alternation where practical. Rival's
observed records were 1-2 against each reference in this bounded sample; the sample is not a skill
ranking or statistical superiority claim.

## Controlled probes

### Fake-challenge family

Five repetitions of each required behavior completed, for 25 scheduled cases. The current
release-like measurement is deliberately broad: any jump, material pitch, or boost response in the
bounded probe window. It is a comparison signal, not a verdict that each response was wrong.

| Ground-truth behavior | Cases | Release-like | Next touch self/opponent/none | Mean ball-separation increase | Mean rank score |
| --- | ---: | ---: | ---: | ---: | ---: |
| `boost_then_brake` | 5 | 5 | 4 / 0 / 1 | 39.0 | 62.0 |
| `boost_then_veer` | 5 | 5 | 5 / 0 / 0 | 91.8 | 63.2 |
| `delayed_challenge` | 5 | 5 | 3 / 0 / 2 | 438.4 | 67.2 |
| `jump_fake` | 5 | 5 | 5 / 0 / 0 | 622.4 | 70.4 |
| `true_commit` | 5 | 5 | 4 / 1 / 0 | 1,000.0 | 52.2 |

The four non-commit behaviors therefore produced 20/20 release-like responses, but no fake probe
gave the opponent the next touch inside the detector window; Rival took the next touch in 17/20.
The evidence supports investigating commitment calibration, while explicitly not proving that each
controlled response threw away possession.

### Resource-aerial family

Eight scheduled cases completed in one state-setting session. The grid varied configured boost,
ball height/distance and velocity, opponent pressure, field position, and availability of a useful
ground alternative. It exposed distinct policy-response regions: only 4/61 decisions were
aerial-like in `low-high-mediumpressure`, versus 24/24 in `high-far-lowpressure`.

| Case | Configured/observed start boost | Height/distance | Pressure; ground alternative | Aerial-like decisions | Jump decisions | Boost decisions |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `low-near-lowpressure` | 8 / 20.0 | 500 / 900 | low; yes | 45/55 (81.8%) | 6 | 9 |
| `low-far-highpressure` | 8 / 8.0 | 900 / 1,700 | high; yes | 22/61 (36.1%) | 2 | 3 |
| `low-high-mediumpressure` | 12 / 11.7 | 1,250 / 1,400 | medium; no | 4/61 (6.6%) | 0 | 2 |
| `mid-near-highpressure` | 28 / 19.4 | 650 / 850 | high; yes | 41/60 (68.3%) | 4 | 9 |
| `mid-far-lowpressure` | 28 / 47.2 | 1,000 / 1,800 | low; no | 39/41 (95.1%) | 5 | 20 |
| `mid-high-mediumpressure` | 35 / 33.3 | 1,450 / 1,500 | medium; no | 25/37 (67.6%) | 2 | 18 |
| `high-near-highpressure` | 70 / 82.0 | 700 / 900 | high; yes | 37/54 (68.5%) | 6 | 23 |
| `high-far-lowpressure` | 70 / 33.3 | 1,200 / 1,900 | low; no | 24/24 (100.0%) | 4 | 19 |

Configured and first-observed boost differ in several cases because the policy can collect pads or
the state can settle before the first logged decision in the scheduled window. The raw observed
start/minimum/end boost, pad state, touch data, scores, and full packet snapshots remain preserved;
the configured values are not substituted for observations in candidate ranking.

## Telemetry and artifact integrity

The 12 primary sessions contain 35,230 policy decisions and 430,203,635 raw bytes (410.274 MiB),
with zero invalid JSON records. Every manifest completed with either `match_phase_ended` or
`controlled_probe_schedule_complete`. The six natural matches contributed 33,423 decisions at a
weighted 13.62 decisions per wall-clock second; individual match rates ranged from 13.34 to 14.14.

Large JSONL files remain ignored beneath `evidence/raw/<session-id>/`. Their exact local paths,
sizes, hashes, scores, side assignments, status, model/reference hashes, and termination reasons
are committed in `evidence/results/v2/session_index.json`. One earlier 53-decision smoke probe from
commit `5c4ab880ce593dc944b4f42d3f22e31574a4ea57` is identified there but excluded from all primary
counts because it predates the final instrumentation commit.

Replay auto-save was requested, but RLBot/Rocket League produced no replay files; all 12 primary
manifests contain an empty replay list. Event records therefore keep `replay_path: null` and retain
raw telemetry timestamps and bounded decision ranges for reconstruction.

## Analyzer results

Detector: `rival-m02-events-v1`. The committed JSON report retains the full counts and the ten
highest-ranked event records per class; the ignored local full report retains all event envelopes.
State-setting probe families are isolated to their intended detectors so discontinuous state reset
changes cannot masquerade as pad pickups or unrelated challenge events.

| Candidate class | Natural | Controlled | Total |
| --- | ---: | ---: | ---: |
| `resource_stressed_aerial` | 273 | 8 | 281 |
| `boost_detour_possession_loss` | 300 | 0 | 300 |
| `apparent_vs_actual_challenge` | 845 | 25 | 870 |
| **All classes** | **1,418** | **33** | **1,451** |

Detector parameters preserved in the JSON report:

| Parameter | Value |
| --- | ---: |
| pre/post window | 1.5 s / 4.0 s |
| aerial minimum ball height | 300 uu |
| aerial low-boost ranking reference | 30 |
| aerial minimum distance | 650 uu |
| boost pickup minimum gain | 5 |
| detour minimum distance increase | 150 uu |
| challenge maximum opponent-ball distance | 1,900 uu |
| challenge maximum Rival-ball distance | 1,000 uu |
| challenge minimum closing speed | 250 uu/s |

### Highest-ranked review candidates

| Class/event | Session and game time | Rank | Bounded next touch | Why it ranked |
| --- | --- | ---: | --- | --- |
| `resource_stressed_aerial` / `reso-6ce872d88ccc` | Wisp blue `...T174422Z-1d48f31f`, 105.175 s | 85.286 | none | elevated-ball aerial transition, low observed boost, no touch or ground recovery in window |
| `resource_stressed_aerial` / `reso-2b6fe4e80ec1` | Wisp blue `...T174422Z-1d48f31f`, 125.083 s | 83.719 | opponent | elevated/distant low-resource transition followed by opponent touch |
| `resource_stressed_aerial` / `reso-59e994329842` | Wisp orange `...T175114Z-7c4865d5`, 160.625 s | 82.769 | opponent | elevated/distant low-resource transition followed by opponent touch |
| `boost_detour_possession_loss` / `boos-161e02d9648e` | Nexto blue `...T172339Z-79e6298a`, 77.708 s | 87.000 | opponent | boost gain, 661 uu added distance, ETA proxy flip, then opponent touch |
| `boost_detour_possession_loss` / `boos-ee643a0639a5` | Nexto blue `...T173729Z-ec302968`, 73.100 s | 87.000 | opponent | boost gain, 990 uu added distance, ETA proxy flip, then opponent touch |
| `boost_detour_possession_loss` / `boos-6c46d0dff757` | Wisp blue `...T174422Z-1d48f31f`, 381.817 s | 87.000 | opponent | boost gain, ETA proxy flip, then opponent touch |
| `apparent_vs_actual_challenge` / `appa-2853d7379f43` | Nexto blue `...T172339Z-79e6298a`, 90.542 s | 95.000 | opponent | closing threshold crossed then aborted; Rival jumped; opponent next touch |
| `apparent_vs_actual_challenge` / `appa-1d8d681ee3a9` | Nexto blue `...T172339Z-79e6298a`, 127.967 s | 95.000 | opponent | closing threshold crossed then aborted; Rival jumped; opponent next touch |
| `apparent_vs_actual_challenge` / `appa-b91d6aa6af1e` | Nexto blue `...T172339Z-79e6298a`, 136.033 s | 95.000 | opponent | closing threshold crossed then aborted; Rival jumped; opponent next touch |

The complete committed top-five-per-class explanations are in
`evidence/results/v2/candidate_events.md`. These are ranked correlation windows; causal review or a
controlled before/after experiment is still required before calling any individual event a defect.

## Curated fixtures

- `fixtures/evidence/apparent_vs_actual_challenge__appa-1662387304b0.json` — controlled
  `jump_fake` ground truth, source hash
  `49c8fb8a0d2617a5cf889c67c37f88a30d9280d54ce0835efbfaf324836c1f38`.
- `fixtures/evidence/boost_detour_possession_loss__boos-161e02d9648e.json` — natural Nexto
  detour/ETA-flip window, source hash
  `8d43127ccddb204458bcf9386523d4a102c20cff12e6e4e18038cf465189cef9`.
- `fixtures/evidence/resource_stressed_aerial__reso-89d7fab671f0.json` — controlled
  `high-near-highpressure` state, source hash
  `ad2e15e2b62aa642e2f300b48267f290f70e52d34dd140ba06dbd4353b815226`.

Each fixture embeds the event envelope, detector identity, source-session/hash identity, bounded
sequential schema-v2 decisions, policy alternatives, packet state, and reconstruction metadata.

## Verification

- `python -m pytest -q --basetemp G:\dev\RLBot-Rival\.pytest_tmp\m02-full-final`:
  **31 passed**, with two TorchScript deprecation warnings only.
- Ruff across every Python file added or modified since the frozen Milestone 01 commit:
  **all checks passed**.
- `python -m compileall -q bot tools scripts probes tests`: **passed**.
- `python scripts/verify_policy_freeze.py`: **passed** against
  `4f2b21c00e2fcb7108ab1006fd950b066fbd0484`.
- `python scripts/smoke_model.py`: **passed**; model hashes matched, output shape was 90, selected
  action and compatibility action were both index 16.
- `python scripts/smoke_decision_tick.py`: **passed**; action index 12 and exactly one
  `rival_session_start`, one decision, and one `rival_session_end` record under schema v2.
- Candidate report, session index, and all curated fixture JSON: **parsed successfully**.
- `git diff --check`: **passed**.

A deliberately broader Ruff scan of untouched baseline/Wisp-source code reports four pre-existing
findings in `bot/backend/gamestate/rot_mat.py` and `bot/eta.py`. Neither file changed in Milestone
02; they were not edited because this milestone freezes gameplay and avoids unrelated baseline
cleanup.

## Runtime anomalies and limitations

- RLBot's server reported outbound queue-full warnings during startup, escalating through 1, 10,
  and 100 missed outbound packets in some launches. Decision telemetry still finalized with zero
  invalid records, but the warnings mean this run cannot claim packet-perfect RLBot delivery.
- One Nexto match logged a transient baseline `rough_eta` NaN-to-integer error at the
  regulation/overtime boundary. The bot recovered on later packets and the match/session completed.
- Sessions launched after overtime briefly exposed a `-300` match timer before the new match
  entered Active phase. The runner normalized after activation, and analyzer reset/time-rewind
  segmentation prevents cross-match event windows.
- RLBot v5 can leave a stale app/server connection between launches. The evidence series reused one
  runner-owned server connection, avoiding repeated app restarts. This is the previously observed
  RLBot app lifecycle issue, not a Rival exit-code diagnosis.
- `SocketRelay disconnected unexpectedly` appeared only after intentional match stop during parts
  of the Wisp series. The affected manifests had already finalized as complete.
- No `.replay` file was produced despite auto-save configuration, so visual replay review is a
  blocker for this evidence set. Raw schema-v2 reconstruction remains available.
- Natural commitment labels are geometric/input inference. Controlled probe labels are stronger,
  but their scripted state distribution is narrower than full-match play.

## Recommended first Milestone 03 defect

**Challenge-commitment calibration: distinguish a genuinely ball-intersecting commitment from
fake, veering, braking, or delayed pressure before choosing a possession-releasing response.**

This is the first recommended behavior target because it has a repeatable 25-case harness, all
20 controlled fake cases elicited the broad release-like response signal, the jump/delayed variants
produced substantial mean ball separation, and the highest-ranked natural Nexto windows repeatedly
combined an aborted closing signal, Rival jump, and opponent next touch. It is competitively
meaningful and narrow enough for an isolated before/after policy-side experiment using the curated
`jump_fake` fixture plus the full fake/true-commit suite.

The recommendation is intentionally calibrated to the limits of the evidence: controlled fake
probes did not give the opponent the next touch inside the bounded window, and Rival retained the
next touch in 17/20. Milestone 03 should test a commitment-sensitive change against both fake and
true-commit controls; it should not implement a blanket "never release" rule.
