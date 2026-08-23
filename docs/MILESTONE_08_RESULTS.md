# Milestone 08 Results

Milestone 08 completed the bounded transfer-safe dual-rate campaign. The final
verdict is:

`partial_architecture_pass_learning_inconclusive`

The architecture, observation, temporal, fallback, checkpoint, export, native-rate
cadence, and severe-transfer-regression gates passed. The mechanics head was sampled
at a measurable rate, including a prospective user-requested increase after the
untouched 2M boundary. It did not become a deterministic controller source: every
deterministic headless and RLBot decision at the final boundary remained PASS.
Consequently, M08 does not demonstrate useful learned mechanics and does not justify
production promotion.

Final verification also records a local artifact-preservation limitation discovered
before the final commit: a legacy calibration script ignored `--help` and overwrote
its fixed ignored output paths. The original initial actor checkpoint was recovered
byte-for-byte, but the original 10,000-observation `.npy` corpus and original initial
TorchScript archive are no longer available locally. The tracked report was restored
exactly, and all campaign checkpoints, final exports, raw telemetry, and production
artifacts were revalidated as unaffected.

Production Rival remains the frozen Wisp policy at tick skip 8. No trained checkpoint
was promoted, and the rejected M06 20M actor remains diagnostic-only.

## Protected boundary and artifacts

- Required M07 ancestor:
  `10c41f708d6e8145bf719f8f322041e7753f6c3f`.
- Frozen `POLICY.lt` SHA-256:
  `1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7`.
- Frozen `SHARED_HEAD.lt` SHA-256:
  `3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42`.
- Frozen M07 zero-step strategic actor SHA-256:
  `4480696d89a3c88770807212dfbb5bc3cd4fb4492511e67a5b7dd1213c9aa3a7`.
- Exact expanded action-table file SHA-256:
  `bed450ded25a7bc624b17695913626fd478d00e11936413c7f5af5022585de2f`.
- Final 5M mechanics TorchScript SHA-256:
  `43065e434f30f3e0cf8a868777b01970f8bd1d581b5bade6e89f90f38adc2cce`.
- Final mechanics state SHA-256:
  `323e3bc09f9cdccb10c59f623610f73954302021cdcdb7927b80a61a47e5b66c`.

The paused pre-v4.1 strategy stash remains present as
`stash@{0}: On main: rival-v4-paused-superseded-before-v4.1`. Large checkpoints,
exports, datasets, raw rollouts, RLBot telemetry, and the RLViser executable remain
ignored. Available artifact paths, sizes, and hashes are recorded in the compact
evidence manifest; the unavailable original corpus/export hashes and quarantined
replacement hashes are recorded separately in
`training/results/milestone08/artifact_preservation_incident.json`.

## Final-verification artifact incident

During final command discovery,
`calibrate_m08_mechanics_prior.py --help` unexpectedly executed calibration because
the legacy script did not parse arguments. It replaced the tracked compact report and
three ignored files before the final commit.

Recovery was bounded and hash-gated:

- the tracked calibration report was restored exactly from the existing HEAD blob;
- the original `mechanics_initial_v1.pt` was reconstructed from its deterministic
  seed, recorded PASS bias, and metadata, then accepted only after its complete
  SHA-256 matched `0485c8d2...67ef6de`;
- a regenerated TorchScript has the exact original state and logits, but different
  archive bytes, so it is quarantined under an explicit regenerated filename and is
  not represented as the original export;
- the accidental replacement corpus is quarantined under an explicit rerun filename;
- the canonical original corpus and initial TorchScript paths are intentionally absent
  so future tooling cannot mistake replacements for the recorded originals.

Controlled reruns from the exact `7e2e119` source boundary with four fixed Python hash
seeds all produced different corpus hashes. The collector seeded its curriculum and
observation RNGs but had not fixed process-level ordering, and the original process
hash seed was not recorded. An unbounded search was not used to manufacture a match.
Therefore exact local replay of the original calibration corpus is no longer possible.

The script now uses normal argument parsing, makes `--help` side-effect-free, refuses
to replace any existing calibration output by default, and requires the explicit
`--overwrite` flag for a prospectively authorized rerun.

## Observation contract v2

`Wisp432ContractV2` repaired the policy-material live/training mismatch before PPO.
It shares the production ETA math and explicit state adapters while preserving the
operationally stable production observation path. The implementation covers the
two-pass 120 Hz ball-prediction lookup, cached ETA update, projected initial velocity,
boost-duration interpretation, exact linear ETA math, touch/handbrake timing, player
flags, and previous-controller semantics.

The gate reused 1,276 held natural live observations from M07:

| Metric | M07 training-style reconstruction | M08 contract v2 | Gate |
| --- | ---: | ---: | ---: |
| Frozen-Wisp masked top-1 agreement | 46.7085% | 98.6677% | >=97% hard; >=99% target |
| Disagreements | 680 / 1,276 | 17 / 1,276 | diagnostic |
| Mean JS divergence | not gate-qualified | 0.000314164 | <=0.002 |
| Largest single-group substitution effect | material ETA/car groups | 0.783699% | <=5% |
| Directly representable maximum error | mixed approximation | 4.768e-7 | <=1e-5 |

All hard conditions passed. The 99% target was narrowly missed, so the result is not
reported as perfect observation equivalence. Ball-prediction horizons accounted for
the largest remaining group sensitivity, but remained far below the 5% stop gate.

Randomized first-90 parity was exact on 4,096 tensors, and held-live first-90 parity
was exact on all 1,276 tensors when the same contract observation was supplied to the
frozen strategic implementations.

## Exact dual-rate architecture

The M08 candidate has two separate policy roles:

- The immutable Wisp strategic actor selects actions 0 through 89 on an eight-tick
  clock. It is not in either optimizer.
- The trainable mechanics actor and critic operate on a four-tick clock with exactly
  69 outputs: PASS plus global action indices 90 through 157.
- PASS leaves the strategic controller row untouched. An appended action owns only
  its mechanics window while the strategic scheduler continues advancing underneath.
- The optional generic eligibility gate always keeps PASS legal and contains no named
  mechanic macro or scripted scenario.

The tested strategic window is exactly:

`[previous, previous, previous, previous, previous, selected, selected, selected]`

The tested mechanics window is exactly:

`[previous_emitted, selected, selected, selected]`

Long traces proved that mechanics-disabled and forced-PASS execution produce identical
observations, first-90 logits, masks, strategic indices, scheduler state, and
controller rows. The fallback proof covered 1,024 physics ticks per agent and 256
mechanics windows with zero controller or observation error.

## What “cadence collapse” means

Cadence is a technical telemetry gate, not a winning/losing gate. A session collapses
cadence if its strategic clock no longer has an eight-tick mode, its mechanics clock
no longer has a four-tick mode, decision sequences stall/duplicate/cross sessions,
the sustained two-mechanics-per-strategic ratio falls below its threshold, lifecycle
partial windows become in-play partial windows, or controller/model/hash/action-map
identity fails.

Score is recorded only as behavioral context and for the separate bounded severe
transfer-regression check. A win cannot make cadence pass, and a loss cannot make it
fail.

The forced-PASS zero-step four-game RLBot v5 battery completed runtime-clean at 1-3,
18-22, goal differential -4. Its gate passed on the clocks, sequences, hashes, masks,
and controller pass-through invariants, not on that score.

At the 1M boundary, the game-speed-5 stress run failed the strict cadence threshold
because accelerated wall-clock scheduling introduced excess jitter. That failure is
retained. A prospective game-speed-1 rerun at Rocket League's native 120 Hz physics
rate passed all four sessions. The two results are attributed rather than overwriting
one another.

## Worker-count result and the earlier 12k measurement

The earlier M06 measurement did select 56 workers at 12,039.09 agent-steps/sec. That
number was measured on the M06 monolithic rollout workload. M08's observation v2,
frozen strategic inference inside every worker, separate mechanics inference, two
clocks, and compositor make each collected step substantially more expensive, so the
rates are not directly comparable.

The authorized short M08 sanity check measured:

| Workers | Sustained agent-steps/sec | CV | Mean worker RSS | Result |
| ---: | ---: | ---: | ---: | --- |
| 48 | 1,185.19 | 4.31% | 26,912.9 MiB | stable |
| 56 | 1,197.30 | 5.49% | 31,389.2 MiB | stable |
| 64 | 1,208.20 | 5.21% | 35,888.9 MiB | short sanity winner |

The 64-worker edge over 56 was only 0.91%, within the short-window variation. The
first 500k boundary ran at 64, but its next full-PPO resume failed before collecting
any new agent-step with Windows `WinError 1455` page-file allocation pressure. The
checkpoint remained exact. A committed prospective fallback authorization selected
56 for reliable continuation. The post-adjustment 2M-to-5M leg sustained a mean of
1,152.88 agent-steps/sec over 60 iterations, with a 1,095.00 to 1,185.19 range.

Thus, 64 was the narrowly fastest short microbenchmark, while 56 was the highest
worker count that completed the resumed campaign reliably on this M08 workload. M08
does not relabel the older 56-at-12k M06 result or claim that 64 is the machine's
universal optimum.

## Mechanics prior and requested exposure increase

The initial 69-output head was calibrated on 10,000 natural observations rather than
using the old monolithic appended-logit schedule:

- target/mean override probability: 3.0% / 3.0000003%;
- sampled override rate: 3.04%;
- deterministic override rate: 0%;
- conditional appended-action entropy: 4.2181 nats;
- 66 of 68 appended outputs were observed in the fixed 10,000-sample audit, while all
  outputs remained numerically non-starved.

These are the already-committed historical measurements. The original corpus hash is
still recorded as `489eabe3...24d720`, but its bytes are no longer locally available
after the final-verification incident described above.

At the untouched 1,999,776-step boundary, sampled use was still about 2.90% and every
deterministic decision was PASS. In response to the explicit request to turn the
mechanics head up, M08 made one hash-bound prospective adjustment before the next
rollout. It preserved the source checkpoint and loaded Adam state, changed only the
mechanics PASS-output bias, and targeted 10% mean natural-state override probability:

| Adjustment metric | Result |
| --- | ---: |
| Source mean override probability | 2.64047% |
| PASS-bias delta | -1.410392 |
| Adjusted mean override probability | 10.000002% |
| Fixed-corpus sampled override rate | 9.85% |
| Appended outputs sampled | 68 / 68 |
| Immediate deterministic override rate | 0% |

The first post-adjustment PPO iteration sampled 9.9648% overrides. Across the complete
2M-to-5M leg, 181,172 of 2,994,350 mechanics decisions were sampled overrides
(6.0505%). PPO then moved the head back toward PASS: the final iteration sampled
3.5352%, and final deterministic headless observations had only 3.4533% mean total
override probability.

This explains the inconclusive learning verdict. The intervention made the head
participate materially in sampled training and exercised all 68 appended actions, but
PASS still beat every individual appended action under argmax. Raising total override
mass enough to force deterministic selection across 68 competing actions would no
longer be a small, transfer-safe bias adjustment. M08 therefore records the limitation
instead of forcing an untrained high-risk controller change at the ceiling.

## Training boundaries

Training stopped below the authorized 5,000,000 ceiling at the nearest complete PPO
iteration: 4,999,790 agent-steps and 288 cumulative model updates.

| Boundary | Actual steps / updates | Sampled override rate | Headless frozen-Wisp record | Goal diff. | Deterministic override |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero-step | 0 / 0 | 3.0400% prior audit | 20-20 | 0 | 0% |
| 500k | 499,748 / 27 | 2.9294% | 22-18 | +4 | 0% |
| 1M | 999,822 / 54 | 2.9310% | 17-23 | -6 | 0% |
| 2M | 1,999,776 / 111 | 2.9028% | 24-16 | +8 | 0% |
| 5M | 4,999,790 / 288 | 6.0505% on adjusted 2M-5M leg; 3.5352% final iteration | 21-19 | +2 | 0% |

Every PPO iteration remained finite and inside the bounded override-health checks.
Outcome reward remained dominant over the separately logged possession, progress,
boost-efficiency, recovery, and mechanics-resource shaping. Natural 1v1 remained the
majority distribution. Strategic weights/hashes remained unchanged, and the final
fresh checkpoint reload recovered exact logits plus both optimizer states.

The headless scores are small comparison cells, not precise skill estimates. Their
variation does not prove mechanics learning, especially because deterministic
evaluation selected zero mechanics overrides at every boundary.

## RLBot v5 transfer

The required 1M and final candidate matrices used installed Nexto and Wisp, one blue
and one orange full match against each, with a fresh Rocket League process for every
match.

| Candidate | Game speed | Record | Goals | Goal differential | Runtime clean | Cadence |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Forced-PASS zero-step | 5x | 1-3 | 18-22 | -4 | 4 / 4 | passed |
| 1M native-rate | 1x | 0-4 | 12-16 | -4 | 4 / 4 | passed |
| 5M native-rate | 1x | 1-3 | 10-14 | -4 | 4 / 4 | passed |

The final 5M games were 0-2 and 0-3 against Nexto, then a 4-5 overtime loss and 6-4
win against Wisp. The final strict analyzer covered 43,704 mechanics decisions:

- minimum mechanics interval-within-one-tick rate: 99.9465%;
- minimum strategic interval-within-one-tick rate: 99.9584%;
- minimum sustained two-to-one clock ratio: 99.9585%;
- every session had mechanics mode 4 and strategic mode 8;
- every runtime/model/hash/mask/action mapping check passed;
- no invalid in-play partial window occurred;
- deterministic mechanics overrides: 0.

The final candidate did not trigger the severe RLBot regression gate, but the equal
-4 goal differential across these small transfer cells does not demonstrate
improvement. More importantly, the live candidate never left strategic PASS-through,
so these matches validate the architecture and frozen baseline, not learned mechanics.

## Optional RLViser spectator

`RivalRLViserSpectatorV2` remains opt-in, disabled by default, and separate from the
training/diagnostic hot path. Its final preflight verified the 69-output 5M actor,
exact TorchScript and pinned RLViser hashes, one spectator-owned environment,
120 Hz physics, four-tick mechanics, eight-tick strategic behavior, and PASS-versus-
override controller-source accounting. No PPO worker is rendered.

Install/check the optional viewer, then watch the final local candidate at
approximately real time:

```powershell
./training/install_rlviser_spectator.ps1

training/.venv/Scripts/python.exe training/scripts/run_m07_rlviser_spectator.py `
  --checkpoint training/artifacts/milestone08/005m/mechanics_actor.ts `
  --opponent frozen-wisp --tick-skip auto
```

Use Ctrl+C to stop. `--opponent selected` enables rendered dual-rate self-play and
`--playback-speed` adjusts wall-clock pacing. This is a spectator seam only; it does
not change production defaults.

## Final verification

The implementation, test, model, checkpoint, export, and transfer checks passed. The
overall machine-readable verification status is
`completed_with_artifact_preservation_limitation` because the original prior corpus
and initial TorchScript archive could not be recovered byte-for-byte:

- complete production suite: 87 passed, 0 failed, 2 known deprecation warnings;
- complete training suite: 43 passed, 0 failed, 43 known Torch deprecation warnings;
- repository-wide Ruff: passed;
- production, test, training-runtime, script, and training-test compileall: passed;
- frozen `POLICY.lt`, `SHARED_HEAD.lt`, and zero-step actor hashes: exact;
- direct 90-row action prefix and exact 158-row table proof: passed;
- observation, scheduler, forced-PASS, zero-step, checkpoint, export, native-rate
  cadence, and RLViser-isolation gates: passed;
- final checkpoint fresh-instance logit reload: exact with zero maximum error;
- policy and critic optimizer state reload: passed;
- all compact M08 JSON reports parsed and revalidated;
- exact original initial actor checkpoint recovery: passed;
- exact original prior-corpus and initial-TorchScript local preservation: failed and
  explicitly bounded in the incident report;
- `git diff --check`: passed.

Regenerate the compact final verification and manifest after the measured reports and
ignored artifacts are present:

```powershell
training/.venv/Scripts/python.exe training/scripts/finalize_m08_evidence.py
```

The exact commands and counts are stored in
`training/results/milestone08/final_verification.json`. The compact report and source-
artifact hashes are stored in `training/results/milestone08/evidence_manifest.json`.
Exact post-push remote readback is reported after the final commit because a commit
cannot contain a readback of itself.

## Evidence index

The complete compact index is
`training/results/milestone08/evidence_manifest.json`. It includes the observation and
pretraining gates; prior and user-directed adjustment; throughput/fallback; every PPO
and headless boundary; 1M accelerated/native attribution; final native RLBot matrix
and cadence gate; both candidate exports; both RLViser preflights; final verification;
the preservation-incident report; and hashes/sizes for all available ignored source
artifacts.

## Final decision

M08 proved that Rival can preserve frozen Wisp's native strategic behavior while a
separate four-tick mechanics actor trains, checkpoints, exports, resumes, and transfers
through RLBot v5 without cadence collapse or severe baseline regression. It did not
prove that the mechanics actor had learned a useful deterministic intervention by 5M.

Therefore:

- verdict: `partial_architecture_pass_learning_inconclusive`;
- production promotion: **not authorized and not performed**;
- production default: frozen Wisp at tick skip 8;
- final 5M candidate: diagnostic/research artifact only;
- exact local artifact preservation: incomplete for the original prior corpus and
  original initial TorchScript archive, with replacements quarantined and reported;
- any longer mechanics campaign, hierarchical PASS/action redesign, or constrained
  strategic fine-tuning requires new prospective authority.
