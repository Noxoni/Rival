# Milestone 10.7 Stage-1 action-policy correction — final result

Status: **completed at the required +2.5 learner-hour review boundary; capability gate failed; no promotion**.

## Executive conclusion

M10.7 fixed the specific M10.6 deployment catastrophe, but it did not produce a useful or monotonically improving ball-acquisition policy.

- The old joint eight-way categorical head is gone. Jump, boost, and handbrake are independent Bernoulli branches and PPO replays the sum of their effective-probability log probabilities plus the five tanh-Gaussian analog terms.
- The originally proposed convex persistence formula was rejected before PPO because, with persistence greater than 0.5, deterministic thresholding makes both button states absorbing. With a reset OFF state, a deterministic button could never turn ON; once ON, it could never turn OFF.
- The user-authorized correction applies persistence as a symmetric log-odds prior:

  `effective_p = sigmoid(policy_logit + (2 * previous_bit - 1) * logit(persistence))`

  The requested persistence values remain unchanged: jump 0.95, boost 0.90, handbrake 0.90. Tests and preflight evidence prove OFF-to-ON and ON-to-OFF are both reachable, all eight physical combinations remain representable, and sampling, log-probability replay, and deterministic mode all use the same effective probability.
- This prevented the M10.6 approximately 97.8% deterministic jump-plus-boost collapse. M10.7 deterministic evaluation selected jump, boost, and handbrake OFF throughout all measured boundaries. That is evidence that the old catastrophe is absent, not evidence that the button branches learned useful control.
- The supervised disposable diagnostic passed easily: held-out MSE fell from 0.384215 to 0.022075 and steering-sign accuracy reached 98.10%. RivalObsV1, the transferred encoder, and the analog head can represent a basic turn-toward-ball mapping.
- PPO capability was not monotonic. Deterministic first contact peaked at 24.6% at +1h and then fell to 11.2% at +2.5h, below the untouched source's 18.0%. The final no-touch rate was 99.0%, second-contact success was 0.8%, and no deterministic episode achieved three contacts.
- Final stochastic first-contact success was 8.8%, below deterministic 11.2%. Therefore stochastic training behavior is not hiding a better policy behind deterministic deployment.

The result points to PPO credit assignment/training formulation and learned policy-output behavior, especially analog control, as the next investigation target. It does **not** authorize another reward-magnitude retune. The M10.6 reward remained frozen throughout.

## Lineage and architecture

- Source checkpoint: `training/checkpoints/milestone10_1/boundaries/plus-010h/032019870`
- Required source actor SHA-256: `e6b9fd1a38dbb5c6670711a8b47769a628d2f20caf7c88738f372d0839b7a3b6` (verified exact)
- Exactly transferred: 1,961,692 encoder/trunk and analog-head parameters, including all five analog means and all five analog log-std parameters
- Newly initialized only: `action_head.button_logits.weight` and `action_head.button_logits.bias` (1,539 parameters)
- Fresh critic and empty fresh actor/critic optimizer states: verified
- Final checkpoint: `training/checkpoints/milestone10_7/stage_1/boundaries/plus-002p5h/001080011`
- Final actor SHA-256: `5d88514e345e9aa73adc398be547104f9ad5e8901ff7274b6a23f839bb45fc3c`
- Final experience: 1,080,011 learner steps, 2.500025 learner-simulated hours, 44 PPO model updates
- Stage 2 and production promotion: not authorized

RivalPolicyV1, RivalObsV1, RivalActionV1, RivalCanonicalStateV1, native 120-Hz control, no action repeat, one-tick delay, Stage-1 curriculum, and dummy-agent exclusion remained unchanged. Frozen production Wisp artifacts remained byte-identical.

## Hard preflight

All preflight gates passed before campaign PPO:

- Frozen M10.6 reward truth table, including three separated +10 contacts, speed-independent acquisition pressure, uncapped heading delta, uncapped car-caused distance progress, and zero for all prohibited components
- Native 120-Hz RocketSim action mapping for throttle, steer, pitch, yaw, roll, jump, boost, handbrake, simultaneous jump+boost, all eight button combinations, one-tick delay, and no hidden action repeat
- Exact physical-action log-probability replay: zero same-distribution error; independent implementation maximum error `9.536743e-07`
- Corrected persistence non-absorption and bidirectional deterministic reachability
- Frozen eight-category observation corpus diagnostics
- Disposable supervised directional-control diagnostic
- Disposable 24,024-step real-PPO smoke, all eight action branches with finite nonzero gradients, all eight button combinations sampled, exact rollout replay, and worker cleanup
- Initialization checkpoint exact reload

## Frozen-corpus capability curve

Each row is the same frozen 500-episode Stage-1 corpus. Stochastic evaluation uses the exact same episode states as deterministic evaluation.

| Boundary | Deterministic first | second | third | all three | Stochastic first | second | third | all three |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Source transfer | 18.0% | 3.2% | 0.8% | 0.8% | 8.4% | 0.2% | 0.0% | 0.0% |
| +0.5h | 17.8% | 3.0% | 1.2% | 1.2% | 9.6% | 1.2% | 0.0% | 0.0% |
| +1h | 24.6% | 6.6% | 5.2% | 5.2% | 10.6% | 1.2% | 0.4% | 0.4% |
| +2.5h | 11.2% | 0.8% | 0.0% | 0.0% | 8.8% | 0.8% | 0.6% | 0.6% |

Final stochastic-minus-deterministic gaps were -2.4 percentage points for first contact, 0.0 for second, +0.6 for third, and +0.6 for all three. This does not meet the predefined materially-better-stochastic threshold.

## Final +2.5h behavior

Deterministic deployment:

- First-contact success: 56/500 (11.2%)
- Second-contact success: 4/500 (0.8%)
- Third-contact/all-three success: 0/500 (0.0%)
- No-touch timeout: 495/500 (99.0%)
- Mean time to first contact among successes: 1.7851 s
- Mean first-to-second interval among successes: 1.8313 s
- Second-to-third interval: unavailable because no third contact occurred
- Failed initial-to-terminal distance: 2,180.42 uu -> 4,010.70 uu
- Failed initial-to-terminal alignment: +0.5949 -> -0.2385
- Corpus totals: heading reward -665.0767, distance reward -266.8023, acquisition-time penalty -8,137.1383
- Mean actions: throttle +0.0743, steer +0.0780; jump 0.0%, boost 0.0%, handbrake 0.0%

Stochastic control:

- First/second/third/all-three: 8.8% / 0.8% / 0.6% / 0.6%
- No-touch timeout: 98.2%
- Mean time to first contact among successes: 2.6129 s
- Mean first-to-second interval: 2.6813 s; mean second-to-third interval: 4.9861 s
- Failed initial-to-terminal distance: 2,108.49 uu -> 4,342.88 uu
- Failed initial-to-terminal alignment: +0.6155 -> -0.0103
- Corpus totals: heading reward -499.4423, distance reward -382.7258, acquisition-time penalty -8,115.9283
- Mean actions: throttle +0.0877, steer -0.0161; jump 41.41%, boost 40.35%, handbrake 50.65%

The button entropy coefficient reached exactly zero after +0.5 learner hour and remained zero. The final frozen-corpus base means remained uncertain (jump 0.4639, boost 0.4527, handbrake 0.5038), while reset-state effective means were 0.0436, 0.0842, and 0.1014 respectively. Deterministic OFF is therefore not being described as learned button competence.

Recorded stochastic run-duration diagnostics at +2.5h were: jump mean 20.72 ticks (p90 48, p99 89, max 129), boost mean 9.85 ticks (p90 22, p99 44, max 73), and handbrake mean 10.20 ticks (p90 23, p99 47.85, max 82). These show persistence without hidden action repeat; a fresh physical action was still produced every 120-Hz tick.

## Verification and disposition

- Production tests: 91 passed
- Training tests: 196 passed
- Ruff: passed
- `compileall` for production, training implementation, scripts, and tests: passed
- Repository JSON parse: passed
- Final checkpoint loaded twice with identical actor state and exact output parity (maximum absolute difference 0.0)
- Frozen Wisp SHA-256 values unchanged: `POLICY.lt` `1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7`; `SHARED_HEAD.lt` `3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42`
- Remaining M10.7 Python training/evaluation processes: 0
- Historical stash preserved: `stash@{0}: On main: rival-v4-paused-superseded-before-v4.1`
- Checkpoints remain Git-ignored

Disposition: stop at +2.5h, preserve the exact checkpoint and evidence, do not promote, do not continue this actor automatically, and do not retune the frozen reward as a response to this failure.
