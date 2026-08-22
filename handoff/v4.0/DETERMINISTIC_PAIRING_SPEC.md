# Rival Milestone 04 — Deterministic Paired Exposure Specification

## Goal

Build a controlled-test mode in which equivalent `off` and `observe` challenge cases receive equivalent policy inputs and therefore produce equivalent Wisp actions until a treatment is actually allowed to diverge them.

Do **not** tune challenge-calibration thresholds first. Do **not** claim improvement from independent trajectories.

---

# 1. Preserve normal gameplay semantics

The default Rival runtime must remain equivalent to the current Milestone 03 default:

- `RIVAL_CHALLENGE_CALIBRATION_MODE=off`;
- inherited Wisp observation semantics;
- existing model files, legal-action mask, deterministic masked argmax, tick skip, and action delay;
- no deterministic-test seed or special reset behavior unless explicitly enabled by the controlled harness.

Do not remove opponent-slot shuffling from normal gameplay just to make tests easy.

---

# 2. Isolate observation randomness

`bot/obs_builder.py` currently calls module-global `random.shuffle()` on teammate and opponent buffers.

Refactor this minimally so `CustomObs` can optionally use an injected/resettable RNG while default behavior preserves the existing module-global random path.

A suitable interface may be equivalent to:

```python
CustomObs(rng=None)
set_test_seed(seed)
reset_test_seed(seed)
```

Exact names may differ.

Requirements:

- no test seed configured => existing production shuffle path;
- deterministic controlled mode => a dedicated `random.Random` or equivalent reproducible generator;
- reset the controlled RNG to a case-specific seed at every controlled scenario boundary;
- the same case ID/repetition must use the same seed under `off`, `observe`, and later `intervene`;
- different repetitions should deliberately use different named seeds so the experiment samples multiple valid Wisp observation permutations rather than one permanent slot arrangement.

Do not replace random shuffling with fixed opponent ordering globally.

---

# 3. Full controlled-case runtime reset

Create one explicit internal reset routine for controlled-case boundaries. Do not leave reset logic scattered across multiple ad-hoc branches.

At minimum reset/reinitialize, where applicable:

- `RocketSimStateAdapter` transient state;
- `prev_actions` for all players;
- `old_action`;
- `new_action`;
- tick-window state (`ticks`, `update_action_flag`);
- policy tick/case-local decision index;
- cached state;
- `last_decision`;
- `last_challenge_decision`;
- challenge calibration/commitment tracker history;
- controlled observation RNG to the case seed;
- any newly discovered temporal feature/state used by the policy wrapper or analyzer that can influence a supposedly fresh case.

Do not reset model weights or rebuild the model every case.

The observation explicitly contains `player.prev_action`, so resetting action history is mandatory.

## Detecting the boundary

Prefer a robust explicit test-only boundary mechanism. If the RLBot/state-setting API has no clean case identifier channel, a controlled-test-only state-discontinuity detector is acceptable because state-setting teleports are far larger than physically possible one-packet motion.

If using discontinuity detection:

- gate it behind an explicit controlled-test setting;
- use a threshold safely above normal 2300 uu/s per-packet motion;
- prevent repeated resets while the state setter is settling the same case;
- record `case_reset_reason`, `case_epoch`, and `case_seed` in telemetry.

Normal natural matches must not acquire new reset behavior merely because this harness exists.

---

# 4. Pair identity

Every controlled case must have a stable identity containing at least:

```text
scenario family
behavior
repetition
case seed
initial-state parameter hash
mode (off/observe/intervene)
code commit
model hashes
```

The runner must be able to pair an `off` case with the corresponding `observe` case by identity without relying on timestamps.

Persist the exact desired car/ball state and controlled-opponent parameters used for each pair.

---

# 5. Observation/action trace identity

Add a lightweight controlled-test trace fingerprint. Do not enable full raw observations in ordinary telemetry.

For controlled deterministic mode, record enough to compare corresponding decisions, preferably:

- stable hash of the exact float32 observation tensor bytes;
- legal-mask hash;
- selected baseline Wisp action index;
- top-N action indices/logits already available;
- previous action;
- case-local decision index;
- relevant packet/state fingerprint.

The observation hash must be computed from the tensor actually passed into the Wisp model, not from a separately reconstructed approximation.

## Reproducibility gate

Before testing `intervene`, run paired `off` vs `observe` cases. Because `observe` does not alter control output, the pair should remain equivalent.

For each paired case before any hypothetical treatment point:

- observation hashes must match at corresponding decision indices;
- legal masks must match;
- selected Wisp action indices must match;
- controller outputs must match;
- top policy logits should match within numerical tolerance;
- case reset seed/epoch must match.

If world/physics timing produces an unavoidable one-decision alignment offset, Codex may implement explicit alignment by simulated game time, but it must document and prove the alignment rather than silently accepting divergent traces.

Target: **5/5 reproducible repetitions** for the chosen exposure scenario. A weaker gate requires explicit evidence that exact identity is impossible for an external RLBot reason.

---

# 6. Reproduce a release-sensitive baseline event

The refined Milestone 03 detector found only one release-sensitive fake-pressure exposure in 20 broad fake cases. v4 must create or discover a controlled case where that event itself is repeatable.

Use a bounded search around existing fake-challenge geometry rather than hand-authoring a desired policy action.

Allowed search dimensions include small ranges of:

- Rival-to-ball distance and velocity;
- opponent separation/lateral offset;
- challenger speed;
- abort timing;
- ball speed/offset;
- Rival/opponent boost;
- controlled observation seed.

Do not change Wisp weights, observations, action masks, or the detector merely to manufacture success.

## Search budget

Keep the search bounded. Suggested maximum: 200 state/seed variants, with cheap early termination once a strong reproducible candidate is found.

A candidate exposure should satisfy:

- Rival plausibly controls/continues a grounded possession state;
- apparent pressure exists;
- baseline Wisp initiates the refined release-sensitive grounded jump/dodge behavior;
- the same baseline action/event appears in repeated `off` and `observe` runs from the paired state;
- no treatment action is required to produce the exposure.

Commit the final small scenario fixture/parameters; keep bulk search telemetry ignored with hashes/indexes.

---

# 7. Treatment boundary

Do **not** retune the rejected Milestone 03 treatment until Sections 1–6 pass.

If deterministic paired exposure passes, v4 may perform **one** prospective treatment experiment against the frozen reproducible cases using a new parameter version name. Do not reuse `m03-conservative-v1` or `m03-candidate-low0-gap1p5` as if they were held-out evidence.

The treatment experiment must compare the same case IDs/seeds and must record the exact first decision where the final Rival action diverges from baseline Wisp.

If no actual intervention occurs, the treatment has not been tested and cannot be accepted.

If the reproducibility gate fails, stop there, document the failure, and push the harness improvements without further gameplay tuning.

---

# 8. Tests

At minimum add tests proving:

- default `CustomObs` still uses the inherited nondeterministic shuffle path;
- controlled seed produces reproducible opponent-slot ordering;
- resetting the same case seed reproduces the same observation ordering;
- different case seeds can produce different valid permutations;
- controlled full-reset zeros/clears prior actions and tracker state;
- state-discontinuity reset does not fire during plausible normal one-packet motion;
- repeated setter packets for the same case do not continuously reset the bot;
- `off` and `observe` output parity remains exact when deterministic control is enabled;
- observation fingerprint hashes the actual model input tensor;
- existing policy-freeze and model/hash tests continue to pass.

Run the complete existing test suite too.

---

# 9. Deliverables

Commit/push:

- deterministic observation-RNG plumbing;
- controlled-case reset implementation;
- paired runner changes;
- observation/action trace fingerprinting;
- bounded exposure search tool/config;
- reproducibility report;
- curated reproducible challenge fixture if found;
- optional single treatment result only if the reproducibility gate passed;
- `docs/MILESTONE_04_RESULTS.md`;
- machine-readable `evidence/results/v4/` summary artifacts.

Large raw telemetry stays Git-ignored with SHA-256 provenance.