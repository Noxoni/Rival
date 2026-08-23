# Dual-rate architecture

## Why this exists

Milestone 07 proved that the Wisp strategic policy itself is cadence-sensitive. The exact zero-step reconstruction remained healthy enough at tick 8 but collapsed at tick 4. Therefore high-frequency mechanics capability must not require querying/retraining the entire strategic policy at high frequency.

## Clocks

Rocket League / RocketSim physics remain 120 Hz.

### Strategic clock

Every 8 physics ticks:

1. build the strategic Wisp-compatible observation;
2. frozen Wisp branch selects one of actions `0..89`;
3. schedule that decision using production Wisp temporal semantics.

For a stable previous action `P` and newly selected strategic action `S`, the eight-tick window is:

`P P P P P S S S`

The next strategic decision advances the same explicit scheduler state. Do not emulate this with the generic RocketSim one-tick delay.

### Mechanics clock

Every 4 physics ticks:

1. build the mechanics observation;
2. mechanics actor chooses one of 69 outputs;
3. output 0 is `PASS`;
4. outputs 1..68 map exactly to global expanded actions `90..157`.

For a mechanics override `M`, its four-tick window follows the verified four-tick delayed semantics:

`previous M M M`

The implementation must define what `previous` means when entering or continuing an override and test consecutive override/pass transitions.

## Compositor

The strategic scheduler always advances according to its own clock, including while a mechanics override is active.

At each physics tick:

- compute the controller row the strategic scheduler would have emitted;
- if no mechanics override owns the tick, emit the strategic row;
- if the mechanics scheduler owns the tick, emit its override row;
- never rewrite the strategic policy logits or selected strategic action because of the mechanics branch.

This makes mechanics a true bounded overlay rather than a second monolithic policy.

## PASS invariant

Two modes must be functionally identical:

- mechanics system disabled;
- mechanics system enabled but forced to PASS forever.

Both must reproduce the frozen/zero-step Wisp strategic path at tick 8, including observation, first-90 logits, legal mask, selected index, temporal schedule and controller output.

Automated tests should compare long generated action traces, not just one isolated row.

## Mechanics actor

Initial M08 mechanics policy is intentionally narrow:

- trainable parameters are separate from the strategic actor;
- strategic actor parameters have `requires_grad=False` and are excluded from every optimizer;
- output dimension 69 = PASS + 68 appended actions;
- initialize strongly toward PASS but not so strongly that appended exploration is effectively zero;
- calibrate initial override probability from natural states rather than choosing a magic bias blindly;
- log PASS probability, sampled PASS rate, deterministic override rate and per-action usage.

The critic may be wholly new and trainable.

## Eligibility

A broad eligibility mask is allowed to avoid nonsense high-frequency overrides. Eligible state families may include generic physical concepts such as:

- airborne or recent jump;
- wall/ceiling contact or proximity;
- active recovery / awkward landing;
- available dodge/flip resource;
- close-ball interaction where a four-tick control can affect contact;
- other evidence-backed mechanics contexts.

Do not encode named sequences such as 'musty now' or exact scripted coordinates. The long-term goal remains emergent mechanics that are useful because they improve 1v1 outcomes.

Always keep PASS legal. When no mechanics context is eligible, force PASS.

## Training opponent modes

Support at least:

- frozen-Wisp opponent anchor;
- dual-rate self-play using the current mechanics head;
- historical mechanics-head checkpoints later if needed.

M08 may use a simple mixture of frozen-Wisp and self-play. Do not let opponent-pool engineering dominate this milestone.

## Export/deployment seam

A portable dual-rate candidate needs:

- immutable strategic Wisp reference/hash;
- mechanics actor checkpoint/export;
- exact 158-row action table hash;
- observation-contract version;
- strategic scheduler version;
- mechanics scheduler version;
- eligibility/gate configuration.

The production runtime must not switch to this candidate unless explicitly opted in. M08 does not authorize promotion.

## Future extension

Only after the dual-rate architecture transfers cleanly may a later milestone consider constrained strategic fine-tuning. If that happens, use explicit teacher-preservation constraints such as KL/logit/top-1 regularization and report legacy drift independently. Do not reopen the strategic trunk during M08.