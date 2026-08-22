# Rival — Codex Handoff v1.1

This package is the execution handoff for the first implementation of **Rival**, a high-end offline RLBot v5 1v1 opponent.

## Canonical repository

```text
https://github.com/Noxoni/Rival
```

`Noxoni/Rival` is the source of truth for implementation work and project history.

Codex should work directly in that repository and push stable progress there. Do not create an unrelated project repository.

## Local RLBot source location

The user's installed RLBot v5 bot directory is:

```text
C:\Users\patri\AppData\Local\RLBot5\bots
```

Treat this installed BotPack directory as **read-only**.

The primary reference bots are:

- **Wisp v2-75B** — initial v5 technical/runtime baseline.
- **Nexto** — major possession, ground-control, flick, and challenge-behavior reference.

Do not waste time analyzing the rest of the installed BotPack unless a specific implementation question makes another bot useful.

## Reference policy

The installed Wisp and Nexto trees are local reference inputs. Snapshot/hash them locally, but do **not** automatically commit their complete trees or model files into this public repository.

Commit:

- source provenance
- exact local paths
- SHA-256 manifests
- upstream URLs
- upstream Git SHAs
- license/attribution notes
- any third-party components actually incorporated into Rival, when redistribution is allowed

If Rival requires a Wisp/Nexto-derived model artifact to run, Codex must verify the artifact's applicable license/redistribution terms before committing it.

## User-observed behavior targets

### Nexto — preserve/learn

- Strong possession acquisition.
- Fast conversion from possession into threatening flicks.
- Strong ground offense.

### Nexto — improve

- Susceptible to fake challenges.
- Can release/flick early when an opponent presents a convincing challenge then aborts.
- Does not prioritize boost enough.

### Wisp v2-75B — preserve/learn

- Strong overall play.
- Aggressive boost denial / opponent starvation.
- Good clock-killing tendencies while ahead.
- Advanced aerial and air-dribble capability.

### Wisp v2-75B — improve

- Can overvalue map boost.
- Often takes possession back toward its own half for very long aerial plays.
- Can attempt/continue resource-expensive aerial offense with inadequate boost.
- Resource feasibility is weaker than its mechanical execution.

These are hypotheses to instrument and test, not instructions for crude hard-coded thresholds.

## Immediate implementation objective

1. Discover and hash the local installed Wisp and Nexto.
2. Establish Rival's runnable v5 baseline from Wisp's Python implementation.
3. Give Rival a unique RLBot identity.
4. Preserve baseline Wisp action selection initially.
5. Expose logits, legal masks, chosen action, alternatives, and confidence/margin.
6. Add tactical/resource telemetry.
7. Verify baseline parity before changing strategy.
8. Use original Wisp and Nexto as live comparison opponents.

Codex should begin with `CODEX_START_PROMPT.md`.
