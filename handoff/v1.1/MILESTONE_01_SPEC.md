# Milestone 01 — Baseline Extraction + Instrumented Wisp

## Goal

Produce Rival as a third, separately identifiable RLBot v5 bot that behaves as the Wisp v2-75B baseline while exposing the policy decision process and enough state telemetry to begin targeted improvements.

This is the first actual implementation milestone.

## Required repository layout

Codex may adjust filenames to match the real Wisp source tree, but preserve these logical boundaries:

```text
/
├─ .local_reference_sources/      # gitignored; machine-local
│  └─ v1/
│     ├─ wisp/
│     ├─ nexto/
│     └─ MANIFEST.json
├─ reference_manifests/
│  └─ v1/
│     ├─ MANIFEST.json
│     └─ SOURCE_PROVENANCE.md
├─ bot/
│  ├─ <Wisp-derived RLBot v5 runtime>
│  ├─ policy/
│  │  ├─ decision.py
│  │  └─ inspector.py
│  ├─ analysis/
│  │  └─ tactical_metrics.py
│  └─ telemetry/
│     └─ decision_logger.py
├─ tests/
├─ scripts/
├─ docs/
│  ├─ RUN_LOCAL.md
│  └─ SOURCE_PROVENANCE.md
├─ CHANGELOG.md
└─ README.md
```

Do not force this exact tree if doing so would unnecessarily fight Wisp's import layout. The separation of responsibilities matters more than the exact folder spelling.

## Step A — discover and snapshot reference bots

Search recursively under:

```text
C:\Users\patri\AppData\Local\RLBot5\bots
```

Identify Wisp using, in descending preference:

1. TOML containing `eastvillage/wisp/v2-75B`
2. TOML/name containing `Wisp v2-75B`
3. known Wisp model/source signatures

Identify Nexto using:

1. TOML/metadata identifying `Nexto`
2. agent id containing `nexto`
3. `nexto-model.pt` plus compatible bot source/config

Copy each discovered bot tree to the immutable **local** reference snapshot.

Exclude obvious runtime caches only (`__pycache__`, `.pytest_cache`, etc.). Do not omit model artifacts, configs, or source needed to reproduce the installed bot.

Create hashes and copy the manifest/provenance into the tracked `reference_manifests/v1/` area.

Do not automatically commit the complete third-party reference trees to the public Rival repository. Verify licenses/redistribution terms before committing any third-party source or model artifact.

If only packaged binaries are present, snapshot them locally, then fetch source from upstream and record both installed-artifact provenance and upstream-source provenance.

## Step B — create development baseline

Use the Wisp v2 Python source as the implementation base.

Do not edit the reference copy.

Give the new bot:

```text
name = Rival Dev
agent_id = noxoni/rival/dev-v1
```

Preserve the Wisp model artifacts and inference behavior.

## Step C — policy-decision seam

Refactor inference so a call can return a structured result equivalent to:

```python
PolicyDecision(
    action_index=...,
    controller_action=...,
    raw_logits=...,
    masked_logits=...,
    legal_mask=...,
    top_actions=...,
    confidence=...,
    margin=...,
    tick=...,
    game_time=...,
)
```

Exact types are up to Codex.

Requirements:

- Avoid unnecessary tensor copies in the hot path.
- Normal play can disable expensive debug materialization.
- The final selected action must remain Wisp-equivalent when no override is enabled.
- Keep model inference and strategic analysis separable.

## Step D — tactical metrics, measurement only

Implement measurements without changing control.

Suggested fields:

```text
self_boost
opponent_boost
ball_height
distance_self_ball
distance_opponent_ball
eta_self_ball
eta_opponent_ball
challenge_closing_speed
self_airborne
opponent_airborne
selected_action_uses_boost
selected_action_uses_jump
selected_action_aerial_like   # only if action mapping supports a defensible definition
score_diff
seconds_remaining
```

ETA may initially use the same or closely related heuristic already used in Wisp if one exists. Clearly label approximate values.

## Step E — telemetry

Prefer JSONL.

A record should be usable later to answer:

- What did the policy see?
- What action did it choose?
- What were the alternatives?
- How much boost did it have?
- Was the opponent actually committing?
- Did the policy decision occur during a likely aerial?
- Did the network strongly prefer the chosen action or barely win the argmax?

Do not log huge tensors or all observations by default. Add a verbose/debug mode if full observation dumps are useful.

## Step F — tests

At minimum:

- decision wrapper selects same argmax as original baseline logic
- legal mask is honored
- top-N ranking is internally consistent
- telemetry serialization works
- telemetry-off mode has negligible/no file output
- tactical metrics handle missing opponent safely
- configuration identifies the new bot uniquely

Where the real Wisp model is loadable in tests, add a smoke inference test. If not, mock the model and separately document live-model verification.

## Acceptance criteria

Milestone 01 is done when there is a runnable dev bot and we can inspect what Wisp wants to do **without yet changing what it does**.

The next milestone will use this instrumentation to begin correcting measurable strategic defects.

## Repository acceptance

Milestone 01 is not fully persisted until its stable commit has been pushed to `Noxoni/Rival`.

A local-only commit is acceptable only when a real push blocker exists and is documented.
