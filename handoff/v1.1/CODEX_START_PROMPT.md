# Prompt to give Codex

You are implementing the first version of a high-end offline Rocket League bot for RLBot v5.

Do not respond with another design plan. Work directly in the repository and complete as much implementation and verification as the environment permits.

## Source of truth

The canonical project repository is:

`https://github.com/Noxoni/Rival`

Work directly in the checked-out `Noxoni/Rival` repository. If this package was provided separately, clone `Noxoni/Rival` first and copy/use the handoff under `handoff/v1.1/`.

Do not initialize or use an unrelated repository.

The installed RLBot v5 bot directory is:

`C:\Users\patri\AppData\Local\RLBot5\bots`

Treat that installed directory as read-only.

The only installed bots we care about as primary references are:

- Wisp v2-75B
- Nexto

Locate them by inspecting bot TOML/config metadata and content rather than assuming folder names.

Known identifiers/repositories:

- Wisp v2-75B agent id: `eastvillage/wisp/v2-75B`
- Wisp source reference: `https://github.com/NicEastvillage/RLBot-Wisp-v2-py`
- Nexto v5 port: `https://github.com/VirxEC/NectoFamily`
- Original Nexto/Necto repo: `https://github.com/Rolv-Arild/Necto`

If the installed BotPack contains packaged/binary-only forms rather than usable source, preserve those installed files as the local baseline reference and obtain the matching public source from the upstream repositories above. Record exactly which source was used and its Git commit SHA.

## Required workflow

1. Confirm the working repository remote is `https://github.com/Noxoni/Rival` (or its authenticated equivalent).
2. Read every file under `handoff/v1.1/` before changing implementation code.
3. Create/maintain repository `CHANGELOG.md`.
4. Run `handoff\v1.1\scripts\snapshot_reference_bots.ps1` or perform equivalent discovery if the installed layout requires adjustment.
5. Preserve the local reference snapshot outside tracked source by default; commit its SHA-256 manifest and provenance, not the full third-party trees.
6. Verify applicable licenses before committing any Wisp/Nexto-derived source or model artifact.
7. Build Rival in the repository implementation tree, separate from local reference snapshots.
8. Commit each stable milestone. Do not silently overwrite a working baseline.
9. Push stable commits to `Noxoni/Rival` so progress does not live only in Codex's workspace.

## Working bot identity

Use:

- RLBot display name: `Rival Dev`
- agent id: `noxoni/rival/dev-v1`

Do not reuse Wisp's or Nexto's agent id.

## Architecture for this milestone

Use Wisp v2-75B as the runnable baseline.

Do not merge neural-network weights with Nexto.
Do not train a new model in this milestone.
Do not add crude gameplay overrides such as `if boost < 30: never aerial`.

The development bot must initially preserve Wisp's selected actions while exposing enough internal information to inspect why those actions were chosen.

Create a clean seam around policy inference so later versions can alter policy scoring, masking, strategy, or training without rewriting the RLBot transport/runtime.

At minimum expose a structured decision object containing:

- selected discrete action index
- selected controller action
- raw policy logits when available
- masked policy logits
- legal action mask
- top-N candidate actions and scores/probabilities
- model/policy tick number
- decision timestamp/game time

If Wisp's current wrapper does not expose logits, minimally refactor the wrapper to return them without changing the model's selected argmax behavior.

## Telemetry

Add structured, toggleable telemetry that can log at decision ticks without overwhelming normal play.

At minimum capture:

### Player/resource state
- self boost
- opponent boost if available
- self position/velocity
- opponent position/velocity
- supersonic state if useful
- wheel/contact/airborne state
- score differential
- game clock / time remaining when available

### Ball/possession state
- ball position/velocity
- self distance to ball
- opponent distance to ball
- rough self ETA to ball
- rough opponent ETA to ball
- relative closing speeds where practical
- possession/control proxy or enough raw fields to derive one later

### Boost/map state
- active large/small pads if exposed by the current Wisp state representation
- nearby reachable boost opportunities if Wisp already derives them
- do not invent expensive path planning yet

### Policy state
- selected action
- top candidate actions
- confidence/margin between top actions
- action mask
- previous action if used by Wisp
- tick-skip/action-delay state

Telemetry format should be machine-readable JSONL or similarly easy to analyze later.

## Initial tactical metrics

Create a small pure-code tactical metrics module, but do not use it to override policy decisions yet.

Compute only metrics that can be justified from currently available state, such as:

- ball height
- ball distance
- self/opponent ETA estimates
- current boost
- score differential
- challenge closing velocity
- whether self is airborne
- whether an action uses boost
- whether selected action appears to initiate/continue an aerial, if the discrete action mapping makes this derivable

Keep these measurements separate from policy control.

## Verification

The milestone is successful only if:

1. Original installed Wisp and Nexto remain untouched.
2. Local reference snapshots and committed hash/provenance records exist.
3. The new bot has a unique RLBot v5 identity.
4. The new bot loads the same Wisp model artifacts intended for the baseline.
5. The new bot can reach policy inference without shape/runtime errors.
6. With strategic overrides disabled, chosen actions are unchanged from the Wisp baseline for identical synthetic/recorded observations where comparison is possible.
7. Telemetry can be enabled and disabled.
8. Basic tests cover the new decision/telemetry seam.
9. A clear run/test document exists for launching `Rival Dev` against Nexto and Wisp in RLBot v5.

If Rocket League or RLBot cannot be launched from the current Codex environment, do not fake gameplay verification. Complete static/unit/smoke verification and document the exact remaining manual launch step.

## Do not do yet

- Do not start long reinforcement-learning runs.
- Do not try to combine Wisp and Nexto model weights.
- Do not replace Wisp behavior with a hand-coded state machine.
- Do not optimize for RLBot public BotPack submission restrictions.
- Do not modify the user's installed BotPack.
- Do not claim behavioral improvement without match evidence.

## End-of-run deliverable

At the end of this Codex run, report:

- Git commit SHA(s) pushed to `Noxoni/Rival`
- exact reference source paths found
- exact upstream commit SHAs used, if any
- files added/changed
- tests executed and results
- whether the dev bot is runnable
- whether live RLBot/Rocket League verification was actually performed
- any blockers
- the next smallest implementation step

Prioritize working code over prose.

## Repository persistence requirement

Do not finish with meaningful work existing only locally.

Before the end-of-run report:

- commit completed/stable work
- push the commit(s) to `Noxoni/Rival`
- verify the remote branch contains the reported commit SHA(s)

If push fails because of credentials, branch protection, or connectivity, report that as a blocker with the exact local commit SHA. Do not claim the work is safely persisted remotely unless verified.
