# Codex Start Prompt — Rival Milestone 03 Resume v3.3

You are resuming the paused Milestone 03 implementation of **Rival**, a high-end offline Rocket League 1v1 bot for RLBot v5.

Do not start over. Do not discard the paused local strategy work. Do not answer with another high-level plan. Safely synchronize the repository, preserve the paused files, then continue the v3.0 challenge-commitment experiment using the v3.1/v3.2 accelerated test configuration.

## Canonical repository

`https://github.com/Noxoni/Rival`

## Exact paused state reported by the previous Codex run

At pause time:

- no tests, probes, matches, commits, or pushes were running;
- no Milestone 03 natural-match budget had been consumed;
- the local view of `origin/main` was `5cd05a9cb1e88df65d5ad417ca6f1e8242356be7`;
- the only partial Milestone 03 code was uncommitted:
  - `bot/strategy/__init__.py`
  - `bot/strategy/challenge_commitment.py`;
- the user's untracked `bot.7z` was untouched;
- no Milestone 03 completion was claimed.

Read `handoff/v3.3/RESUME_STATE.md` before doing anything else.

---

# Phase 0 — preserve paused work before synchronization

1. Record current repository state with `git status --short`, current HEAD, pre-fetch `origin/main`, the two strategy-file diffs/content state, and untracked files.
2. Do **not** run `git reset --hard`, `git clean`, destructive checkout, or any command that could discard user work.
3. Do not touch, add, move, archive, inspect destructively, or commit `bot.7z`.
4. Create the raw local backup described in `RESUME_STATE.md` for both strategy files plus status/patch metadata.
5. Create the named stash including untracked files: `rival-m03-paused-before-v3.3-resume`.
6. Fetch `origin` and inspect commits that arrived after the paused local state.
7. Fast-forward the local main branch to `origin/main` with `git pull --ff-only origin main` only if the branch relationship is safe.
8. Confirm the newer remote commits are compatible handoff/documentation/config overlays. If any fetched commit unexpectedly modifies `bot/strategy/` or other overlapping partial implementation code, reconcile instead of overwriting.
9. Restore the named stash and verify the two paused strategy files match the pre-sync raw backup.
10. Preserve any unexpected additional local modification as user work.

Do not proceed until the paused work is demonstrably recoverable.

---

# Phase 1 — load the complete current instructions

Read, in this order:

1. every file under `handoff/v3.3/`;
2. every file under `handoff/v3.2/`;
3. every file under `handoff/v3.1/`;
4. every file under `handoff/v3.0/`;
5. `docs/MILESTONE_02_RESULTS.md`;
6. `evidence/results/v2/candidate_events.md`;
7. the existing curated challenge fixture(s).

v3.3 only changes resume mechanics. v3.2 changes test/config execution. v3.1 changes acceleration/concurrency. v3.0 remains the authoritative gameplay design and acceptance experiment.

Review the preserved partial implementation rather than blindly continuing it. Keep useful work, fix it if it violates the v3.0 design, and avoid unnecessary rewrites.

---

# Phase 2 — continue Milestone 03 implementation

Continue only the challenge-commitment calibration experiment defined in v3.0.

Do not work on boost greed, resource-stressed aerials, model retraining, observation redesign, tick-skip changes, unrelated Wisp cleanup, or other gameplay behavior in this milestone.

Maintain the required modes:

- `off` — exact pre-v3 action-selection behavior;
- `observe` — calculate/log hypothetical intervention but return baseline Wisp action;
- `intervene` — apply only the accepted challenge calibration.

Treatment must remain a conservative re-ranking among Wisp's already legal discrete actions. Do not synthesize a separate hand-coded controller policy.

---

# Phase 3 — implement accelerated automated test execution before expensive live validation

Apply the v3.2 configuration optimization contract to the evidence/match runner.

For automated natural validation, preserve the actual **five-minute match length** and normal Soccar rules, but target **5.0x Rocket League simulation speed**.

Required automated-match defaults where supported:

- `skip_replays = true`;
- `auto_save_replay = false`;
- debug rendering = `AlwaysOff`;
- performance monitor = `NeverShow`;
- `auto_start_agents = true`;
- `wait_for_agents = true`;
- `existing_match_behavior = Restart`;
- normal kickoff countdowns for natural matches;
- normal boost, gravity, demolish, physics, scoring, map, and five-minute match rules.

Do not substitute Rocket League's `TimeWarp` mutator for accelerated simulation. Use the direct game-speed/state-setting mechanism documented in v3.1/v3.2.

## 5x integrity gate

Before using 5x as evidence-producing natural validation:

1. run a bounded frozen-baseline comparison at 1x and 5x;
2. verify both bots remain responsive;
3. verify policy/telemetry decision rates are not catastrophically degraded;
4. record RLBot missed-packet/queue warnings;
5. compare representative action/event distributions and obvious gameplay/runtime anomalies;
6. document whether 5x is accepted for the current machine.

Do not demand statistical identity between 1x and 5x, but do not silently use 5x if it clearly corrupts behavior or packet delivery.

If 5x fails, test the highest lower speed that remains useful (for example 4x, 3x, or 2x), record the reason, and use the fastest validated speed. Do not revert automatically to watching every match at 1x unless necessary.

---

# Phase 4 — bounded concurrency experiment

Attempt true match parallelism once, after the single-lane accelerated runner is stable.

A valid parallel lane requires its own isolated RLBotServer endpoint/port and Rocket League process. Two Python runners attaching to one existing server are **not** parallel matches.

Try at most **two lanes** initially.

Required isolation includes:

- independent RLBotServer endpoints/ports;
- independent Rocket League match processes/instances;
- independent session IDs and telemetry paths;
- no shared writable replay or transient output path that can collide;
- no cross-lane process cleanup;
- no accidentally shared `MatchManager` server discovery.

Run one bounded smoke comparison using two simple test lanes.

If Rocket League/Steam/Epic/RLBot prevents multiple clean simultaneous instances, or if telemetry/packet behavior becomes unstable, mark parallelism unsupported on this machine and **immediately fall back to sequential accelerated matches**. Do not burn substantial time engineering launcher bypasses.

If two lanes are stable, use at most two concurrent live validation games for this milestone unless measured headroom clearly supports more and there is an evidence need. Reliability matters more than maximizing instance count.

---

# Phase 5 — original Milestone 03 experiment order

Follow the v3.0 experiment order:

1. static/unit verification;
2. offline fixtures/evidence;
3. paired controlled A/B fake-vs-true challenge probes;
4. natural-match validation only if controlled gates pass.

The natural-match acceptance budget is still **six games total maximum** because none were used before the pause:

- at most 3 vs installed Nexto;
- at most 3 vs installed Wisp v2-75B.

Those games should remain full five-minute matches, executed at the fastest validated accelerated speed. Run them in two isolated concurrent lanes only if the concurrency smoke gate passed; otherwise run sequentially.

Do not launch extra matches because a result is inconvenient.

---

# Required verification and output

All v3.0 verification requirements still apply, plus:

- prove the paused two-file work survived synchronization;
- document pre-resume local SHA and fetched remote SHA;
- document the selected live-test speed and its integrity-gate evidence;
- document whether two-lane concurrency passed or failed;
- record wall-clock duration for accelerated validation so we can quantify the improvement;
- confirm automated replay saving is disabled and goal replays remain skipped;
- confirm debug rendering/performance overlay are disabled in automated runs;
- verify no natural-match budget was consumed before the actual Stage 4 acceptance run.

Commit and push stable work/results to `origin/main` as required by the original v3.0 handoff. Do not push a knowingly broken checkpoint merely to preserve partial work; the local raw backup and stash provide recovery until the first coherent implementation commit exists.

At the end, return:

- commit SHA(s);
- exact resume/sync result;
- confirmation the paused files were preserved;
- accepted/rejected challenge-calibration result;
- accelerated game speed used;
- 1x-vs-accelerated integrity findings;
- parallel-lane supported/unsupported result;
- actual natural matches run and wall-clock duration;
- all original v3.0 experiment metrics and test results;
- final `origin/main` SHA;
- next smallest evidence-backed target.

Do the work in this run. Prioritize safe preservation, measured behavior, and fast reliable execution.