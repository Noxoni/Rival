# Paused Local State — Resume Contract

This file records the exact state reported when Codex was paused so the resume operation can be checked against it.

## Reported paused state

At pause time:

- no tests were running;
- no controlled probes were running;
- no natural matches were running;
- no commits or pushes were running;
- no Milestone 03 natural-match budget had been consumed;
- the local view of `origin/main` was `5cd05a9cb1e88df65d5ad417ca6f1e8242356be7`;
- only the following Milestone 03 implementation work was uncommitted:
  - `bot/strategy/__init__.py`
  - `bot/strategy/challenge_commitment.py`;
- the user's untracked `bot.7z` was untouched;
- no Milestone 03 completion claim had been made.

## Resume invariants

Before any implementation continues, Codex must verify and preserve this state as far as the current working tree allows.

1. Run and record:
   - `git status --short`;
   - `git rev-parse HEAD`;
   - `git rev-parse origin/main` before fetch;
   - `git diff -- bot/strategy/__init__.py bot/strategy/challenge_commitment.py`;
   - `git ls-files --others --exclude-standard`.
2. Do not run `git reset --hard`, `git clean`, or any destructive checkout.
3. Do not touch or add `bot.7z`.
4. Preserve both strategy files even if they are untracked.
5. Treat any unexpected additional local modification as user work and preserve it rather than deleting it.

## Required safety snapshot

Before fetching/synchronizing remote handoff commits, create a local recoverable snapshot of the paused strategy work.

Preferred procedure:

```powershell
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$backup = ".resume_backup\m03-v3.3-$stamp"
New-Item -ItemType Directory -Force $backup | Out-Null
Copy-Item bot\strategy\__init__.py $backup\__init__.py -ErrorAction SilentlyContinue
Copy-Item bot\strategy\challenge_commitment.py $backup\challenge_commitment.py -ErrorAction SilentlyContinue
git diff --binary -- bot/strategy/__init__.py bot/strategy/challenge_commitment.py | Set-Content "$backup\tracked.patch"
git status --short | Set-Content "$backup\status.txt"
```

Then protect the working tree using a named stash that includes untracked files:

```text
git stash push -u -m "rival-m03-paused-before-v3.3-resume"
```

Do not rely on the stash as the only copy; keep the raw backup above until Milestone 03 is complete.

The backup directory is machine-local recovery material and must not be committed. Add `.resume_backup/` to `.git/info/exclude` or the project `.gitignore` if needed without disturbing unrelated ignore rules.

## Synchronize documentation-only remote progress

After the local work is protected:

```text
git fetch origin
git pull --ff-only origin main
```

The remote is expected to contain the v3.1/v3.2/v3.3 handoff/documentation overlays after the paused `5cd05a9...` state. Inspect the fetched commits before restoring local work. Do not assume they are documentation-only if the actual diff says otherwise.

If implementation files under `bot/strategy/` were changed remotely, stop the automatic restore path and reconcile carefully; do not overwrite either side.

If the fetched changes are compatible handoff/docs only, restore the paused work:

```text
git stash pop
```

Then compare the restored files against the raw backup and confirm the paused contents survived exactly unless a deliberate conflict resolution was required.

## Checkpoint after safe synchronization

Once the paused work is restored on top of the current handoff commits, inspect it before editing further.

Do not assume the partial implementation is correct just because it was preserved. Review it against:

- `handoff/v3.0/CHALLENGE_CALIBRATION_DESIGN.md`;
- `handoff/v3.0/MILESTONE_03_SPEC.md`;
- `handoff/v3.1/EXECUTION_ACCELERATION.md`;
- `handoff/v3.2/CONFIG_OPTIMIZATIONS.md`.

After that review, make the first stable implementation commit when the partial strategy layer is coherent and testable. Do not push a knowingly broken WIP state to `main` merely to create a checkpoint.

## Natural-match budget

The Milestone 03 natural-match budget is still fully unused at resume time.

Controlled tests, probes, speed-integrity checks, and launcher/concurrency checks do not consume the six natural-match acceptance budget unless they are actual natural Rival-vs-Nexto/Wisp validation games counted by the Milestone 03 experiment.
