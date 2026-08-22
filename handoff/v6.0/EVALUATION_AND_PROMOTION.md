# Milestone 06 Evaluation and Promotion

## 1. Do not judge by training reward alone

Every serious checkpoint is evaluated against frozen external references. Training reward, mechanic counters and self-play Elo-like measures are supporting evidence only.

## 2. Headless frozen-Wisp gate

At least every 5M agent-steps, run a deterministic headless evaluation against the frozen reconstructed Wisp teacher over a sufficiently large sample of natural 1v1 episodes with balanced sides.

Suggested minimum: 100 games; prefer 200 when inexpensive.

Record:

- wins/losses/ties if applicable;
- goals for/against and goal differential;
- touches / possession proxy;
- boost use and boost remaining after aerials;
- appended action frequency;
- recovery metrics;
- reset/mechanics metrics;
- reward-component values for diagnostics only.

The evaluation policy must not use stochastic exploration. Record any appended-action training prior applied during evaluation; mature checkpoints should ultimately evaluate without artificial training-only suppression.

## 3. Stage-boundary RLBot benchmark

At major boundaries (roughly 20M, 50M, and final/promotion candidate), export the student and test it in actual RLBot/Rocket League at ~5x effective game speed.

Use the v4.1 deployment benchmark composition as the reference shape:

- installed Nexto;
- installed Wisp v2-75B;
- balanced blue/orange sides;
- full five-minute Soccar;
- normal kickoff countdowns;
- no replay saving;
- debug rendering/performance overlay off.

For ordinary stage checks, 8 total games (4 Nexto / 4 Wisp, balanced sides) is sufficient context. Do not overinterpret a small sample.

## 4. Final promotion battery

A checkpoint may replace the frozen-Wisp deployment only after a larger final battery.

Minimum final battery:

- 16 full five-minute RLBot matches;
- 8 against Nexto and 8 against Wisp v2-75B;
- balanced sides;
- standard natural play at validated accelerated speed;
- no hand-selected favorable starts.

Promotion requires all of the following:

1. no numerical/runtime correctness regression;
2. no obvious catastrophic gameplay regression or exploit behavior;
3. positive or at least non-inferior overall mixed-opponent results relative to the v4.1 frozen-Wisp deployment benchmark;
4. evidence that newly available mechanics/recovery behavior is being used naturally and productively rather than as random action noise;
5. no serious regression in recovery/boost/resource metrics that offsets offensive gains;
6. production inference/export parity passes.

Do not hard-code promotion solely to one 16-game win count. Consider goal differential, opponent split, touch/possession, concessions after possession loss, recovery, and the headless Wisp sample together.

A checkpoint that looks promising but ambiguous remains a candidate and continues training; it does not replace production.

## 5. Historical benchmark to preserve

v4.1 frozen-Wisp natural benchmark:

- 8 games total;
- 4 Nexto / 4 Wisp;
- balanced sides;
- record 4–4;
- goals 35–30;
- goal differential +5;
- favorable ETA share ~64.07%;
- Rival share of recorded touches ~75.74%.

These are contextual baseline metrics, not statistical guarantees. Use them as a fixed deployment reference, not as a target to game.

## 6. Mechanics-success interpretation

Examples of convincing mechanics progress include:

- appended actions become nonzero and concentrated in plausible contexts rather than random everywhere;
- airborne flips reduce boost spend while preserving/improving useful touches;
- resets are followed by retained possession, shots, goals or defender outplays more often than immediate loss;
- wall/wavedash-like actions shorten recovery time or increase usable speed;
- fewer concessions occur immediately after failed aerial/offensive plays;
- ceiling/wall possession produces more options and fewer possession dumps.

Mechanic frequency alone is not success.

## 7. Checkpoint ranking

Keep a small leaderboard of healthy checkpoints using separate metrics rather than one opaque score. At minimum track:

- headless Wisp win rate and goal differential;
- RLBot Wisp/Nexto results where available;
- goal differential;
- possession/touch share;
- recovery-concession rate;
- boost/aerial efficiency;
- appended-action share;
- productive reset rate.

Never delete the Wisp bootstrap checkpoint or the best previously verified healthy checkpoint merely because a newer model has more training steps.
