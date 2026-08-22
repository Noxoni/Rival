# Wisp Teacher Bootstrap

## Goal

Do not make Rival relearn basic Rocket League from random initialization if the current Wisp v2-75B policy can provide the starting competence.

The teacher artifacts are frozen inputs. Rival becomes a separately trainable student.

---

## Bootstrap order

Codex must try these in order and stop spending time on a method once it is clearly impractical.

### Path A — direct trainable reconstruction / head expansion

Inspect the frozen TorchScript `SHARED_HEAD.lt` and `POLICY.lt` modules, their graphs, parameter tensors, and layer shapes.

If the architecture can be reconstructed faithfully into ordinary trainable PyTorch modules without guesswork:

1. reproduce the Wisp actor/shared-head computation;
2. load the existing weights;
3. verify logits against the frozen modules over a large randomized observation batch;
4. expand the final actor output from 90 actions to `RivalExpandedActionV1`;
5. copy the original 90 output rows exactly;
6. initialize new action rows conservatively so the initial student mostly behaves like Wisp;
7. save a trainable student checkpoint.

Required gate for this path: numerical agreement with the frozen Wisp output must be demonstrated, not assumed.

If TorchScript structure is opaque or reconstruction becomes reverse-engineering work, stop and use Path B.

### Path B — behavior distillation

Use the frozen Wisp model as a teacher in headless RocketSim.

Generate a dataset from **natural 1v1 trajectories**, preferably teacher self-play or teacher-controlled play, not exact hand-authored test states.

Each record should contain at minimum:

- student observation;
- teacher observation if different;
- teacher masked logits or probability distribution when available;
- teacher selected Wisp action index;
- resulting controller action;
- legal-action mask/state needed to reconstruct the target;
- episode/time identity sufficient for validation splits.

For a 4-tick student cadence with an 8-tick Wisp teacher:

- run teacher inference at its native 8-tick cadence;
- use the chosen Wisp action as the target for both 4-tick student decisions within that teacher window;
- record the intermediate 4-tick state as a valid student observation with the same target action.

This teaches the student to reproduce the teacher while retaining the option to learn mid-window corrections later.

### Supervised objective

Prefer distribution matching when reliable teacher logits are available:

- cross-entropy / KL on the original Wisp action subset;
- optional small penalty against prematurely selecting appended actions during bootstrap.

If only selected actions are robustly available, ordinary action classification is acceptable.

The student output head covers the full expanded action space. During teacher imitation, target probability for newly appended actions should be near zero rather than pretending the teacher had opinions about actions it never possessed.

---

## Student architecture

Use an actor architecture compatible with the PPO training path so bootstrap weights can seed PPO directly.

Do not build a one-off imitation network that cannot be loaded into the PPO actor without a tested conversion.

Reasonable initial policy size is in the same general class as RLGym PPO examples (hundreds of hidden units per layer), but Codex should inspect Wisp's effective capacity and measure inference/training throughput before locking a size.

The architecture must be recorded in a committed config.

---

## Bootstrap dataset size

Milestone 05 needs a bounded proof, not a giant offline corpus.

Suggested sequence:

1. generate a small dataset sufficient to validate the full pipeline;
2. train until held-out agreement clearly exceeds random and loss is decreasing;
3. if the pipeline is healthy, scale the dataset only enough to demonstrate useful teacher retention;
4. save a resumable checkpoint and report exact counts/timing.

Do not spend the entire Milestone 05 run chasing a perfect imitation score.

Useful smoke gates:

- train/validation split by trajectory/session, not random adjacent frames only;
- held-out top-1 teacher action agreement materially high (target at least ~80% for the bounded smoke if feasible);
- top-k agreement and cross-entropy/KL reported when logits exist;
- no NaNs;
- expanded actions remain low-frequency before PPO unless deliberately explored;
- checkpoint reload reproduces inference.

The exact acceptance threshold may be adjusted if evidence shows Wisp is highly multimodal/noisy, but the reason must be documented.

---

## Teacher-controlled opponent use

Once a trainable student exists, Wisp can also serve as a benchmark/opponent.

Do not require a complex historical-opponent league for Milestone 05. Initial PPO may use student self-play after Wisp bootstrap. Because both sides begin from the same Wisp-derived competence, this avoids starting self-play from random driving.

Later milestones can add:

- frozen Wisp opponent sampling;
- historical Rival checkpoints;
- skill/Elo-weighted opponent pools;
- replay-derived human states.

---

## Integrity

Before and after bootstrap, verify the original model hashes:

- `POLICY.lt`: `1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7`
- `SHARED_HEAD.lt`: `3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42`

Never overwrite the teacher files.

Store generated datasets/checkpoints outside normal Git tracking unless intentionally small. Commit manifests containing path convention, format version, record count, hashes, configs, and reproduction command.
