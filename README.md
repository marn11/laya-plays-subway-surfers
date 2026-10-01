# laya-plays-subway-surfers

Fine-tuning [Laya](https://huggingface.co/convaiinnovations/laya) — the same decision model used in [laya-model-router](https://github.com/marn11/laya-model-router) — to play a Subway-Surfers-style endless runner, by picking one of a handful of actions (left/right/jump/duck/none) from a text description of the game state each tick.

## Why a clone, not the real mobile game

Laya is a text-only decision model (no vision component). Playing the real mobile game would require solving real-time computer vision first — turning screen pixels into a reliable symbolic state every frame — which is a separate, much harder problem that has nothing to do with Laya itself. Running a JS clone locally sidesteps that entirely: the game's own in-memory state (player lane, obstacle positions/types) is read directly, no pixels involved.

## The game

`game.html` is a fork of [justinmimurray-coder/Subway-Surfers](https://github.com/justinmimurray-coder/Subway-Surfers) (Three.js, single self-contained file, no license specified in the original — used here for personal/non-commercial experimentation with attribution). Chosen specifically because its state (`currentLane`, `activeObstacles`, `isPlaying`, ...) and action functions (`moveLane()`, `jump()`, `slide()`) are plain top-level variables/functions in one `<script>` block — nothing buried in a framework abstraction.

Added at the bottom of that same `<script>` block, so it has direct access to the game's own scope:

- `window.__getState()` — returns `{isPlaying, currentLane, isGrounded, isSliding, speed, score, distanceTraveled, obstaclesAhead}`, where `obstaclesAhead` is the nearest few `{type, lane, isHigh, distance}` ahead of the player (`distance` > 0 means not yet reached).
- `window.__action(name)` — `"left" | "right" | "jump" | "duck" | "none"`.
- `window.__restart()` — clicks the game's own start/retry button.

## The oracle

`oracle.js` is **not a model** — it's a small, fixed set of reflex rules (if an obstacle is in my lane within 15 units: dodge trains sideways, duck high barriers, jump low ones). It's the *teacher*: good enough to survive a while, cheap to run thousands of times, and every tick it plays produces one labeled training example — `(game state, the action the oracle took)`. Laya is the *student* that gets fine-tuned to imitate those examples. Same idea as generating labeled difficulty examples via Codex earlier, just for a sequential control task instead of a one-shot classification.

## Running it yourself

```bash
npm install
npx playwright install chromium   # one-time browser download

node play_oracle.js                                        # watch one run (headed, a window opens)
node play_oracle.js --headless --max-ticks 50000 \
  --record data/oracle_dataset.jsonl                       # real data-collection run, no window
```

Each episode prints `ticks`, `score`, and `distance`. `--record <path>` appends one JSON line per tick (`{"state": {...}, "action": "..."}`) to that file — this is the raw material for fine-tuning. `--max-ticks` is a total-tick budget (finishes the in-progress episode, doesn't cut it mid-run) — episode length is too variable (seen as low as ~450 and as high as ~5900 ticks) for `--episodes` alone to give a predictable run time.

**Real dataset** (`data/oracle_dataset.jsonl`, committed): 23 episodes, **51,073 ticks**, avg score 1025. One important finding before this gets used for training: the action distribution is heavily imbalanced —

| action | count | % |
| --- | --- | --- |
| none | 48,426 | 94.8% |
| jump | 1,211 | 2.4% |
| duck | 927 | 1.8% |
| left | 261 | 0.5% |
| right | 248 | 0.5% |

Makes sense (most ticks have no obstacle within reaction distance), but training directly on this as-is would teach Laya to mostly just say "none" and get ~95% tick-level accuracy while being useless at the only ticks that matter.

**Fixed via `balance_dataset.js`** (`node balance_dataset.js data/oracle_dataset.jsonl data/oracle_dataset_balanced.jsonl`): randomly downsamples "none" rows down to the same total as all non-"none" rows combined, leaving jump/duck/left/right untouched (no duplication of the smallest classes, to avoid overfitting on a handful of repeated frames). Result, `data/oracle_dataset_balanced.jsonl` (committed): **5,294 rows** — 2,647 "none" (50%), 1,211 jump, 927 duck, 261 left, 248 right. This specific 50/50 ratio is one defensible choice, not the only correct one — revisit if the fine-tuned result still leans "none"-biased.

## The Laya prompt

`src/laya_plays_subway_surfers/policy.py` is the Python side (Python, not JS, since that's where the `laya` package lives — set up the same way as `laya-model-router`: `uv init --lib` + `uv add laya`):

- `render_state(state)` — turns a `__getState()` snapshot into the text Laya sees, e.g. `"You are running in lane 1 ... Obstacles ahead, nearest first: - lane 1, 12 units away: a high barrier (duck under it)"`. Obstacle descriptions spell out duck-vs-jump in plain language rather than exposing the raw `isHigh` boolean.
- `QUESTIONS` — a `choice` question over `{left, right, jump, duck, none}`.

**Tested zero-shot against 30 real recorded states before trusting it** (`test_zero_shot.py`, same "validate the wording before building on it" approach as `laya-model-router`): **23/30 (76.7%) matched the oracle's action, with no fine-tuning at all.** One specific, real weakness found: Laya **never predicted "left" or "right" even once** across all 30 samples — including the actual left/right ground-truth cases (both read as "none") and some "train blocking the lane" cases where a lane change was exactly correct. Tried a more explicit criteria wording (`"dodge left, into the lane on the left"` vs. plain `"move left"`) — **identical result, same confusion matrix.** That rules out a surface wording problem; this is a genuine zero-shot blind spot, not a prompt-tuning one. Good news: the balanced dataset has 261 left + 248 right labeled examples specifically for fine-tuning to fix this.

## v1: fine-tuning (done)

Followed the exact same recipe as `laya-model-router/finetune/` (RLCD/GRPO via the official `train_ddp.py`, unmodified, run on Kaggle T4×2 via the `kaggle` CLI):

1. **Episode-level train/test split** (`finetune/split_dataset.py`), not a random tick-level split — consecutive ticks within one episode look nearly identical, so a random split would leak near-duplicate frames between train and test and inflate apparent accuracy. 23 episodes → 18 train / 5 test, each split then balanced independently with `balance_dataset.js`: **4,040 train rows, 1,254 test rows** (held-out episodes, never touched during training).
2. **Tokenization** (`finetune/build_train_items.py`) — same CPU-only approach as before, adapted for a `choice` question (5 options) instead of a `score` question (3 levels). 4,040/4,040 items built cleanly.
3. **Training on Kaggle** (`finetune/kaggle_kernel/`) — completed fully in ~23 minutes (4 epochs), checkpoint saved, calibration temperatures fitted. The notebook's own eval cell then failed — I'd hardcoded the dataset input path instead of discovering it with `glob`, the exact mistake already fixed once in `laya-model-router`'s pipeline and not propagated here. Training itself wasn't affected, so no need to rerun Kaggle: the checkpoint downloaded successfully, evaluated locally instead (`finetune/eval_checkpoint.py`, CPU).
4. **Real before/after, same 1,254 held-out test cases for both:**

| | Zero-shot | Fine-tuned |
| --- | --- | --- |
| Accuracy | 76.3% | **99.8%** |
| left | 0/39 correct | 39/39 |
| right | 0/35 correct | 35/35 |
| Mean latency | 42.3ms (CPU) | 143.9ms (CPU) |

The exact blind spot found during prompt validation reproduces exactly on this held-out set (left/right almost universally read as "none") and fine-tuning **eliminates it completely** — only 3 misclassifications total out of 1,254, all "none" read as an unnecessary (harmless) dodge. +23.5 points overall from fine-tuning on 4,040 examples.

## Status

Full pipeline verified working end-to-end: game bridge → oracle → data collection → balancing → episode-level split → prompt design → Kaggle fine-tuning → checkpoint evaluated against a real held-out test set, via committed runnable scripts, not one-off commands. Two real performance bugs were found and fixed while scaling data collection up (see the commit history on `play_oracle.js`/`game.html`): per-tick Node↔browser round trips were the real bottleneck (fixed by running the whole episode loop in-browser), and headless Chromium was still fully rendering every frame for no one to see (fixed with a `window.__fastMode` flag) — together, a 3.6x speedup.

## Watching it actually play

`play_laya.py` swaps the oracle for a real Laya agent (zero-shot or a fine-tuned checkpoint) making every decision live, instead of just measuring classifier accuracy on recorded states.

**The architecture problem this had to solve:** Laya inference only runs in Python, unlike the oracle (pure JS, could run a whole episode loop inside one `page.evaluate()` call). Every decision now needs a real Python↔browser round trip, which reintroduces per-tick latency (~100-150ms on CPU). Naively, that would cost the agent real in-game reaction time — the game would keep moving while Python "thinks." Fixed by adding a manual-step mode to `game.html` (`window.__tickOnce(dt)`): physics only advances by a fixed `dt` when explicitly told to, so slow inference never costs reaction time. This is fair, not a workaround — the game's difficulty scales with `distanceTraveled`, not wall-clock time, so stepping in "slow motion" doesn't change the difficulty curve at all, just how much real time elapses between decisions.

```bash
uv run python play_laya.py --zero-shot --episodes 1                                    # watch one run, headed
uv run python play_laya.py --checkpoint finetune/checkpoints/v1 --episodes 10 --headless
```

**Real result, 10 episodes each, actually playing the game (not just classifying recorded states):**

| | Zero-shot | Fine-tuned (v1) |
| --- | --- | --- |
| Avg score | 438.8 | **1562.6** |
| Avg distance survived | 439.2 | **1169.8** |

~3.6x higher score, ~2.7x farther survived. This is the number that actually matters — the 99.8% classifier accuracy above doesn't by itself prove the fine-tuned model plays *better*, only that it matches the oracle's labels more often; this result confirms that translates into materially better real gameplay, not just higher held-out accuracy on recorded frames.

## Open questions / next steps

- Make the oracle (and Laya) coin-seeking, not just obstacle-avoiding: `__getState()` currently only reports `obstaclesAhead` — the game's own `activeCoins` array isn't exposed at all yet. Would need (a) a `coinsAhead` field added to `__getState()` the same way `obstaclesAhead` works, and (b) oracle.js logic to prefer an adjacent lane with coins when it's not already forced to dodge something. Not done yet — just scoped.
- Tune `TICK_DT` in `play_laya.py` (currently 0.1s) — untested whether a smaller step size changes results meaningfully
- Try a bigger training set (the whole project used ~4k fine-tuning examples from 18 episodes) now that the pipeline is proven end to end
