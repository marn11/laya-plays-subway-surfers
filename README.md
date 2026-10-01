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

node play_oracle.js                                  # watch one run (headed, a window opens)
node play_oracle.js --episodes 5 --headless \
  --record data/oracle_sample.jsonl                  # play 5 runs, no window, log every tick
```

Each episode prints `ticks`, `score`, and `distance`. `--record <path>` appends one JSON line per tick (`{"state": {...}, "action": "..."}`) to that file — this is the raw material for fine-tuning.

**Real run** (5 episodes, headless, `data/oracle_sample.jsonl`): scores of 287, 534, 158, 2299, 386 — avg **733**, highly variable (random obstacle generation plus ever-increasing speed means the simple reflex rules occasionally get caught out, e.g. two obstacles in adjacent lanes at once). 857 (state, action) pairs logged in this one sample run.

## Status

Bridge + oracle both verified working end-to-end, with a real committed, runnable script (`play_oracle.js`) rather than one-off commands — see "Running it yourself" above.

## Open questions / next steps

- Make the oracle coin-seeking, not just obstacle-avoiding: `__getState()` currently only reports `obstaclesAhead` — the game's own `activeCoins` array isn't exposed at all yet. Would need (a) a `coinsAhead` field added to `__getState()` the same way `obstaclesAhead` works, and (b) oracle.js logic to prefer an adjacent lane with coins when it's not already forced to dodge something. Not done yet — just scoped.
- Scale data collection: run many more episodes (hundreds+) to build a real training set, not just the 857-tick sample above
- Design the actual Laya prompt: a `choice` question over `{left, right, jump, duck, none}` given a text rendering of `obstaclesAhead`
- Fine-tune on that data using the same Kaggle T4x2 pipeline built in `laya-model-router/finetune/`
- Swap the oracle for the fine-tuned Laya checkpoint in `play_oracle.js` (or a new `play_laya.js`) and watch it play live
- Decide how to judge "did it work" — survival time / score vs. the oracle and vs. zero-shot Laya, on held-out runs
