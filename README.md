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

## Status

**Bridge verified working end-to-end.** Served locally (`python3 -m http.server`, needed because Playwright blocks `file://`) and driven via Playwright:
- `__getState()` correctly reports approaching obstacles and crash state (verified: doing nothing crashes into a lane-1 train right at the start, exactly as expected).
- A simple hand-written reactive rule ("if an obstacle is in my lane within 15 units: dodge trains sideways, duck high barriers, jump low ones") using only `__getState`/`__action` survived 150 ticks (~12s simulated) without crashing, reaching a score of 848.

That reactive rule is the "oracle" policy the next step needs — not the end goal, but the thing that generates labeled (state, correct-action) training data for Laya, the same imitation-learning approach Laya's own documented browser-agent fine-tune used.

## Open questions / next steps

- Scale the oracle into a proper data-collection script: play many runs, log every tick's state + oracle action as a training example
- Design the actual Laya prompt: a `choice` question over `{left, right, jump, duck, none}` given a text rendering of `obstaclesAhead`
- Fine-tune on that data using the same Kaggle T4x2 pipeline built in `laya-model-router/finetune/`
- Swap the oracle for the fine-tuned Laya checkpoint and watch it play live
- Decide how to judge "did it work" — survival time / score vs. the oracle and vs. zero-shot Laya, on held-out runs
