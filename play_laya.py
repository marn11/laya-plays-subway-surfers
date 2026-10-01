"""Play the game with a Laya agent making every decision -- zero-shot or a
fine-tuned checkpoint -- instead of the oracle.

Laya inference only runs in Python, unlike oracle.js, so it can't be
injected into the browser and run in one page.evaluate() call the way
play_oracle.js does. Each decision needs a real Python<->browser round
trip. To keep that from being unfair to the agent, the game runs in
"manual physics" mode (game.html's window.__tickOnce(dt)): physics only
advances by a fixed dt when explicitly told to, so slow Python-side
inference never costs the agent in-game reaction time. Difficulty scales
with distanceTraveled, not wall-clock time, so this doesn't change the
game itself -- just how much real time elapses between ticks.

Usage (from the repo root):
    uv run python play_laya.py --zero-shot --episodes 1                      # watch one run, headed
    uv run python play_laya.py --checkpoint finetune/checkpoints/v1 \\
        --episodes 10 --headless                                             # compare to the oracle's numbers
"""

from __future__ import annotations

import argparse
import functools
import http.server
import sys
import threading
from pathlib import Path

import laya
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).parent / "src"))
from laya_plays_subway_surfers.policy import QUESTIONS, render_state  # noqa: E402

ROOT = Path(__file__).parent
DEFAULT_TICK_DT = 0.1  # seconds of simulated game time per decision -- see module docstring.
# As speed grows (uncapped, see game.html), a fixed dt gives fewer and fewer
# decision opportunities within the oracle's 15-unit reaction window -- down
# to ~2.6 ticks of lead time by distance 2454 in one real run. Pass
# --tick-dt to test whether a finer step size raises the score ceiling.


def serve_dir(root: Path, port: int) -> http.server.ThreadingHTTPServer:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    server = http.server.ThreadingHTTPServer(("localhost", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def play_one_episode(page, agent, tick_dt: float) -> dict:
    page.evaluate("() => window.__restart()")
    state = page.evaluate("() => window.__getState()")
    ticks = 0

    while state["isPlaying"]:
        answer = agent.predict(render_state(state), QUESTIONS)["answers"]["move"]
        action = answer["choice"]
        if action != "none":
            page.evaluate("(a) => window.__action(a)", action)
        state = page.evaluate("(dt) => window.__tickOnce(dt)", tick_dt)
        ticks += 1

    return {"ticks": ticks, "score": state["score"], "distanceTraveled": state["distanceTraveled"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--zero-shot", action="store_true")
    group.add_argument("--checkpoint", type=str)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--port", type=int, default=8732)
    parser.add_argument("--tick-dt", type=float, default=DEFAULT_TICK_DT)
    args = parser.parse_args()

    if args.zero_shot:
        agent = laya.Router()
        label = "zero-shot"
    else:
        agent = laya.Agent(args.checkpoint, device="cpu")
        label = f"fine-tuned ({args.checkpoint})"

    server = serve_dir(ROOT, args.port)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=args.headless)
            page = browser.new_page()
            page.goto(f"http://localhost:{args.port}/game.html")
            page.evaluate("(fast) => { window.__manualPhysics = true; window.__fastMode = fast; }", args.headless)

            results = []
            for ep in range(1, args.episodes + 1):
                result = play_one_episode(page, agent, args.tick_dt)
                results.append(result)
                print(f"[{label}] episode {ep}/{args.episodes}: {result['ticks']} ticks, score {result['score']}, distance {result['distanceTraveled']:.1f}")

            browser.close()
    finally:
        server.shutdown()

    avg_score = sum(r["score"] for r in results) / len(results)
    avg_distance = sum(r["distanceTraveled"] for r in results) / len(results)
    print(f"\n[{label}] {len(results)} episode(s) -- avg score {avg_score:.1f}, avg distance {avg_distance:.1f}")


if __name__ == "__main__":
    main()
