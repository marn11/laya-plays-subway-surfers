"""Laya's view of the game: how a game state becomes a prompt, and the
choice question that picks an action.

Mirrors what we learned in laya-model-router: Laya's criteria wording
matters a lot (short phrases beat long paragraphs there) -- so keep each
option here to one or two words and put any real explanation in
`instructions` / the state text instead.
"""

from __future__ import annotations

ACTION_NAMES = ("left", "right", "jump", "duck", "none")

QUESTIONS = {
    "move": {
        "type": "choice",
        "instructions": "What should the runner do right now?",
        "criteria": {
            "left": "dodge left, into the lane on the left",
            "right": "dodge right, into the lane on the right",
            "jump": "jump over a low obstacle",
            "duck": "duck under a high obstacle",
            "none": "do nothing, stay in the current lane",
        },
    }
}


def render_state(state: dict) -> str:
    """Turn a __getState() snapshot into the text Laya sees as `state`."""
    lane = state["currentLane"]
    obstacles = state["obstaclesAhead"]

    lines = [f"You are running in lane {lane} (lanes are 0=left, 1=middle, 2=right)."]
    if not obstacles:
        lines.append("No obstacles ahead.")
    else:
        lines.append("Obstacles ahead, nearest first:")
        for o in obstacles:
            if o["type"] == "train":
                desc = "a train blocking the whole lane (must change lanes)"
            elif o["isHigh"]:
                desc = "a high barrier (duck under it)"
            else:
                desc = "a low barrier (jump over it)"
            lines.append(f"- lane {o['lane']}, {o['distance']:.0f} units away: {desc}")

    return "\n".join(lines)


def decide_action(router, state: dict) -> dict:
    """Run the choice question through a laya.Router and return the full answer."""
    result = router.predict(render_state(state), QUESTIONS)
    return result["answers"]["move"]
