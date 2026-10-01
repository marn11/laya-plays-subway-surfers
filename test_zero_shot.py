"""Sanity-check the prompt design (policy.py) against zero-shot Laya on real
recorded states, before trusting it enough to build the fine-tuning data
pipeline on top of it. Same "test the wording before committing" approach
used in laya-model-router.

Run from the repo root: uv run python test_zero_shot.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from laya import Router

from laya_plays_subway_surfers.policy import ACTION_NAMES, QUESTIONS, render_state

DATA_PATH = Path(__file__).parent / "data" / "oracle_dataset_balanced.jsonl"
SAMPLE_SIZE = 30
SEED = 20261002


def load_sample(path: Path, n: int, seed: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    random.Random(seed).shuffle(rows)
    return rows[:n]


def main() -> None:
    rows = load_sample(DATA_PATH, SAMPLE_SIZE, SEED)
    router = Router()

    matrix = {(gt, pred): 0 for gt in ACTION_NAMES for pred in ACTION_NAMES}
    correct = 0

    for row in rows:
        state_text = render_state(row["state"])
        result = router.predict(state_text, QUESTIONS)
        predicted = result["answers"]["move"]["choice"]
        actual = row["action"]
        matrix[(actual, predicted)] += 1
        if predicted == actual:
            correct += 1

        mark = "match" if predicted == actual else "DIFF "
        print(f"[{mark}] oracle={actual:6s} laya={predicted:6s} | {state_text.splitlines()[-1] if row['state']['obstaclesAhead'] else 'no obstacles'}")

    print(f"\n{correct}/{len(rows)} match the oracle's action (zero-shot, no fine-tuning)")
    print("\nConfusion matrix (rows=oracle action, cols=laya prediction):")
    header = "          " + "  ".join(f"{a:>6s}" for a in ACTION_NAMES)
    print(header)
    for gt in ACTION_NAMES:
        print(f"{gt:>8s}  " + "  ".join(f"{matrix[(gt, pred)]:>6d}" for pred in ACTION_NAMES))


if __name__ == "__main__":
    main()
