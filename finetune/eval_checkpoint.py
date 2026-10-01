"""Evaluate a Laya agent (zero-shot base model OR a fine-tuned checkpoint)
against finetune/data/test.jsonl -- the held-out episode-level test split,
never trained on.

The Kaggle notebook's own eval cell hit the same hardcoded-dataset-path
mistake already fixed once in laya-model-router (used /kaggle/input/<slug>/
directly instead of discovering it with glob) -- but training itself
completed fully and the checkpoint downloaded fine, so there was no need to
rerun Kaggle; evaluating locally on CPU here instead.

Usage (from the repo root):
    uv run python finetune/eval_checkpoint.py --zero-shot
    uv run python finetune/eval_checkpoint.py --checkpoint finetune/checkpoints/v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import laya

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from laya_plays_subway_surfers.policy import ACTION_NAMES, QUESTIONS, render_state  # noqa: E402

ROOT = Path(__file__).parent
TEST_PATH = ROOT / "data" / "test.jsonl"


def load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--zero-shot", action="store_true", help="Use the un-fine-tuned base checkpoint.")
    group.add_argument("--checkpoint", type=str, help="Path to a fine-tuned checkpoint directory.")
    args = parser.parse_args()

    if args.zero_shot:
        agent = laya.Router()
        predict = lambda text: agent.predict(text, QUESTIONS)["answers"]["move"]
        label = "zero-shot"
    else:
        agent = laya.Agent(args.checkpoint, device="cpu")
        predict = lambda text: agent.predict(text, QUESTIONS)["answers"]["move"]
        label = f"fine-tuned ({args.checkpoint})"

    rows = load_rows(TEST_PATH)
    matrix = {(gt, pred): 0 for gt in ACTION_NAMES for pred in ACTION_NAMES}
    correct = 0
    latencies = []

    for row in rows:
        state_text = render_state(row["state"])
        start = time.perf_counter()
        answer = predict(state_text)
        latencies.append(time.perf_counter() - start)
        predicted = answer["choice"]
        actual = row["action"]
        matrix[(actual, predicted)] += 1
        if predicted == actual:
            correct += 1

    accuracy = correct / len(rows)
    print(f"{label} accuracy on {len(rows)} held-out test cases: {accuracy:.1%}")
    print(f"Mean latency: {sum(latencies) / len(latencies) * 1000:.1f} ms")
    print("\nConfusion matrix (rows=ground truth, cols=predicted):")
    header = "          " + "  ".join(f"{a:>6s}" for a in ACTION_NAMES)
    print(header)
    for gt in ACTION_NAMES:
        print(f"{gt:>8s}  " + "  ".join(f"{matrix[(gt, pred)]:>6d}" for pred in ACTION_NAMES))


if __name__ == "__main__":
    main()
