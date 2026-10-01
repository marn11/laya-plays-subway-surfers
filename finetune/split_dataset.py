"""Episode-level train/test split, then balance each split independently.

Splitting by individual tick (as the earlier balance_dataset.js pass did,
for a quick dataset-imbalance fix) risks leaking near-duplicate frames
between train and test -- consecutive ticks within one episode look nearly
identical. data/oracle_dataset.jsonl preserves episode order (written
sequentially by play_oracle.js, before data/oracle_dataset_balanced.jsonl
shuffled everything), so episode boundaries can be recovered here by
watching for distanceTraveled resetting to near-zero.

Run from the repo root: uv run python finetune/split_dataset.py
Then balance each split the same way as before:
    node balance_dataset.js finetune/data/train_raw.jsonl finetune/data/train.jsonl
    node balance_dataset.js finetune/data/test_raw.jsonl finetune/data/test.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).parent
RAW_PATH = ROOT.parent / "data" / "oracle_dataset.jsonl"
OUT_DIR = ROOT / "data"
N_TEST_EPISODES = 5  # out of 23 total -- small dataset, so held out a bit more than a strict 90/10 would
SEED = 20261003


def load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def assign_episode_ids(rows: list[dict]) -> list[int]:
    episode_ids = []
    current = 0
    prev_distance = None
    for row in rows:
        distance = row["state"]["distanceTraveled"]
        if prev_distance is not None and distance < prev_distance - 1.0:
            current += 1
        episode_ids.append(current)
        prev_distance = distance
    return episode_ids


def main() -> None:
    rows = load_rows(RAW_PATH)
    episode_ids = assign_episode_ids(rows)
    n_episodes = episode_ids[-1] + 1
    print(f"Loaded {len(rows)} ticks across {n_episodes} episodes")

    rng = random.Random(SEED)
    all_episodes = list(range(n_episodes))
    rng.shuffle(all_episodes)
    test_episodes = set(all_episodes[:N_TEST_EPISODES])
    train_episodes = set(all_episodes[N_TEST_EPISODES:])

    train_rows = [r for r, eid in zip(rows, episode_ids) if eid in train_episodes]
    test_rows = [r for r, eid in zip(rows, episode_ids) if eid in test_episodes]

    OUT_DIR.mkdir(exist_ok=True)
    for name, split_rows in [("train_raw.jsonl", train_rows), ("test_raw.jsonl", test_rows)]:
        with (OUT_DIR / name).open("w") as f:
            for r in split_rows:
                f.write(json.dumps(r) + "\n")

    print(f"train: {len(train_episodes)} episodes, {len(train_rows)} ticks -> finetune/data/train_raw.jsonl")
    print(f"test:  {len(test_episodes)} episodes, {len(test_rows)} ticks -> finetune/data/test_raw.jsonl")
    print("\nNow balance each split:")
    print("  node balance_dataset.js finetune/data/train_raw.jsonl finetune/data/train.jsonl")
    print("  node balance_dataset.js finetune/data/test_raw.jsonl finetune/data/test.jsonl")


if __name__ == "__main__":
    main()
