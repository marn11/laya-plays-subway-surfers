"""Tokenize finetune/data/train.jsonl into train_items.pt, ready for Kaggle.

Runs entirely on CPU -- same approach as laya-model-router/finetune/build_train_items.py,
adapted for a "choice" question (5 options) instead of a "score" question (3 levels).
Reuses laya.common.build_sequence directly, the same function the official
fine-tuning notebook's data-prep cell uses.

Run from the repo root: uv run python finetune/build_train_items.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from laya.agent import _fix_tokenizer_config
from laya.common import QTYPES, build_sequence, render_options

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from laya_plays_subway_surfers.policy import QUESTIONS, render_state  # noqa: E402

MODEL_ID = "convaiinnovations/laya"
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

QUESTION = QUESTIONS["move"]
ACTION_KEYS = list(QUESTION["criteria"].keys())  # ["left", "right", "jump", "duck", "none"]

# No calibrated per-item probability exists (labels came from a rule-based
# oracle, not a teacher with real confidence scores) -- same fixed soft
# target used in laya-model-router: 0.85 on the correct action, split evenly
# across the other 4.
CONFIDENT = 0.85
OTHER = (1.0 - CONFIDENT) / (len(ACTION_KEYS) - 1)


def gold_probabilities(action: str) -> dict[str, float]:
    return {k: (CONFIDENT if k == action else OTHER) for k in ACTION_KEYS}


def build_training_item(tok, cfg, state_text: str, gold_probs: dict[str, float]):
    """Same logic as the official notebook's cell 6, specialized to our one choice question."""
    target = [gold_probs.get(k, 0.0) for k in ACTION_KEYS]
    s = sum(target)
    target = [v / s for v in target] if s > 0 else [1.0 / len(target)] * len(target)
    label = target.index(max(target))
    k = len(render_options({"t": "choice", "crit": QUESTION["criteria"]}))

    seq, markers = build_sequence(
        tok,
        state_text,
        {"t": "choice", "ins": QUESTION["instructions"], "crit": QUESTION["criteria"]},
        cfg["max_len"],
        cfg["head_max_len"],
    )
    if len(markers) != k:
        return None
    return {
        "ids": seq,
        "markers": markers,
        "qtype": QTYPES["choice"],
        "target": target,
        "label": label,
    }


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    rows = load_jsonl(DATA_DIR / "train.jsonl")

    print(f"Fetching tokenizer/config for {MODEL_ID}...")
    model_dir = snapshot_download(MODEL_ID)
    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)

    items = []
    skipped = 0
    for row in rows:
        state_text = render_state(row["state"])
        it = build_training_item(tok, cfg, state_text, gold_probabilities(row["action"]))
        if it is None:
            skipped += 1
            continue
        items.append(it)

    print(f"Built {len(items)} items ({skipped} skipped due to marker-count mismatch)")
    out_path = DATA_DIR / "train_items.pt"
    torch.save(items, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
