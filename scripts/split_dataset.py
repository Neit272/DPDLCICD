import json
import random
from pathlib import Path

random.seed(42)
input_path = "data/datasets/clean_diversevul_20230702.jsonl"
output_dir = Path("data/preprocessed")
output_dir.mkdir(parents=True, exist_ok=True)

with open(input_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

random.shuffle(data)
n = len(data)
splits = {
    "train": data[:int(0.8 * n)],
    "val": data[int(0.8 * n):int(0.9 * n)],
    "test": data[int(0.9 * n):]
}

for split, items in splits.items():
    out_path = output_dir / f"{split}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
