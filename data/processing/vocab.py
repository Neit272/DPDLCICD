import os
import json
from datetime import datetime
from collections import Counter

def build_vocab(file_paths, min_freq=1):
    token_counter = Counter()

    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                tokens = obj.get('tokens', [])
                token_counter.update(tokens)

    # Lọc theo min_freq
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }
    idx = 2
    for token, count in token_counter.items():
        if count >= min_freq:
            vocab[token] = idx
            idx += 1

    return vocab

def save_vocab(vocab, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=4)

if __name__ == "__main__":
    today = datetime.now().strftime("%d-%m-%Y")
    base_path = "../preprocessed/token"
    output_path = f"../preprocessed/vocab/vocab_{today}.json"

    train_file = os.path.join(base_path, "train.jsonl")
    val_file = os.path.join(base_path, "val.jsonl")

    vocab = build_vocab([train_file, val_file], min_freq=1)
    save_vocab(vocab, output_path)
    print(f"Saved vocab to {output_path}, size = {len(vocab)}")
