import os
import json
from collections import Counter

VOCAB_LIMIT = 30000
UNK_START = 30000
UNK_END = 80000

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1
}

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def build_vocab(file_paths, min_freq=1, limit=VOCAB_LIMIT):
    token_counter = Counter()

    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                tokens = obj.get('tokens', [])
                token_counter.update(tokens)

    # Lọc theo min_freq và giới hạn số lượng
    most_common = token_counter.most_common()
    vocab = dict(SPECIAL_TOKENS)  # Start with special tokens
    idx = max(vocab.values()) + 1

    for token, count in most_common:
        if count >= min_freq and idx < limit:
            vocab[token] = idx
            idx += 1

    # Preallocate UNK bucket tokens
    for unk_idx in range(UNK_START, UNK_END):
        vocab[f"<UNK_{unk_idx}>"] = unk_idx

    return vocab

def save_vocab(vocab, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=4)

if __name__ == "__main__":
    base_path = os.path.join(ROOT_DIR, "data", "preprocessed", "token")
    output_path = os.path.join(ROOT_DIR, "data", "preprocessed", "vocab", "vocab.json")

    train_file = os.path.join(base_path, "train.jsonl")
    val_file = os.path.join(base_path, "val.jsonl")

    vocab = build_vocab([train_file, val_file], min_freq=4)
    save_vocab(vocab, output_path)
    print(f"Saved vocab to {output_path}, size = {len(vocab)}")
