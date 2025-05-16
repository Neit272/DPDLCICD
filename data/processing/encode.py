import os
import json
from tqdm import tqdm

UNK_ID = 1
PAD_ID = 0
UNK_RANGE_START = 30000
UNK_RANGE_END = 80000

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def encode_tokens(tokens, vocab, max_len=256):
    token_ids = []
    next_unk_id = UNK_RANGE_START

    for tok in tokens:
        if tok in vocab:
            token_ids.append(vocab[tok])
        else:
            # Assign next available UNK bucket
            dynamic_unk = f"<UNK_{next_unk_id}>"
            if dynamic_unk in vocab:
                token_ids.append(vocab[dynamic_unk])
                next_unk_id += 1
            else:
                token_ids.append(UNK_ID)

    if len(token_ids) < max_len:
        token_ids += [PAD_ID] * (max_len - len(token_ids))
    else:
        token_ids = token_ids[:max_len]

    return token_ids

def encode_file(input_path, output_path, vocab, max_len=256):
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin, desc=f"Encoding {os.path.basename(input_path)}"):
            obj = json.loads(line)
            tokens = obj.get("tokens", [])
            label = obj.get("target", 0)
            input_ids = encode_tokens(tokens, vocab, max_len)
            fout.write(json.dumps({"input_ids": input_ids, "target": label}) + '\n')

if __name__ == "__main__":
    base_in = os.path.join(ROOT_DIR, "data", "preprocessed", "token")
    base_out = os.path.join(ROOT_DIR, "data", "preprocessed", "encoded")
    vocab_path = os.path.join(ROOT_DIR, "data", "preprocessed", "vocab", "vocab.json")
    max_len = 256

    os.makedirs(base_out, exist_ok=True)
    vocab = load_vocab(vocab_path)

    for split in ["train", "val", "test"]:
        in_file = os.path.join(base_in, f"{split}.jsonl")
        out_file = os.path.join(base_out, f"{split}.jsonl")
        encode_file(in_file, out_file, vocab, max_len)
