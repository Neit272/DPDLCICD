import os
import json
from tqdm import tqdm

def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def encode_tokens(tokens, vocab, max_len=256):
    unk_id = vocab.get("<UNK>", 1)
    pad_id = vocab.get("<PAD>", 0)

    token_ids = [vocab.get(tok, unk_id) for tok in tokens]

    if len(token_ids) < max_len:
        token_ids += [pad_id] * (max_len - len(token_ids))
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
    base_in = "../preprocessed/token"
    base_out = "../preprocessed/encoded"
    vocab_path = "../preprocessed/vocab/vocab.json"
    max_len = 256

    os.makedirs(base_out, exist_ok=True)
    vocab = load_vocab(vocab_path)

    for split in ["train", "val", "test"]:
        in_file = os.path.join(base_in, f"{split}.jsonl")
        out_file = os.path.join(base_out, f"{split}.jsonl")
        encode_file(in_file, out_file, vocab, max_len)
