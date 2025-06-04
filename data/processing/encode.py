import os
import json
from tqdm import tqdm

PAD_ID = 0
UNK_ID = 1

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def load_vocabulary(vocab_path):
    """Load vocabulary from JSON file"""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    print(f"Loaded vocabulary: {len(vocab):,} tokens")
    return vocab

def encode_tokens(tokens, vocab, max_len=256):
    """Encode tokens to IDs with padding/truncation"""
    token_ids = []
    
    for token in tokens:
        if token in vocab:
            token_ids.append(vocab[token])
        else:
            token_ids.append(UNK_ID)
    
    # Pad or truncate
    if len(token_ids) < max_len:
        token_ids += [PAD_ID] * (max_len - len(token_ids))
    else:
        token_ids = token_ids[:max_len]
    
    return token_ids

def encode_file(input_path, output_path, vocab, max_len=256):
    """Encode a single file from tokens to IDs"""
    processed = 0
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in tqdm(fin, desc=f"Encoding {os.path.basename(input_path)}"):
            try:
                obj = json.loads(line)
                tokens = obj.get("tokens", [])
                label = obj.get("target", 0)
                
                if not tokens or label not in [0, 1]:
                    continue
                
                # Encode tokens
                input_ids = encode_tokens(tokens, vocab, max_len)
                
                # Write encoded sample
                encoded_sample = {
                    "input_ids": input_ids,
                    "target": label
                }
                
                fout.write(json.dumps(encoded_sample) + '\n')
                processed += 1
                
            except (json.JSONDecodeError, KeyError):
                continue
    
    print(f"   Processed: {processed:,} samples")
    return processed > 0

def encode_dataset(vocab_path, input_dir, output_dir, max_len=256):
    """Encode entire dataset"""
    # Load vocabulary
    vocab = load_vocabulary(vocab_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each split
    for split in ["train", "val", "test"]:
        input_file = os.path.join(input_dir, f"{split}.jsonl")
        output_file = os.path.join(output_dir, f"{split}.jsonl")
        
        if os.path.exists(input_file):
            print(f"\nProcessing {split.upper()}:")
            encode_file(input_file, output_file, vocab, max_len)
        else:
            print(f"Skipping {split} - file not found")
    
    print("\nEncoding completed!")

if __name__ == "__main__":
    # Configuration
    input_dir = os.path.join(ROOT_DIR, "data", "preprocessed", "token")
    output_dir = os.path.join(ROOT_DIR, "data", "preprocessed", "encoded")
    vocab_path = os.path.join(ROOT_DIR, "data", "preprocessed", "vocab", "vocab.json")
    max_len = 256
    
    # Encode dataset
    encode_dataset(vocab_path, input_dir, output_dir, max_len)