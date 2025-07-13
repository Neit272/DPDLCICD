import os
import json
import glob
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

def encode_dataset(vocab_path, input_dir, output_dir, input_files, max_len=256):
    """Encode entire dataset"""
    # Load vocabulary
    vocab = load_vocabulary(vocab_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    total_processed = 0
    successful_files = 0
    
    # Process each file
    for input_name, output_name in input_files:
        input_file = os.path.join(input_dir, input_name)
        output_file = os.path.join(output_dir, output_name)
        
        if os.path.exists(input_file):
            print(f"\nProcessing {input_name}:")
            success = encode_file(input_file, output_file, vocab, max_len)
            
            if success:
                # Count samples in output file
                with open(output_file, 'r') as f:
                    count = sum(1 for _ in f)
                total_processed += count
                successful_files += 1
            else:
                print(f"   Failed to process {input_name}")
        else:
            print(f"   Skipping {input_name} - file not found")
    
    print(f"\n✅ Encoding completed!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Results:")
    print(f"   Files processed: {successful_files}/{len(input_files)}")
    print(f"   Total samples encoded: {total_processed:,}")

def main():
    print("🚀 BALANCED ENCODER")
    print("=" * 50)
    
    # Configuration - simplified vocab path
    token_dir = os.path.join(ROOT_DIR, "data", "preprocessed", "balanced_tokenized")
    output_dir = os.path.join(ROOT_DIR, "data", "preprocessed", "encoded_balanced")
    vocab_path = os.path.join(ROOT_DIR, "data", "preprocessed", "vocab", "vocab.json") 
    
    # Check if balanced_tokenized directory exists
    if not os.path.exists(token_dir):
        print("❌ balanced_tokenized directory not found!")
        print(f"Expected location: {token_dir}")
        print("   Please run tokenize_balanced.py first!")
        return
    
    # Check if vocabulary exists
    if not os.path.exists(vocab_path):
        print(f"❌ Vocabulary not found: {vocab_path}")
        print("   Please run vocab_balanced.py first!")
        return
    
    # Find balanced tokenized files
    input_files = []
    
    # Look for val.jsonl, test.jsonl (from balanced mode)
    for split in ['val.jsonl', 'test.jsonl']:
        file_path = os.path.join(token_dir, split)
        if os.path.exists(file_path):
            input_files.append((split, split))
    
    # Look for train_balanced_*.jsonl files
    train_pattern = os.path.join(token_dir, "train_balanced_*.jsonl")
    train_files = sorted(glob.glob(train_pattern))
    
    # Add all train files
    for train_file in train_files:
        filename = os.path.basename(train_file)
        input_files.append((filename, filename))
    
    if not input_files:
        print("❌ No balanced tokenized files found!")
        print(f"Checked directory: {token_dir}")
        print("   Expected files: val.jsonl, test.jsonl, train_balanced_*.jsonl")
        print("   Please run tokenize_balanced.py first!")
        return
    
    print(f"✅ Found balanced tokenized files")
    print(f"📊 Found {len(input_files)} files to encode:")
    for inp, out in input_files:
        print(f"   {inp} -> {out}")
    
    print("=" * 50)
    
    # Encode dataset
    max_len = 256
    encode_dataset(vocab_path, token_dir, output_dir, input_files, max_len)
    
    print(f"\n🎯 Mode: BALANCED")
    print(f"📁 Vocabulary: {os.path.basename(vocab_path)}")
    print(f"📏 Max sequence length: {max_len}")

if __name__ == "__main__":
    main()