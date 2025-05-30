import json
import os
from pathlib import Path
from tqdm import tqdm
from scripts.preprocessing_pipeline import VulnerabilityPreprocessor

def build_vocabulary(processed_files: list, min_freq: int = 2) -> dict:
    """Build vocabulary from processed tokens"""
    token_counts = {}
    
    for file_path in processed_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                tokens = data.get('tokens', [])
                
                for token in tokens:
                    token_counts[token] = token_counts.get(token, 0) + 1
    
    # Filter by minimum frequency
    vocab = {'<PAD>': 0, '<UNK>': 1}  # Special tokens
    idx = 2
    
    for token, count in sorted(token_counts.items(), key=lambda x: x[1], reverse=True):
        if count >= min_freq:
            vocab[token] = idx
            idx += 1
    
    return vocab

def encode_tokens(tokens: list, vocab: dict, max_len: int = 256) -> dict:
    """Encode tokens to IDs with attention mask"""
    # Convert tokens to IDs
    input_ids = [vocab.get(token, vocab['<UNK>']) for token in tokens[:max_len]]
    
    # Pad to max_len
    attention_mask = [1] * len(input_ids) + [0] * (max_len - len(input_ids))
    input_ids = input_ids + [vocab['<PAD>']] * (max_len - len(input_ids))
    
    return {
        'input_ids': input_ids[:max_len],
        'attention_mask': attention_mask[:max_len]
    }

def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "data" / "preprocessed"
    output_dir = base_dir / "data" / "preprocessed" 
    
    # ✅ Create necessary directories FIRST
    (output_dir / "token").mkdir(parents=True, exist_ok=True)
    (output_dir / "vocab").mkdir(parents=True, exist_ok=True)
    (output_dir / "encoded").mkdir(parents=True, exist_ok=True)  # FIX: encode not encoded
    
    # Initialize preprocessor
    preprocessor = VulnerabilityPreprocessor()
    
    # Process each split
    splits = ['train', 'val', 'test']
    processed_files = []
    
    print("Step 1: Processing and tokenizing...")
    for split in splits:
        input_file = input_dir / f"{split}.jsonl"
        output_file = output_dir / "token" / f"{split}_tokens.jsonl"
        
        if not input_file.exists():
            print(f"⚠️  Input file not found: {input_file}")
            continue
            
        print(f"Processing {split} split...")
        
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line in tqdm(f_in, desc=f"Tokenizing {split}"):
                try:
                    data = json.loads(line.strip())
                    code = data.get('func', '')
                    
                    # Skip empty functions
                    if not code.strip():
                        continue
                    
                    # Process with our pipeline
                    tokens = preprocessor.process_and_tokenize(code, apply_slicing=False)
                    
                    # Skip if no tokens
                    if not tokens:
                        continue
                    
                    # Save processed data
                    data['tokens'] = tokens
                    f_out.write(json.dumps(data) + '\n')
                    
                except Exception as e:
                    print(f"Error processing line: {e}")
                    continue
        
        processed_files.append(output_file)
        print(f"✅ Saved tokenized {split} to {output_file}")
    
    if not processed_files:
        print("❌ No files were processed!")
        return
    
    print("Step 2: Building vocabulary...")
    vocab = build_vocabulary(processed_files, min_freq=10)
    
    # Save vocabulary
    vocab_file = output_dir / "vocab" / "vocab.json"
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2)
    
    print(f"✅ Vocabulary size: {len(vocab)}")
    print(f"✅ Vocabulary saved to {vocab_file}")
    
    print("Step 3: Encoding tokens...")
    for split in splits:
        # ✅ FIX: Use correct path for token files
        token_file = output_dir / "token" / f"{split}_tokens.jsonl"
        encoded_file = output_dir / "encoded" / f"{split}.jsonl"  # FIX: encode not encoded
        
        if not token_file.exists():
            print(f"⚠️  Token file not found: {token_file}")
            continue
            
        print(f"Encoding {split} split...")
        
        lines_processed = 0
        with open(token_file, 'r', encoding='utf-8') as f_in, \
             open(encoded_file, 'w', encoding='utf-8') as f_out:
            
            for line in tqdm(f_in, desc=f"Encoding {split}"):
                try:
                    data = json.loads(line.strip())
                    tokens = data.get('tokens', [])
                    
                    # Skip empty token lists
                    if not tokens:
                        continue
                    
                    # Encode tokens
                    encoded = encode_tokens(tokens, vocab, max_len=256)
                    
                    # Create final data
                    final_data = {
                        'input_ids': encoded['input_ids'],
                        'attention_mask': encoded['attention_mask'],
                        'target': data.get('target', 0)
                    }
                    
                    f_out.write(json.dumps(final_data) + '\n')
                    lines_processed += 1
                    
                except Exception as e:
                    print(f"Error encoding line: {e}")
                    continue
        
        print(f"✅ Encoded {lines_processed} samples for {split}")
        print(f"✅ Saved to {encoded_file}")
    
    print("\n🎉 Preprocessing complete!")

if __name__ == "__main__":
    main()