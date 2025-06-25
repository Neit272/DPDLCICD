import os
import json
import re
import glob
from collections import Counter
from tqdm import tqdm

# Import từ tokenize_balanced thay vì tokenizer2
from tokenize_balanced import (
    CRITICAL_FUNCTIONS, C_KEYWORDS, C_TYPES, OPERATORS, 
    MEANINGFUL_IDENTIFIERS, is_complex_identifier, is_noise_token
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def build_vocabulary(file_paths, min_freq=2, max_vocab_size=30000):
    """Build vocabulary with semantic prioritization"""
    print("Building vocabulary...")
    
    token_counter = Counter()
    
    # Count all tokens
    for path in file_paths:
        print(f"Counting tokens in {os.path.basename(path)}...")
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Counting {os.path.basename(path)}"):
                try:
                    obj = json.loads(line)
                    for token in obj.get('tokens', []):
                        # Skip obvious noise during counting
                        if not is_noise_token(token):
                            token_counter[token] += 1
                except json.JSONDecodeError:
                    continue
    
    print(f"Found {len(token_counter):,} unique meaningful tokens")
    
    # Build vocabulary with smart prioritization
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    
    # Track what we add
    added_stats = {
        'keywords': 0, 'types': 0, 'critical_funcs': 0,
        'operators': 0, 'complex_ids': 0, 'meaningful_ids': 0,
        'variables': 0, 'functions': 0, 'other': 0
    }
    
    # 1. C Keywords (highest priority)
    for token in C_KEYWORDS:
        if token in token_counter and token not in vocab:
            vocab[token] = idx
            idx += 1
            added_stats['keywords'] += 1
    
    # 2. C Types
    for token in C_TYPES:
        if token in token_counter and token not in vocab:
            vocab[token] = idx
            idx += 1
            added_stats['types'] += 1
    
    # 3. Critical functions (vulnerability-related)
    for token in CRITICAL_FUNCTIONS:
        if token in token_counter and token not in vocab:
            vocab[token] = idx
            idx += 1
            added_stats['critical_funcs'] += 1
    
    # 4. Operators and symbols
    for token in OPERATORS:
        if token in token_counter and token not in vocab:
            vocab[token] = idx
            idx += 1
            added_stats['operators'] += 1
    
    # 5. Semantic tokens (HIGH PRIORITY!)
    semantic_tokens = []
    for token, count in token_counter.most_common():
        if token.startswith(('VAR_', 'FUNC_')) and count >= min_freq:
            semantic_tokens.append((token, count))
    
    print(f"Found {len(semantic_tokens)} semantic tokens")
    
    # Add semantic tokens with high priority
    for token, count in semantic_tokens:
        if idx >= max_vocab_size:
            break
        if token not in vocab:
            if token.startswith('VAR_'):
                vocab[token] = idx
                idx += 1
                added_stats['variables'] += 1
            elif token.startswith('FUNC_'):
                vocab[token] = idx
                idx += 1
                added_stats['functions'] += 1
    
    # 6. Complex identifiers (like SSL_CTX_set_mode)
    complex_tokens = [
        (token, count) for token, count in token_counter.most_common()
        if is_complex_identifier(token) and token not in vocab and count >= min_freq
    ]
    
    for token, count in complex_tokens:
        if idx >= max_vocab_size:
            break
        vocab[token] = idx
        idx += 1
        added_stats['complex_ids'] += 1
    
    # 7. Meaningful identifiers
    for token in MEANINGFUL_IDENTIFIERS:
        if token in token_counter and token not in vocab:
            vocab[token] = idx
            idx += 1
            added_stats['meaningful_ids'] += 1
    
    # 8. Common numbers (small ones)
    for token, count in token_counter.most_common():
        if idx >= max_vocab_size:
            break
        if (token not in vocab and count >= min_freq and
            (re.match(r'^\d{1,3}$', token) or re.match(r'^0[xX][0-9a-fA-F]{1,4}$', token))):
            vocab[token] = idx
            idx += 1
            added_stats['other'] += 1
    
    # 9. Remaining high-frequency meaningful tokens
    for token, count in token_counter.most_common():
        if idx >= max_vocab_size:
            break
        if (token not in vocab and count >= min_freq and
            re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token) and
            len(token) >= 3 and not is_noise_token(token)):
            vocab[token] = idx
            idx += 1
            added_stats['other'] += 1
    
    # Print statistics
    print(f"\nVocabulary composition:")
    total_meaningful = sum(added_stats.values())
    for category, count in added_stats.items():
        if count > 0:
            percentage = count / total_meaningful * 100
            print(f"   {category.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    return vocab, token_counter

def validate_vocabulary_detailed(vocab, file_paths):
    """Detailed vocabulary validation"""
    print("\nValidating vocabulary coverage...")
    
    total_tokens = covered_tokens = 0
    unknown_counter = Counter()
    
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Validating {os.path.basename(path)}"):
                try:
                    obj = json.loads(line)
                    for token in obj.get('tokens', []):
                        total_tokens += 1
                        
                        if token in vocab:
                            covered_tokens += 1
                        else:
                            unknown_counter[token] += 1
                            
                except json.JSONDecodeError:
                    continue
    
    coverage = covered_tokens / total_tokens if total_tokens > 0 else 0
    
    print(f"   Coverage Results:")
    print(f"   Total tokens: {total_tokens:,}")
    print(f"   Covered: {covered_tokens:,}")
    print(f"   Unknown: {len(unknown_counter):,} unique ({total_tokens - covered_tokens:,} total)")
    print(f"   Coverage: {coverage:.2%}")
    
    # Show semantic token statistics
    semantic_in_vocab = {k: v for k, v in vocab.items() if k.startswith(('VAR_', 'FUNC_'))}
    if semantic_in_vocab:
        print(f"\nSemantic tokens in vocabulary:")
        var_count = len([k for k in semantic_in_vocab.keys() if k.startswith('VAR_')])
        func_count = len([k for k in semantic_in_vocab.keys() if k.startswith('FUNC_')])
        print(f"   VAR_ tokens: {var_count}")
        print(f"   FUNC_ tokens: {func_count}")
        print(f"   Total semantic: {len(semantic_in_vocab)}")
    
    return coverage

def main():
    print("BALANCED VOCABULARY BUILDER")
    print("=" * 50)
    
    # Configuration - only balanced mode
    token_dir = os.path.join(ROOT_DIR, "data", "preprocessed", "balanced_tokenized")
    output_dir = os.path.join(ROOT_DIR, "data", "preprocessed", "vocab_balanced")
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if balanced_tokenized directory exists
    if not os.path.exists(token_dir):
        print("balanced_tokenized directory not found!")
        print(f"Expected location: {token_dir}")
        print("   Please run tokenize_balanced.py first!")
        return
    
    # Find balanced tokenized files
    files = []
    
    # Look for val.jsonl and test.jsonl (from balanced mode)
    for split in ['val.jsonl', 'test.jsonl']:
        file_path = os.path.join(token_dir, split)
        if os.path.exists(file_path):
            files.append(file_path)
    
    # Look for train_balanced_*.jsonl files (use first 3 for vocab building)
    train_pattern = os.path.join(token_dir, "train_balanced_*.jsonl")
    train_files = sorted(glob.glob(train_pattern))
    
    if train_files:
        # Use first 3 train files for vocabulary building (should be representative)
        files.extend(train_files[:3])
    
    if not files:
        print("No balanced tokenized files found!")
        print(f"Checked directory: {token_dir}")
        print("   Expected files: val.jsonl, test.jsonl, train_balanced_*.jsonl")
        print("   Please run tokenize_balanced.py first!")
        return
    
    print(f"Found balanced tokenized files")
    print(f"Building vocabulary from {len(files)} files:")
    for f in files:
        print(f"   {os.path.basename(f)}")
    
    print("=" * 50)
    
    # Build vocabulary with semantic support
    vocab, token_counter = build_vocabulary(
        files, 
        min_freq=2, 
        max_vocab_size=30000
    )
    
    # Validate coverage on all available balanced files
    validation_files = []
    
    # Add val and test files for validation
    for split in ['val.jsonl', 'test.jsonl']:
        file_path = os.path.join(token_dir, split)
        if os.path.exists(file_path):
            validation_files.append(file_path)
    
    # Add all train files for validation
    validation_files.extend(sorted(glob.glob(train_pattern)))
    
    if validation_files:
        coverage = validate_vocabulary_detailed(vocab, validation_files)
    else:
        coverage = 0.0
        print("No files available for validation")
    
    # Save vocabulary
    vocab_path = os.path.join(output_dir, "vocab_balanced.json")
    
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2)
    
    print(f"\nVocabulary completed!")
    print(f"Output directory: {output_dir}")
    print(f"Results:")
    print(f"   Size: {len(vocab):,} tokens")
    print(f"   Coverage: {coverage:.2%}")
    print(f"   Semantic tokens included: ")
    print(f"   Saved to: {vocab_path}")

if __name__ == "__main__":
    main()