import os
import json
import re
from collections import Counter
from tqdm import tqdm

# Import từ tokenizer2 thay vì tokenizer
from data.processing.tokenizer import (
    CRITICAL_FUNCTIONS, C_KEYWORDS, C_TYPES, OPERATORS, 
    MEANINGFUL_IDENTIFIERS, is_complex_identifier, is_noise_token
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def build_vocabulary(file_paths, min_freq=2, max_vocab_size=30000):
    print("Building vocabulary...")
    
    token_counter = Counter()
    
    # Count all tokens
    for path in file_paths:
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
    
    # 5. Complex identifiers (like SSL_CTX_set_mode)
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
    
    # 6. Meaningful identifiers
    for token in MEANINGFUL_IDENTIFIERS:
        if token in token_counter and token not in vocab:
            vocab[token] = idx
            idx += 1
            added_stats['meaningful_ids'] += 1
    
    # 7. Common numbers (small ones)
    for token, count in token_counter.most_common():
        if idx >= max_vocab_size:
            break
        if (token not in vocab and count >= min_freq and
            (re.match(r'^\d{1,3}$', token) or re.match(r'^0[xX][0-9a-fA-F]{1,4}$', token))):
            vocab[token] = idx
            idx += 1
            added_stats['other'] += 1
    
    # 8. Normalized variables and functions (SEMANTIC TOKENS - PRIORITY!)
    semantic_tokens = []
    for token, count in token_counter.most_common():
        if token.startswith(('VAR_', 'FUNC_')) and count >= min_freq:
            semantic_tokens.append((token, count))
    
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
    
    # Show problematic unknown tokens
    if unknown_counter:
        print(f"\nTop unknown tokens:")
        for token, count in unknown_counter.most_common(10):
            token_type = "semantic" if token.startswith(('VAR_', 'FUNC_')) else "regular"
            noise_flag = " (noise)" if is_noise_token(token) else ""
            print(f"   '{token}': {count} ({token_type}){noise_flag}")
    
    return coverage

if __name__ == "__main__":
    base_path = os.path.join(ROOT_DIR, "data", "preprocessed", "token")
    output_dir = os.path.join(ROOT_DIR, "data", "preprocessed", "vocab")
    os.makedirs(output_dir, exist_ok=True)
    
    files = [
        os.path.join(base_path, "train.jsonl"),
        os.path.join(base_path, "val.jsonl")    
    ]
    existing_files = [f for f in files if os.path.exists(f)]
    
    if not existing_files:
        print("❌ No tokenized files found!")
        print(f"   Expected files: {files}")
        exit(1)
    
    print(f"Found {len(existing_files)} input files")

    # Build vocabulary with semantic support
    vocab, token_counter = build_vocabulary(
        existing_files, 
        min_freq=2, 
        max_vocab_size=30000
    )
    
    # Validate coverage
    coverage = validate_vocabulary_detailed(vocab, existing_files)
    
    # Save vocab file
    vocab_path = os.path.join(output_dir, "vocab.json")
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2)
    
    print(f"\n✅ Vocabulary completed!")
    print(f"   Size: {len(vocab):,} tokens")
    print(f"   Coverage: {coverage:.2%}")
    print(f"   Semantic tokens included: ✅")
    print(f"   Saved to: {vocab_path}")
