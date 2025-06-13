import os
import sys
import json
import re
import glob
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from typing import Dict, List, Tuple, Optional

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

# Enhanced tokenizer pattern
TOKENIZER_PATTERN = r'''
    \b[a-zA-Z_][a-zA-Z0-9_]*(?:_[a-zA-Z0-9_]+)*\b  # Complex identifiers
  | \b0[xX][0-9a-fA-F]+[uUlL]*\b                    # Hex numbers
  | \b\d+\.?\d*[fFlLuU]*\b                          # Numbers with suffixes
  | \+\+|\-\-|\+=|\-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>= # Compound operators
  | ==|!=|<=|>=|<<|>>|&&|\|\||->|\.\*              # Multi-char operators
  | [\[\](){};.,:<>+\-*/%=&|^~!?]                  # Single-char symbols
  | "(?:[^"\\]|\\.)*"                              # String literals
  | '(?:[^'\\]|\\.)*'                              # Character literals
'''

tokenizer = RegexpTokenizer(TOKENIZER_PATTERN, flags=re.VERBOSE)

# Variable type patterns
VARIABLE_TYPE_PATTERNS = {
    'PTR': [r'\*\s*\w+', r'\w+\s*\*', r'\w+_ptr\b', r'\w+_p\b', r'p[A-Z]\w*', r'\w*[Pp]ointer\w*', r'\w*[Bb]uf(?:fer)?\w*'],
    'INT': [r'\bint\s+\w+', r'\b(size|len|count|num|index|offset|id|fd)\w*\b', r'\b\w*_(size|len|count|num|index|id)\b', r'\bi[A-Z]\w*', r'\bn[A-Z]\w*', r'\w*[Ii]ndex\w*', r'\w*[Cc]ount\w*', r'\w*[Ss]ize\w*'],
    'STR': [r'\bchar\s+\w+', r'\b\w*_(name|str|text|msg|path|file)\b', r'\bsz\w*', r'\b\w*_buf\b', r'\w*[Ss]tring\w*', r'\w*[Tt]ext\w*', r'\w*[Nn]ame\w*'],
    'STRUCT': [r'\bstruct\s+\w+', r'\b\w+_(info|ctx|data|node|item|entry)\b', r'\b\w+_t\s+\w+', r'\w*[Ii]nfo\w*', r'\w*[Cc]ontext\w*', r'\w*[Nn]ode\w*'],
    'BOOL': [r'\bbool\s+\w+', r'\bis_\w+', r'\bhas_\w+', r'\bcan_\w+', r'\benable\w*', r'\w*[Ff]lag\w*'],
    'STATUS': [r'\b\w*_(ret|err|status|result|code)\b', r'\berr\w*', r'\bret\w*', r'\w*[Ss]tatus\w*', r'\w*[Rr]esult\w*']
}

# Function return type patterns
FUNCTION_RETURN_PATTERNS = {
    'VOID': ['init', 'destroy', 'free', 'close', 'print', 'set_', 'write_', 'clear', 'reset', 'update', 'add_', 'remove_', 'delete_', 'cleanup', 'setup', 'configure'],
    'PTR': ['alloc', 'malloc', 'create', 'get_', 'find_', 'search_', 'new_', 'copy_', 'clone_', 'acquire_', 'obtain_'],
    'BOOL': ['is_', 'has_', 'can_', 'check_', 'validate_', 'compare', 'test_', 'verify_', 'match_', 'equals_'],
    'INT': ['count_', 'size_', 'len_', 'read_', 'write_', 'send_', 'recv_', 'process_', 'parse_', 'calculate_', 'compute_'],
    'STATUS': ['open_', 'connect_', 'bind_', 'listen_', 'exec', 'run_', 'start_', 'stop_', 'load_', 'save_']
}

# Context indicators
CONTEXT_INDICATORS = {
    'PTR': ['*', '->', 'malloc', 'alloc', 'ptr', '&'],
    'STR': ['strcpy', 'strlen', 'sprintf', '"', 'strcat', 'strcmp'],
    'INT': ['++', '--', '+=', 'for', 'while', 'sizeof', '+', '-'],
    'STRUCT': ['struct', '.', 'typedef', '->'],
    'BOOL': ['if', '&&', '||', '!', 'true', 'false']
}

# Preserved tokens
CRITICAL_FUNCTIONS = {
    # Memory management
    "malloc", "calloc", "realloc", "free", "alloca", "mmap", "munmap",
    
    # String operations (vulnerable)
    "strcpy", "strncpy", "strcat", "strncat", "strlen", "strcmp", "strncmp",
    "strchr", "strstr", "strtok", "strdup",
    
    # Memory operations (vulnerable)
    "memcpy", "memmove", "memset", "memcmp", "memchr",
    
    # I/O operations (vulnerable)
    "printf", "sprintf", "snprintf", "fprintf", "scanf", "sscanf", "fscanf",
    "gets", "fgets", "puts", "fputs", "fopen", "fclose", "fread", "fwrite",
    
    # System calls (dangerous)
    "system", "exec", "execl", "execv", "popen", "pclose", "fork", "exit",
    
    # SSL/TLS functions
    "SSL_CTX_new", "SSL_CTX_free", "SSL_CTX_set_mode", "SSL_CTX_set_options",
    "SSL_new", "SSL_free", "SSL_connect", "SSL_accept", "SSL_read", "SSL_write",
    "BIO_new", "BIO_free", "BIO_read", "BIO_write", "BIO_s_mem",
    "EVP_CIPHER_CTX_new", "EVP_EncryptInit_ex", "EVP_DecryptInit_ex",
    
    # Database functions
    "mysql_query", "mysql_real_query", "mysql_connect", "mysql_close",
    "PQexec", "PQconnectdb", "PQfinish", "PQprepare",
    "sqlite3_open", "sqlite3_prepare", "sqlite3_step", "sqlite3_finalize",
    
    # Conversion functions (vulnerable)
    "atoi", "atol", "atof", "strtol", "strtoul", "strtod"
}

C_KEYWORDS = {"if", "else", "for", "while", "switch", "case", "break", "continue", "return", "sizeof", "struct", "typedef", "static", "const", "goto", "do", "default", "enum", "extern", "volatile", "inline", "union", "auto", "register", "restrict"}

C_TYPES = {"int", "char", "float", "double", "void", "long", "short", "unsigned", "signed", "bool", "size_t", "ssize_t", "FILE", "NULL", "nullptr", "int8_t", "int16_t", "int32_t", "int64_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t", "off_t", "pid_t", "time_t"}

OPERATORS = {'+', '-', '*', '/', '%', '=', '==', '!=', '<', '>', '<=', '>=', '&&', '||', '!', '&', '|', '^', '~', '++', '--', '+=', '-=', '->', '.', ',', ';', ':', '(', ')', '{', '}', '[', ']', '?'}

MEANINGFUL_IDENTIFIERS = {"i", "j", "k", "n", "x", "y", "z", "p", "s", "c", "a", "b", "t", "m", "h", "w", "len", "size", "count", "index", "offset", "ptr", "buf", "data", "tmp", "ret", "err", "fd", "fp", "ctx", "info", "type", "flag", "mode", "status"}

PRESERVE_TOKENS = CRITICAL_FUNCTIONS | C_KEYWORDS | C_TYPES | OPERATORS | MEANINGFUL_IDENTIFIERS

def infer_variable_type(token: str, context_tokens: List[str], position: int) -> str:
    for var_type, patterns in VARIABLE_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, token, re.IGNORECASE):
                return var_type
    
    context_window = 5
    start_idx = max(0, position - context_window)
    end_idx = min(len(context_tokens), position + context_window + 1)
    context_str = ' '.join(context_tokens[start_idx:end_idx]).lower()
    
    type_scores = {}
    for var_type, indicators in CONTEXT_INDICATORS.items():
        score = sum(1 for indicator in indicators if indicator in context_str)
        type_scores[var_type] = score
    
    if type_scores and max(type_scores.values()) > 0:
        return max(type_scores, key=type_scores.get)
    
    return 'GENERIC'

def infer_function_return_type(func_name: str, context_tokens: List[str]) -> str:
    func_lower = func_name.lower()
    
    for return_type, patterns in FUNCTION_RETURN_PATTERNS.items():
        for pattern in patterns:
            if pattern in func_lower:
                return return_type
    
    context_str = ' '.join(context_tokens).lower()
    
    if any(indicator in context_str for indicator in ['return', '=', 'if', 'while']):
        if any(indicator in context_str for indicator in ['null', 'nil', '0']):
            return 'PTR'
        elif any(indicator in context_str for indicator in ['true', 'false']):
            return 'BOOL'
        elif any(indicator in context_str for indicator in ['error', 'err', 'fail']):
            return 'STATUS'
    
    return 'GENERIC'

def is_complex_identifier(token: str) -> bool:
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token):
        return False
    
    if '_' in token and len(token) > 6:
        parts = token.split('_')
        if len(parts) >= 2 and all(len(part) >= 2 for part in parts):
            return True
    
    if re.search(r'[a-z][A-Z]', token) and len(token) > 6:
        return True
    
    if any(pattern in token.lower() for pattern in ['init', 'create', 'destroy', 'alloc', 'free', 'get', 'set', 'read', 'write', 'open', 'close', 'connect', 'send', 'recv', 'ssl_', 'http_', 'mysql_', 'sqlite_', 'crypto_']):
        return True
    
    return False

def is_noise_token(token: str) -> bool:
    if token in PRESERVE_TOKENS or is_complex_identifier(token):
        return False
    
    noise_patterns = [r'^[A-Z]\d+$', r'^[a-z][A-Z]$', r'^[A-Z]{2,4}$', r'^[a-z]{1,2}[A-Z]$', r'^[A-Z][a-z]$', r'^[a-z]\d+$', r'^tmp\d*$', r'^[a-z]$', r'^[A-Z]_\d+$']
    
    for pattern in noise_patterns:
        if re.match(pattern, token):
            return True
    
    return False

def is_noise_string(s: str) -> bool:
    if not s.startswith('"') or not s.endswith('"'):
        return False
    
    content = s[1:-1]
    
    if len(content) <= 1:
        return True
    
    if re.match(r'^[^a-zA-Z0-9]+$', content):
        return True
    
    if any(pattern in content.lower() for pattern in ['debug', 'trace', '%d', '%s', '%x', 'printf']):
        return True
    
    return False

def smart_normalize_semantic(tokens: List[str]) -> List[str]:
    var_map, func_map = {}, {}
    result = []

    for i, tok in enumerate(tokens):
        if not tok or not tok.strip():
            continue
        
        if is_noise_token(tok) or is_noise_string(tok):
            continue
        
        if tok in PRESERVE_TOKENS or is_complex_identifier(tok):
            result.append(tok)
            continue
        
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', tok):
            context_start = max(0, i - 3)
            context_end = min(len(tokens), i + 4)
            context = tokens[context_start:context_end]
            
            is_function_call = (i + 1 < len(tokens) and tokens[i + 1] == "(")
            
            if is_function_call:
                if len(tok) > 12:
                    return_type = infer_function_return_type(tok, context)
                    semantic_name = f"FUNC_{return_type}"
                    
                    if tok not in func_map:
                        func_map[tok] = semantic_name
                    result.append(func_map[tok])
                else:
                    result.append(tok)
            else:
                if len(tok) > 15:
                    var_type = infer_variable_type(tok, context, i)
                    semantic_name = f"VAR_{var_type}"
                    
                    if tok not in var_map:
                        var_map[tok] = semantic_name
                    result.append(var_map[tok])
                else:
                    result.append(tok)
        else:
            result.append(tok)
    
    return result

def tokenize_file_semantic(input_path: str, output_path: str):
    processed = skipped = 0
    
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in tqdm(fin, desc=f"Processing {os.path.basename(input_path)}"):
            try:
                # Clean line for malformed JSON
                line = line.strip()
                if not line:
                    continue
                    
                clean_line = ''.join(char for char in line if ord(char) >= 32 or char in '\t\n\r')
                obj = json.loads(clean_line)
                
                code = obj['func']
                
                code = re.sub(r'//.*', '', code)
                code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
                
                tokens = tokenizer.tokenize(code)
                tokens = smart_normalize_semantic(tokens)
                
                if len(tokens) >= 5:
                    result = {'tokens': tokens, 'target': obj['target']}
                    fout.write(json.dumps(result) + '\n')
                    processed += 1
                else:
                    skipped += 1
                    
            except (json.JSONDecodeError, KeyError):
                skipped += 1
                continue
    
    return processed, skipped

def main():
    print("🚀 BALANCED TOKENIZER")
    print("=" * 50)
    
    # Configuration - only balanced mode
    balanced_dir = os.path.join(ROOT_DIR, "data", "preprocessed", "balanced_splits")
    tokenized_dir = os.path.join(ROOT_DIR, "data", "preprocessed", "balanced_tokenized")
    
    os.makedirs(tokenized_dir, exist_ok=True)
    
    # Check if balanced_splits exists and has files
    if not os.path.exists(balanced_dir):
        print("❌ balanced_splits directory not found!")
        print(f"Expected location: {balanced_dir}")
        return
    
    balanced_files = [f for f in os.listdir(balanced_dir) if f.endswith('.jsonl')]
    if not balanced_files:
        print("❌ No .jsonl files found in balanced_splits directory!")
        return
    
    print(f"✅ Found balanced_splits directory with {len(balanced_files)} files")
    print("=" * 50)
    
    # Process balanced files
    input_files = []
    
    # Add val and test files if they exist
    for split in ['val_balanced.jsonl', 'test_balanced.jsonl']:
        input_path = os.path.join(balanced_dir, split)
        if os.path.exists(input_path):
            # Output name without '_balanced' suffix  
            output_name = split.replace('_balanced', '')
            input_files.append((split, output_name, balanced_dir))
    
    # Add all train files
    train_pattern = os.path.join(balanced_dir, "train_balanced_*.jsonl")
    train_files = sorted(glob.glob(train_pattern))
    
    for train_file in train_files:
        filename = os.path.basename(train_file)
        # Keep original filename for train files
        input_files.append((filename, filename, balanced_dir))
    
    print(f"📊 Found {len(input_files)} balanced files to process:")
    for inp, out, _ in input_files:
        print(f"   {inp} -> {out}")
    
    # Process files
    total_processed = total_skipped = 0
    successful_files = 0
    
    for input_file, output_file, source_dir in input_files:
        input_path = os.path.join(source_dir, input_file)
        output_path = os.path.join(tokenized_dir, output_file)
        
        if os.path.exists(input_path):
            print(f"\n🔄 Processing {input_file}...")
            processed, skipped = tokenize_file_semantic(input_path, output_path)
            print(f"   ✅ {processed:,} processed, {skipped:,} skipped")
            
            total_processed += processed
            total_skipped += skipped
            successful_files += 1
        else:
            print(f"   ❌ File not found: {input_path}")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"✅ BALANCED TOKENIZATION COMPLETED!")
    print(f"📁 Output directory: {tokenized_dir}")
    print(f"📊 Results:")
    print(f"   Files processed: {successful_files}/{len(input_files)}")
    print(f"   Total samples processed: {total_processed:,}")
    print(f"   Total samples skipped: {total_skipped:,}")

if __name__ == "__main__":
    main()