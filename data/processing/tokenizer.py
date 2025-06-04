import os
import sys
import json
import re
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

# Enhanced tokenizer pattern - handles complex cases better
pattern = r'''
    \b[a-zA-Z_][a-zA-Z0-9_]*(?:_[a-zA-Z0-9_]+)*\b  # Complex identifiers
  | \b0[xX][0-9a-fA-F]+[uUlL]*\b                    # Hex numbers
  | \b\d+\.?\d*[fFlLuU]*\b                          # Numbers with suffixes
  | \+\+|\-\-|\+=|\-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>= # Compound operators
  | ==|!=|<=|>=|<<|>>|&&|\|\||->|\.\*              # Multi-char operators
  | [\[\](){};.,:<>+\-*/%=&|^~!?]                  # Single-char symbols
  | "(?:[^"\\]|\\.)*"                              # String literals
  | '(?:[^'\\]|\\.)*'                              # Character literals
'''

tokenizer = RegexpTokenizer(pattern, flags=re.VERBOSE)

# Expanded important functions - covers more libraries
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

C_KEYWORDS = {
    "if", "else", "for", "while", "switch", "case", "break", "continue",
    "return", "sizeof", "struct", "typedef", "static", "const", "goto",
    "do", "default", "enum", "extern", "volatile", "inline", "union",
    "auto", "register", "restrict"
}

C_TYPES = {
    "int", "char", "float", "double", "void", "long", "short", "unsigned", 
    "signed", "bool", "size_t", "ssize_t", "FILE", "NULL", "nullptr",
    "int8_t", "int16_t", "int32_t", "int64_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "off_t", "pid_t", "time_t"
}

OPERATORS = {
    '+', '-', '*', '/', '%', '=', '==', '!=', '<', '>', '<=', '>=',
    '&&', '||', '!', '&', '|', '^', '~', '++', '--', '+=', '-=',
    '->', '.', ',', ';', ':', '(', ')', '{', '}', '[', ']', '?'
}

# Common meaningful variables/constants
MEANINGFUL_IDENTIFIERS = {
    "i", "j", "k", "n", "x", "y", "z", "p", "s", "c", "a", "b", "t", "m", "h", "w",
    "len", "size", "count", "index", "offset", "ptr", "buf", "data", "tmp", "ret",
    "err", "fd", "fp", "ctx", "info", "type", "flag", "mode", "status"
}

PRESERVE_TOKENS = CRITICAL_FUNCTIONS | C_KEYWORDS | C_TYPES | OPERATORS | MEANINGFUL_IDENTIFIERS

def is_complex_identifier(token):
    """Enhanced check for complex identifiers"""
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token):
        return False
    
    # Has underscores and meaningful structure
    if '_' in token and len(token) > 3:
        parts = token.split('_')
        if len(parts) >= 2 and all(len(part) >= 2 for part in parts):
            return True
    
    # CamelCase patterns
    if re.search(r'[a-z][A-Z]', token) and len(token) > 4:
        return True
    
    # Function-like patterns
    if any(pattern in token.lower() for pattern in [
        'init', 'create', 'destroy', 'alloc', 'free', 'get', 'set',
        'read', 'write', 'open', 'close', 'connect', 'bind', 'send', 'recv'
    ]):
        return True
    
    return False

def is_noise_token(token):
    """Enhanced noise detection"""
    if token in PRESERVE_TOKENS or is_complex_identifier(token):
        return False
    
    # Definitive garbage patterns
    noise_patterns = [
        r'^[A-Z]\d+$',                    # C1, F1, T2
        r'^[a-z][A-Z]$',                  # yC, xM, vC
        r'^[A-Z]{2,4}$',                  # BB, UV, KB, XYZW
        r'^[a-z]{1,2}[A-Z]$',             # yM, abC
        r'^[A-Z][a-z]$',                  # To, We
        r'^[a-z]\d+$',                    # x1, y2, z3
        r'^tmp\d*$',                      # tmp, tmp1, tmp2
        r'^var\d*$',                      # var, var1, var2
        r'^[a-zA-Z]{1,2}\d{1,3}$'         # a1, xy12, A3
    ]
    
    for pattern in noise_patterns:
        if re.match(pattern, token):
            return True
    
    # Very short tokens (except meaningful ones)
    if len(token) <= 2 and token not in MEANINGFUL_IDENTIFIERS:
        return True
    
    return False

def is_noise_string(token):
    """Filter out meaningless string literals"""
    if not (token.startswith('"') and token.endswith('"')):
        return False
    
    content = token[1:-1]
    
    # Empty or whitespace only
    if len(content) == 0 or re.match(r'^[\s\n\t\r]*$', content):
        return True
    
    # Format strings only
    if re.match(r'^%[sdxoefgc%\d\.\-\+\s]*$', content):
        return True
    
    # Very long strings (likely file paths or error messages)
    if len(content) > 50:
        return True
    
    # File paths
    if '/' in content or '\\' in content:
        return True
    
    # URLs
    if any(proto in content.lower() for proto in ['http', 'www', '.com']):
        return True
    
    return False

def smart_normalize(tokens):
    """Improved normalization with better preservation"""
    var_map, func_map = {}, {}
    var_count = func_count = 1
    result = []

    for i, tok in enumerate(tokens):
        if not tok or not tok.strip():
            continue
        
        # Filter noise
        if is_noise_token(tok) or is_noise_string(tok):
            continue
        
        # Always preserve important tokens
        if tok in PRESERVE_TOKENS or is_complex_identifier(tok):
            result.append(tok)
            continue
        
        # Handle identifiers
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', tok):
            is_function_call = (i + 1 < len(tokens) and tokens[i + 1] == "(")
            
            if is_function_call:
                # Only normalize very long function names
                if len(tok) > 12:
                    if tok not in func_map:
                        func_map[tok] = f"FUNC_{func_count}"
                        func_count += 1
                    result.append(func_map[tok])
                else:
                    result.append(tok)
            else:
                # Variable - only normalize very long ones
                if len(tok) > 15:
                    if tok not in var_map:
                        var_map[tok] = f"VAR_{var_count}"
                        var_count += 1
                    result.append(var_map[tok])
                else:
                    result.append(tok)
        else:
            # Keep numbers, operators, etc.
            result.append(tok)
    
    return result

def tokenize_file(input_path, output_path):
    """Enhanced file tokenization"""
    processed = skipped = 0
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in tqdm(fin, desc=f"Processing {os.path.basename(input_path)}"):
            try:
                obj = json.loads(line)
                code = obj['func']
                
                # Clean comments but preserve structure
                code = re.sub(r'//.*', '', code)
                code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
                
                # Tokenize and normalize
                tokens = tokenizer.tokenize(code)
                tokens = smart_normalize(tokens)
                
                # Quality check - ensure meaningful content
                if len(tokens) >= 5:
                    result = {'tokens': tokens, 'target': obj['target']}
                    fout.write(json.dumps(result) + '\n')
                    processed += 1
                else:
                    skipped += 1
                    
            except (json.JSONDecodeError, KeyError):
                skipped += 1
                continue
    
    print(f"   Processed: {processed:,}, Skipped: {skipped:,}")

if __name__ == "__main__":
    base_in = os.path.join(ROOT_DIR, "data", "preprocessed")
    base_out = os.path.join(ROOT_DIR, "data", "preprocessed", "token")
    os.makedirs(base_out, exist_ok=True)

    print("Starting improved tokenization...")
    
    for split in ["train", "val", "test"]:
        in_file = os.path.join(base_in, f"{split}.jsonl")
        out_file = os.path.join(base_out, f"{split}.jsonl")
        
        if os.path.exists(in_file):
            print(f"\nProcessing {split.upper()}:")
            tokenize_file(in_file, out_file)

    print("\nTokenization completed!")