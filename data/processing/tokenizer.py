import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

import json
import os
import re
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from data.processing.semantic_slicer import slice_semantic_blocks

# Tokenizer regex
pattern = r'''
    "(?:\\.|[^"\\])*"             # String literals
  | '(?:\\.|[^'\\])*'              # Char literals
  | \b0x[0-9a-fA-F]+\b               # Hex numbers
  | \b\d+\.?\d*\b                 # Decimal numbers
  | \b[a-zA-Z_][a-zA-Z0-9_]*\b       # Identifiers / keywords
  | \+\+ | \-\- | \+= | \-= | \*= | /=  # Compound operators
  | == | != | <= | >= | ->            # Comparisons & pointer ops
  | && | \|\| | & | \*               # Logical ops, address, pointer
  | [\[\](){};.,:<>+\-*/%=&|^~!]    # Single-char symbols
'''
tokenizer = RegexpTokenizer(pattern, flags=re.VERBOSE)

# Reserved keywords and stdlib functions
C_KEYWORDS = {
    "if", "else", "for", "while", "switch", "case", "break", "continue",
    "return", "sizeof", "struct", "typedef", "static", "const", "goto"
}
C_TYPES = {
    "int", "char", "float", "double", "void", "long", "short", "unsigned", "signed",
    "bool", "size_t", "FILE"
}
STD_FUNCS = {
    "strcpy", "strncpy", "memcpy", "sprintf", "snprintf", "malloc", "calloc",
    "free", "fopen", "fread", "fwrite", "system", "exit", "perror", "printf",
    "scanf", "strlen", "strcat", "strcmp", "atoi", "atof"
}
RESERVED = C_KEYWORDS | C_TYPES | STD_FUNCS

def symbolic_normalize(tokens):
    var_map = {}
    func_map = {}
    new_tokens = []
    var_count = 1
    func_count = 1

    for i, tok in enumerate(tokens):
        if re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", tok):
            if tok in RESERVED:
                new_tokens.append(tok)
            elif i + 1 < len(tokens) and tokens[i + 1] == "(":
                if tok not in func_map:
                    func_map[tok] = f"FUNC_{func_count}"
                    func_count += 1
                new_tokens.append(func_map[tok])
            else:
                if tok not in var_map:
                    var_map[tok] = f"VAR_{var_count}"
                    var_count += 1
                new_tokens.append(var_map[tok])
        else:
            new_tokens.append(tok)

    return new_tokens

def tokenize_c_code(code):
    return tokenizer.tokenize(code)

def tokenize_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin, desc=f"Tokenizing {os.path.basename(input_path)}"):
            obj = json.loads(line)
            if 'func' not in obj:
                continue
            code = obj['func']
            blocks = slice_semantic_blocks(code)
            if not blocks:
                blocks = [code]  # fallback: use whole function
            merged_code = "\n".join(blocks)
            tokens = tokenize_c_code(merged_code)
            tokens = symbolic_normalize(tokens)
            out = {
                'tokens': tokens,
                'target': obj['target']
            }
            fout.write(json.dumps(out) + '\n')

if __name__ == "__main__":
    base_in = os.path.join(ROOT_DIR, "data", "preprocessed")
    base_out = os.path.join(ROOT_DIR, "data", "preprocessed", "token")
    os.makedirs(base_out, exist_ok=True)

    for split in ["train", "val", "test"]:
        in_file = os.path.join(base_in, f"{split}.jsonl")
        out_file = os.path.join(base_out, f"{split}.jsonl")
        tokenize_file(in_file, out_file)
