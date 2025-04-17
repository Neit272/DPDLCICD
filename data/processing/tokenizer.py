import json
import os
import re
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer

pattern = r'''
    "(?:\\.|[^"\\])*"             # String literals
  | '(?:\\.|[^'\\])*'             # Char literals
  | \b0x[0-9a-fA-F]+\b            # Hex numbers
  | \b\d+\.?\d*\b                 # Decimal numbers
  | \b[a-zA-Z_][a-zA-Z0-9_]*\b    # Identifiers / keywords
  | \+\+ | \-\- | \+= | \-= | \*= | /=  # Compound operators
  | == | != | <= | >= | ->        # Comparisons & pointer ops
  | && | \|\| | & | \*            # Logical ops, address, pointer
  | [\[\](){};.,:<>+\-*/%=&|^~!]  # Single-char symbols
'''

tokenizer = RegexpTokenizer(pattern, flags=re.VERBOSE)

def tokenize_c_code(code):
    return tokenizer.tokenize(code)

def tokenize_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin, desc=f"Tokenizing {os.path.basename(input_path)}"):
            obj = json.loads(line)
            if 'func' not in obj:
                continue
            tokens = tokenize_c_code(obj['func'])
            out = {
                'tokens': tokens,
                'target': obj['target']
            }
            fout.write(json.dumps(out) + '\n')

if __name__ == "__main__":
    base_in = "../preprocessed"
    base_out = "../preprocessed/token"
    os.makedirs(base_out, exist_ok=True)

    for split in ["train", "val", "test"]:
        in_file = os.path.join(base_in, f"{split}.jsonl")
        out_file = os.path.join(base_out, f"{split}.jsonl")
        tokenize_file(in_file, out_file)
