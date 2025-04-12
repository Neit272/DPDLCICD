import json
from nltk.tokenize import RegexpTokenizer
import re

code_tokenizer = RegexpTokenizer(r'''
    "(?:\\.|[^"\\])*"          | # String literals
    '(?:\\.|[^'\\])*'          | # Char literals
    \b\d+\.?\d*\b              | # Numbers
    \b0x[0-9a-fA-F]+\b         | # Hex numbers
    \b[a-zA-Z_][a-zA-Z0-9_]*\b | # Identifiers
    \+\+|\-\-|\+=|\-=|\*=|/=   | # Compound operators
    ==|!=|<=|>=                | # Comparisons
    &&|\|\|                    | # Logical operators
    [\[\](){},.;+\-*/%&|^~!]     # Single chars
''', flags=re.VERBOSE)

def tokenize_code(code):
    try:
        return code_tokenizer.tokenize(code)
    except Exception as e:
        print(f"Lỗi khi tokenize: {e}")
        return []

input_file = "D:\\Project_CICD\\Deep-Pentest-using-ML-DL-in-CI-CD\\data\\processed\\diversevul_clean.json"
output_file_tokens_only = "D:\\Project_CICD\\Deep-Pentest-using-ML-DL-in-CI-CD\\data\\processed\\func_tokens2.json"

func_tokens_list = []

with open(input_file, 'r', encoding='utf-8') as f_in:
    for line in f_in:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            func_tokens = tokenize_code(data["func"])
            func_tokens_list.append(func_tokens)
        except json.JSONDecodeError as e:
            print(f"[JSON ERROR] {e} – Dòng: {line[:50]}...")

with open(output_file_tokens_only, 'w', encoding='utf-8') as f_out:
    json.dump(func_tokens_list, f_out, indent=2, ensure_ascii=False)

print("Finished tokenizing!")
