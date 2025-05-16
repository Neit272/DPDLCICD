import json
import os
import re
from tqdm import tqdm


INPUT_PATH = "data/datasets/diversevul_20230702.json"
OUTPUT_PATH = "data/datasets/clean_diversevul_20230702.jsonl"

def remove_comments(code: str) -> str:
    # Remove single-line and multi-line comments (C/C++ style)
    code = re.sub(r"//.*", "", code)                        # single-line
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)  # multi-line
    return code

def normalize_code(code: str) -> str:
    try:
        # Decode escape characters like \n, \t
        code = code.encode().decode("unicode_escape")
    except Exception:
        pass  # fallback if already decoded

    # Remove non-ASCII
    code = ''.join(c for c in code if ord(c) < 128)

    # Remove comments
    code = remove_comments(code)

    # Normalize line spacing
    lines = code.splitlines()
    clean_lines = [line.strip() for line in lines if line.strip()]

    return "\n".join(clean_lines)

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(INPUT_PATH, "r", encoding="utf-8") as f_in, open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
        for line in tqdm(f_in, desc="Cleaning dataset"):
            try:
                obj = json.loads(line)
                raw_code = obj.get("func", "")
                obj["func"] = normalize_code(raw_code)
                f_out.write(json.dumps(obj) + "\n")
            except Exception as e:
                print("Error:", e)
                continue

    print(f"Cleaned dataset saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
