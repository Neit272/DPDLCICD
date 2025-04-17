import json
import os
from sklearn.model_selection import train_test_split

def clean_code(code):
    """
    Clean C code: remove comments, normalize indent.
    Supports both // and /* */.
    """
    code = code.replace('\t', '    ')
    lines = code.split('\n')
    cleaned_lines = []
    in_block_comment = False

    for line in lines:
        line = line.strip()
        if '/*' in line:
            in_block_comment = True
        if not in_block_comment and not line.startswith('//'):
            cleaned_lines.append(line)
        if '*/' in line:
            in_block_comment = False

    return '\n'.join(cleaned_lines)

def preprocess_dataset(input_path, output_dir, test_size=0.2, val_size=0.1):
    """
    Preprocess a .json file but format like .jsonl file: clean func field, split into train/val/test, save.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            if 'func' not in sample:
                continue
            sample['func'] = clean_code(sample['func'])
            data.append(sample)

    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=42)

    os.makedirs(output_dir, exist_ok=True)
    for name, subset in [('train', train_data), ('val', val_data), ('test', test_data)]:
        path = os.path.join(output_dir, f'{name}.jsonl')
        with open(path, 'w', encoding='utf-8') as f:
            for item in subset:
                f.write(json.dumps(item) + '\n')

    print(f"Saved {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples to {output_dir}")

if __name__ == "__main__":
    input_path = "../datasets/diversevul_20230702.json"
    output_dir = "../preprocessed"
    preprocess_dataset(input_path, output_dir)
