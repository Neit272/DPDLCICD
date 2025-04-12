import json
import re
from tqdm import tqdm

def clean_code(func_code):
    """Làm sạch mã nguồn trong trường `func`"""
    # Xóa block comments (/* ... */)
    func_code = re.sub(r'/\*.*?\*/', '', func_code, flags=re.DOTALL)
    # Xóa line comments (// ...)
    func_code = re.sub(r'//.*?$', '', func_code, flags=re.MULTILINE)
    # Chuẩn hóa khoảng trắng
    func_code = re.sub(r'\s+', ' ', func_code).strip()
    return func_code

def process_json_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
               
        for line in tqdm(f_in, desc="Processing"):
            try:
                # Parse JSON từng dòng
                sample = json.loads(line.strip())
                
                # Chỉ làm sạch trường `func` nếu tồn tại
                if 'func' in sample:
                    sample['func'] = clean_code(sample['func'])
                
                # Ghi lại dòng đã xử lý
                f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            except json.JSONDecodeError as e:
                print(f"Lỗi JSON ở dòng: {line[:50]}... | Chi tiết: {e}")
                continue

process_json_file("D:\\Project_CICD\\Deep-Pentest-using-ML-DL-in-CI-CD\\data\\processed\\diversevul_20230702.json", "D:\\Project_CICD\\Deep-Pentest-using-ML-DL-in-CI-CD\\data\\processed\\diversevul_clean.json")