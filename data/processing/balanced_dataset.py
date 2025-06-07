import json
from typing import Counter
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import numpy as np

def smart_balance_with_smote():
    
    print("Using SMOTE for intelligent balancing...")
    
    # Load original train data
    input_file = "../../data/preprocessed/encoded/train.jsonl"
    
    X_train = []
    y_train = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading train data"):
            try:
                obj = json.loads(line)
                X_train.append(obj.get('input_ids', []))
                y_train.append(obj.get('target', 0))
            except json.JSONDecodeError:
                continue
    
    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"Original shape: {X_train.shape}")
    print(f"Label distribution: {Counter(y_train)}")
    
    # Apply SMOTE + Tomek (removes borderline samples)
    smote_tomek = SMOTETomek(
        sampling_strategy=0.4,  # 40% vulnerable after balancing
        random_state=42
    )
    
    X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
    
    print(f"After SMOTE-Tomek:")
    print(f"New shape: {X_resampled.shape}")
    print(f"Label distribution: {Counter(y_resampled)}")
    
    # Save resampled data
    output_file = "../../data/preprocessed/encoded/train_smote.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(len(X_resampled)):
            sample = {
                'input_ids': X_resampled[i].tolist(),
                'target': int(y_resampled[i])
            }
            f.write(json.dumps(sample) + '\n')
    
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    smart_balance_with_smote()
    print("Balancing completed!")