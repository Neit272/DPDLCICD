
import os
import json
import argparse
import numpy as np
from collections import Counter

def analyze_jsonl_file(file_path):
    """Analyzes a single .jsonl file for sequence length and label distribution."""
    sequences_lengths = []
    labels = []
    line_count = 0

    print(f"\n--- Analyzing: {os.path.basename(file_path)} ---")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'input_ids' in data and 'target' in data:
                        sequences_lengths.append(len(data['input_ids']))
                        labels.append(data['target'])
                        line_count += 1
                    else:
                        print(f"Warning: Skipping line {line_count + 1} due to missing 'input_ids' or 'target'.")
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON on line {line_count + 1}.")

        if not sequences_lengths:
            print("No valid data found in the file.")
            return

        # --- Statistics ---
        label_counts = Counter(labels)
        total_samples = len(labels)

        print(f"Total Samples: {total_samples}")

        # Label Distribution
        print("Label Distribution:")
        for label, count in sorted(label_counts.items()):
            percentage = (count / total_samples) * 100
            print(f"  - Label {label}: {count} samples ({percentage:.2f}%)")

        # Sequence Length Statistics
        print("Sequence Length Stats:")
        print(f"  - Min Length:    {np.min(sequences_lengths)}")
        print(f"  - Max Length:    {np.max(sequences_lengths)}")
        print(f"  - Avg Length:    {np.mean(sequences_lengths):.2f}")
        
        # Check how many sequences would be truncated by a MAX_LEN of 256
        truncated_count = sum(1 for length in sequences_lengths if length > 256)
        if truncated_count > 0:
            percentage_truncated = (truncated_count / total_samples) * 100
            print(f"\n! Alert: {truncated_count} samples ({percentage_truncated:.2f}%) have length > 256.")
        else:
            print("\n> Info: All sequences have length <= 256.")


    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze .jsonl data files for ML model training.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the .jsonl files.')
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"Error: Directory not found: {args.data_dir}")
    else:
        for filename in sorted(os.listdir(args.data_dir)):
            if filename.endswith('.jsonl'):
                file_path = os.path.join(args.data_dir, filename)
                analyze_jsonl_file(file_path)
