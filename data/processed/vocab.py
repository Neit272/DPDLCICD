import json
from collections import Counter

input_tokens_path = 'D:\\Project_CICD\\Deep-Pentest-using-ML-DL-in-CI-CD\\data\\processed\\func_tokens2.json'
output_vocab_path = 'D:\\Project_CICD\\Deep-Pentest-using-ML-DL-in-CI-CD\\data\\processed\\vocab.json'

# Read file containing token lists
with open(input_tokens_path, 'r', encoding='utf-8') as infile:
    all_token_lists = json.load(infile)  # all_token_lists là list[list[str]]

# Count frequency of token
token_counter = Counter()
for token_list in all_token_lists:
    token_counter.update(token_list)

# Make vocab: Sort by descending frequency
# add <PAD>, <UNK> if using in training model
vocab = {"<PAD>": 0, "<UNK>": 1}
idx = 2

for token, _ in token_counter.most_common():
    vocab[token] = idx
    idx += 1

with open(output_vocab_path, 'w', encoding='utf-8') as outfile:
    json.dump(vocab, outfile, indent=2, ensure_ascii=False)

print(f"Finished! Total of token: {len(vocab)}")
