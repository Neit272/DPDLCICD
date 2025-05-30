import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import torch
from torch.utils.data import Dataset, DataLoader
from models.self_attention.model import TransformerClassifier
from tqdm import tqdm

# === Hyperparams ===
BATCH_SIZE = 32
EPOCHS = 2
MAX_LEN = 256
LR = 1e-4

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed', 'encoded'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

# === Dataset (streaming-friendly) ===
class VulnDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.offsets = []
        with open(self.path, 'r', encoding='utf-8') as f:
            offset = f.tell()
            line = f.readline()
            while line:
                self.offsets.append(offset)
                offset = f.tell()
                line = f.readline()

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        with open(self.path, 'r', encoding='utf-8') as f:
            f.seek(self.offsets[idx])
            item = json.loads(f.readline())
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long).to(DEVICE),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.bool).to(DEVICE),
            'label': torch.tensor(item['target'], dtype=torch.long).to(DEVICE)
        }

# === Training ===
def train():
    train_set = VulnDataset(os.path.join(DATA_DIR, 'train.jsonl'))
    val_set = VulnDataset(os.path.join(DATA_DIR, 'val.jsonl'))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=4)

    # Estimate vocab size from just the first 100 samples for speed
    vocab_size = 0
    for i in range(min(100, len(train_set))):
        vocab_size = max(vocab_size, max(train_set[i]['input_ids']))
    vocab_size += 1

    model = TransformerClassifier(vocab_size=vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} - Train"):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} train loss: {total_loss / len(train_loader):.4f}")

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} - Eval"):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['label']
                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total * 100
        print(f"Epoch {epoch+1} val acc: {acc:.2f}%")

if __name__ == '__main__':
    train()
