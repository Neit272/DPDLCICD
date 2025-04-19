import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.self_attention.model import TransformerClassifier
from tqdm import tqdm

# ===== Dataset class =====
class VulDataset(Dataset):
    def __init__(self, path):
        self.samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                self.samples.append((obj["input_ids"], obj["target"]))
                
        self.samples = self.samples[:1000] # for testing


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids, label = self.samples[idx]
        return torch.tensor(input_ids), torch.tensor(label)

# ===== Accuracy =====
def accuracy(preds, labels):
    return (preds.argmax(dim=1) == labels).float().mean().item()

# ===== Evaluation =====
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            total_acc += accuracy(logits, y)
    return total_loss / len(dataloader), total_acc / len(dataloader)

# ===== Train loop =====
def train(model, train_loader, val_loader, epochs, lr, device, save_path):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"[Epoch {epoch+1}] Train loss: {total_loss/len(train_loader):.4f} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")

        # save model mỗi epoch
        torch.save(model.state_dict(), os.path.join(save_path, f"epoch{epoch+1}.pt"))

# ===== Main entry =====
if __name__ == "__main__":
    # === Config ===
    data_dir = "data/preprocessed/encoded"
    vocab_path = "data/preprocessed/vocab/vocab.json"
    save_path = "models/checkpoint/"
    os.makedirs(save_path, exist_ok=True)

    batch_size = 8  # for testing
    epochs = 1      # for testing
    lr = 1e-4
    max_len = 256
    d_model = 128
    num_heads = 4
    num_layers = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load vocab size ===
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)

    # === Load datasets ===
    train_set = VulDataset(os.path.join(data_dir, "train.jsonl"))
    val_set = VulDataset(os.path.join(data_dir, "val.jsonl"))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # === Init model ===
    model = TransformerClassifier(
        vocab_size=vocab_size,
        d_model=d_model,
        heads=num_heads,
        layers=num_layers,
        max_len=max_len,
        num_classes=2
    ).to(device)

    # === Train ===
    train(model, train_loader, val_loader, epochs, lr, device, save_path)
