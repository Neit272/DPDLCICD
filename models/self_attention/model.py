import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import math
import warnings
warnings.filterwarnings('ignore')

# Implement TransformerClassifier
class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention module."""
    
    def __init__(self, d_model, heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % heads == 0
        
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations and split into heads
        Q = self.w_q(x).view(batch_size, seq_len, self.heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply softmax
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear transformation
        output = self.w_o(context)
        
        return output

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Single Transformer block with self-attention and feed-forward."""
    
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention with residual connection and layer norm
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TransformerClassifier(nn.Module):
    """Transformer model for vulnerability detection."""
    
    def __init__(self, vocab_size, d_model=256, heads=8, layers=6, 
                 max_len=256, num_classes=2, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, heads, d_model * 4, dropout)
            for _ in range(layers)
        ])
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Global pooling strategies
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
    
    def forward(self, input_ids):
        # Get sequence length and create embeddings
        seq_len = input_ids.size(1)
        
        # Token embeddings
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Global pooling - combine average and max pooling
        x_transposed = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        
        avg_pool = self.global_avg_pool(x_transposed).squeeze(-1)  # (batch_size, d_model)
        max_pool = self.global_max_pool(x_transposed).squeeze(-1)  # (batch_size, d_model)
        
        # Concatenate pooled features
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # (batch_size, d_model * 2)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
ENCODED_DIR = os.path.join(ROOT_DIR, "data", "preprocessed", "encoded")
VOCAB_PATH = os.path.join(ROOT_DIR, "data", "preprocessed", "vocab", "vocab.json")
MODEL_DIR = os.path.join(ROOT_DIR, "models", "self_attention")
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyperparameters
D_MODEL = 256        # Model dimension
HEADS = 8            # Number of attention heads
LAYERS = 6           # Number of transformer layers
MAX_LEN = 256        # Maximum sequence length
BATCH_SIZE = 32      # Batch size
EPOCHS = 10          # Training epochs
LEARNING_RATE = 1e-4 # Learning rate
DROPOUT = 0.1        # Dropout rate
WARMUP_STEPS = 1000  # Warmup steps for learning rate scheduler
WEIGHT_DECAY = 1e-5  # Weight decay for regularization

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

class VulnerabilityDataset(Dataset):
    """Custom Dataset for vulnerability detection."""
    
    def __init__(self, input_ids, labels):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'label': self.labels[idx]
        }

def validate_data_files():
    """Validate that all required data files exist."""
    required_files = [
        os.path.join(ENCODED_DIR, "train.jsonl"),
        os.path.join(ENCODED_DIR, "val.jsonl"),
        os.path.join(ENCODED_DIR, "test.jsonl"),
        VOCAB_PATH
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    print("✅ All required files found")

def load_data(file_path):
    """Load and validate encoded data."""
    X, y = [], []
    invalid_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc=f"Loading {os.path.basename(file_path)}")):
                try:
                    obj = json.loads(line)
                    input_ids = obj.get('input_ids', [])
                    target = obj.get('target', 0)
                    
                    # Validation
                    if len(input_ids) != MAX_LEN:
                        invalid_count += 1
                        continue
                    
                    if target not in [0, 1]:
                        invalid_count += 1
                        continue
                    
                    X.append(input_ids)
                    y.append(target)
                    
                except json.JSONDecodeError:
                    invalid_count += 1
                    continue
                    
    except Exception as e:
        raise Exception(f"Error loading {file_path}: {e}")
    
    if invalid_count > 0:
        print(f"⚠️  Skipped {invalid_count} invalid samples from {os.path.basename(file_path)}")
    
    print(f"📊 Loaded {len(X):,} samples from {os.path.basename(file_path)}")
    return np.array(X, dtype=np.int32), np.array(y, dtype=np.int32)

def analyze_data_distribution(y_train, y_val, y_test):
    """Analyze and print data distribution."""
    print("\n📊 DATA DISTRIBUTION:")
    print("-" * 30)
    
    def print_dist(y, name):
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        print(f"{name}:")
        for label, count in zip(unique, counts):
            label_name = "vulnerable" if label == 1 else "non-vulnerable"
            print(f"  {label_name}: {count:,} ({count/total:.1%})")
    
    print_dist(y_train, "Training")
    print_dist(y_val, "Validation")
    print_dist(y_test, "Test")

class WarmupScheduler:
    """Learning rate scheduler with warmup."""
    
    def __init__(self, optimizer, d_model, warmup_steps=1000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(self.step_num ** (-0.5), 
                                          self.step_num * self.warmup_steps ** (-1.5))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

def get_class_weights(y_train):
    """Compute class weights for imbalanced dataset."""
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
    weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"⚖️  Class weights: {dict(enumerate(class_weights))}")
    return weights

def train_epoch(model, dataloader, criterion, optimizer, scheduler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item()
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(dataloader), 100. * correct / total

def evaluate_epoch(model, dataloader, criterion):
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            probabilities = torch.softmax(outputs, dim=1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of vulnerable class
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    return (total_loss / len(dataloader), 
            100. * correct / total,
            np.array(all_labels),
            np.array(all_predictions),
            np.array(all_probabilities))

def plot_training_history(train_losses, val_losses, train_accs, val_accs, model_dir):
    """Plot and save training history."""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save history data
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    }
    
    with open(os.path.join(model_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)

def evaluate_model(model, test_loader, model_dir):
    """Comprehensive model evaluation."""
    print("\n🔍 MODEL EVALUATION:")
    print("=" * 50)
    
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids)
            predicted = torch.argmax(outputs, dim=1)
            probabilities = torch.softmax(outputs, dim=1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, 
                              target_names=["non-vulnerable", "vulnerable"]))
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_predictions)
    print(cm)
    
    # ROC AUC and Precision-Recall
    try:
        # ROC Curve
        auc_score = roc_auc_score(all_labels, all_probabilities)
        print(f"\nROC AUC Score: {auc_score:.4f}")
        
        fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(all_labels, all_probabilities)
        pr_auc = np.trapz(precision, recall)
        
        # Plot both curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # ROC Curve
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend(loc="lower right")
        ax1.grid(True)
        
        # Precision-Recall Curve
        ax2.plot(recall, precision, color='darkgreen', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc="lower left")
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'evaluation_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"⚠️  Could not compute evaluation metrics: {e}")
    
    # Save predictions
    results = {
        'y_true': all_labels.tolist(),
        'y_pred': all_predictions.tolist(),
        'y_pred_proba': all_probabilities.tolist(),
        'metrics': {
            'roc_auc': float(auc_score) if 'auc_score' in locals() else None,
            'pr_auc': float(pr_auc) if 'pr_auc' in locals() else None
        }
    }
    
    with open(os.path.join(model_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

def save_model(model, model_dir, filename="best_model.pth"):
    """Save model checkpoint."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': model.embedding.num_embeddings,
            'd_model': D_MODEL,
            'heads': HEADS,
            'layers': LAYERS,
            'max_len': MAX_LEN,
            'num_classes': 2
        }
    }, os.path.join(model_dir, filename))

def load_model(model_path, vocab_size):
    """Load model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['model_config']
    
    model = TransformerClassifier(
        vocab_size=vocab_size,
        d_model=config['d_model'],
        heads=config['heads'],
        layers=config['layers'],
        max_len=config['max_len'],
        num_classes=config['num_classes']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def main():
    print("Multi-Head Self-Attention Training")
    print("=" * 60)
    
    # Environment info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    # Validate files
    validate_data_files()
    
    # Load vocabulary
    try:
        with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        vocab_size = len(vocab)
        print(f"✅ Loaded vocabulary size: {vocab_size:,}")
        
        if vocab_size > 50000:
            print(f"⚠️  Large vocabulary size ({vocab_size:,}) may cause memory issues")
            
    except Exception as e:
        raise Exception(f"Error loading vocabulary: {e}")
    
    # Load data
    train_file = os.path.join(ENCODED_DIR, "train.jsonl")
    val_file = os.path.join(ENCODED_DIR, "val.jsonl")
    test_file = os.path.join(ENCODED_DIR, "test.jsonl")

    X_train, y_train = load_data(train_file)
    X_val, y_val = load_data(val_file)
    X_test, y_test = load_data(test_file)
    
    # Analyze data distribution
    analyze_data_distribution(y_train, y_val, y_test)
    
    # Create datasets and data loaders
    train_dataset = VulnerabilityDataset(X_train, y_train)
    val_dataset = VulnerabilityDataset(X_val, y_val)
    test_dataset = VulnerabilityDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Get class weights
    class_weights = get_class_weights(y_train)
    
    # Build model
    print(f"\n🏗️  Building Transformer model...")
    print(f"Architecture: {LAYERS} layers, {HEADS} heads, {D_MODEL} d_model")
    
    model = TransformerClassifier(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        heads=HEADS,
        layers=LAYERS,
        max_len=MAX_LEN,
        num_classes=2
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler with warmup
    scheduler = WarmupScheduler(optimizer, D_MODEL, WARMUP_STEPS)
    
    # Training loop
    print(f"\n🎯 Starting training...")
    print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}")
    
    best_val_acc = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    patience = 5
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 50)
        
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler)
        
        # Validation
        val_loss, val_acc, _, _, _ = evaluate_epoch(model, val_loader, criterion)
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, MODEL_DIR, "best_model.pth")
            print(f"💾 New best model saved (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs, MODEL_DIR)
    
    # Load best model for evaluation
    best_model_path = os.path.join(MODEL_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"\n📥 Loading best model from {best_model_path}")
        model = load_model(best_model_path, vocab_size)
    
    # Final evaluation
    evaluate_model(model, test_loader, MODEL_DIR)
    
    # Save final model
    save_model(model, MODEL_DIR, "final_model.pth")
    print(f"\n💾 Final model saved to: {os.path.join(MODEL_DIR, 'final_model.pth')}")
    
    print(f"\n✅ Training completed!")
    print(f"📁 All outputs saved to: {MODEL_DIR}")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise
