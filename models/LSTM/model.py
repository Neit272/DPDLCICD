import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import pickle

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
ENCODED_DIR = os.path.join(ROOT_DIR, "data", "preprocessed", "encoded")
VOCAB_PATH = os.path.join(ROOT_DIR, "data", "preprocessed", "vocab", "vocab.json")
MODEL_DIR = os.path.join(ROOT_DIR, "models", "LSTM")
os.makedirs(MODEL_DIR, exist_ok=True)

# Improved Hyperparameters
EMBEDDING_DIM = 128  # Increased
MAX_LEN = 256
LSTM_UNITS = 64      # Reduced to prevent overfitting
DROPOUT_RATE = 0.3   # Reduced
BATCH_SIZE = 32      # Increased for better gradient estimates
EPOCHS = 10          # Increased with early stopping
LEARNING_RATE = 0.001

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

def build_improved_lstm_model(vocab_size, embedding_dim, max_len, lstm_units, learning_rate):
    """Build improved LSTM model with better architecture."""
    model = Sequential([
        # Embedding layer
        Embedding(
            input_dim=vocab_size, 
            output_dim=embedding_dim, 
            input_length=max_len,
            mask_zero=True,  # Handle padding
            name='embedding'
        ),
        
        # Bidirectional LSTM for better context
        tf.keras.layers.Bidirectional(
            LSTM(units=lstm_units, return_sequences=True, dropout=DROPOUT_RATE),
            name='bidirectional_lstm_1'
        ),
        
        # Second LSTM layer
        LSTM(units=lstm_units//2, return_sequences=False, dropout=DROPOUT_RATE, name='lstm_2'),
        
        # Batch normalization
        BatchNormalization(name='batch_norm'),
        
        # Dense layers with residual-like connection
        Dense(128, activation='relu', name='dense_1'),
        Dropout(DROPOUT_RATE, name='dropout_1'),
        
        Dense(64, activation='relu', name='dense_2'),
        Dropout(DROPOUT_RATE, name='dropout_2'),
        
        # Output layer
        Dense(1, activation='sigmoid', name='output')
    ])
    
    # Custom optimizer
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def setup_callbacks(model_dir):
    """Setup training callbacks."""
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(model_dir, "best_model.h5"),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        ),
        
        EarlyStopping(
            monitor='val_loss',
            patience=7,  # Increased patience
            restore_best_weights=True,
            verbose=1
        ),
        
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    return callbacks

def plot_training_history(history, model_dir):
    """Plot and save training history."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    
    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Training Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
    
    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Training Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save history
    with open(os.path.join(model_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)

def evaluate_model(model, X_test, y_test, model_dir):
    """Comprehensive model evaluation."""
    print("\n🔍 MODEL EVALUATION:")
    print("=" * 50)
    
    # Predictions
    y_pred_proba = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["non-vulnerable", "vulnerable"]))
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # ROC AUC
    try:
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC AUC Score: {auc_score:.4f}")
        
        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(model_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"⚠️  Could not compute ROC AUC: {e}")
    
    # Save predictions
    results = {
        'y_true': y_test.tolist(),
        'y_pred': y_pred.tolist(),
        'y_pred_proba': y_pred_proba.flatten().tolist()
    }
    
    with open(os.path.join(model_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

def main():
    print("🚀 LSTM Vulnerability Detection Training")
    print("=" * 50)
    
    # Environment info
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Devices: {tf.config.list_physical_devices()}")
    
    # Validate files
    validate_data_files()
    
    # Load vocabulary
    try:
        with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        vocab_size = len(vocab)
        print(f"✅ Loaded vocabulary size: {vocab_size:,}")
        
        # Warning for large vocab
        if vocab_size > 20000:
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
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"\n⚖️  Class weights: {class_weight_dict}")

    # Build model
    print(f"\n🏗️  Building LSTM model...")
    model = build_improved_lstm_model(vocab_size, EMBEDDING_DIM, MAX_LEN, LSTM_UNITS, LEARNING_RATE)
    model.summary()
    
    # Setup callbacks
    callbacks = setup_callbacks(MODEL_DIR)
    
    # Training
    print(f"\n🎯 Starting training...")
    print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history, MODEL_DIR)
    
    # Load best model for evaluation
    best_model_path = os.path.join(MODEL_DIR, "best_model.h5")
    if os.path.exists(best_model_path):
        print(f"\n📥 Loading best model from {best_model_path}")
        model = tf.keras.models.load_model(best_model_path)
    
    # Evaluation
    evaluate_model(model, X_test, y_test, MODEL_DIR)
    
    # Save final model
    final_model_path = os.path.join(MODEL_DIR, "final_model.h5")
    model.save(final_model_path)
    print(f"\n💾 Final model saved to: {final_model_path}")
    
    print(f"\n✅ Training completed!")
    print(f"📁 All outputs saved to: {MODEL_DIR}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise
